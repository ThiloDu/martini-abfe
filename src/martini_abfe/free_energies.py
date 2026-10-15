"""Calculate free energies using MBAR."""

import logging
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logging.getLogger("pymbar").setLevel(logging.ERROR)
logging.getLogger("jax").setLevel(logging.INFO)
import MDAnalysis as mda
from alchemlyb.estimators import MBAR
from alchemlyb.parsing.gmx import extract_u_nk
from alchemlyb.preprocessing.subsampling import decorrelate_u_nk
from MDAnalysis.lib.distances import minimize_vectors
from pymbar.timeseries import statistical_inefficiency

from .utils import (
    local_path,
    read_json_file,
    write_json_file,
)

logger = logging.getLogger(__name__)

k_B = 8.314 / 1000  # kJ/mol/K
V0 = 1.6605  # nm^3 (Standard volume for 1 M)
kT_to_kcal = 0.5962


def standard_state_correction(system_directory: Path) -> float | None:
    """
    Analytical standard state correction for the free energy of binding. The correction
    is given by -kT ln(V0/V), where V0 is the standard state volume (1 mol/L) and V is
    the volume of the flat-bottom restraint.
    :param system_directory: Directory where the system files are located.
    :return: Correction in kcal/mol to be added to the free energy
    """
    # Load restraint settings from the mdp file
    mdp_file = system_directory / "lambda-000" / "production.out.mdp"
    if not mdp_file.exists():
        return None
    settings = {}
    for line in mdp_file.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.split(";")[0].partition("=")
        if separator:
            settings[key.strip().lower().replace("_", "-")] = value.strip().lower()
    # Determine restraint type and radius
    if settings.get("pull") != "yes":
        return None
    supported_types = ["flat-bottom", "umbrella"]
    if (
        settings.get("pull-coord1-geometry") != "distance"
        or settings.get("pull-coord1-type") not in supported_types
        or settings.get("pull-coord1-start", "no") != "no"
    ):
        logger.warning("Unsupported pull setup in %s", local_path(mdp_file))
        return None
    K = float(settings.get("pull-coord1-k", 0.0))  # kJ/mol/nm^2
    flat_bottom = settings.get("pull-coord1-type") == "flat-bottom"
    r_min = float(settings.get("pull-coord1-init", 0.0))  # nm
    T = float(settings.get("ref-t", "300").split()[0])  # K
    r_0 = r_min if flat_bottom else 0.0  # nm
    # Calculate the volume of the restraint
    k = K / (2 * T * k_B)  # 1/nm^2, factor 2 because K is force constant
    i1 = 2 * np.pi * (r_min + r_0) / k * np.exp(-k * (r_0 - r_min) ** 2)
    i2 = (
        (np.pi / k) ** (3 / 2)
        * (1 + 2 * k * r_min**2)
        * math.erfc(np.sqrt(k) * (r_0 - r_min))
    )
    integral = i1 + i2
    if flat_bottom:
        integral += 4 / 3 * np.pi * r_min**3
    # Calculate the standard state correction
    dG = k_B * T * np.log(integral / V0)  # kJ/mol
    return dG * 0.239006  # kcal/mol


def flat_bottom_correction(system_directory: Path) -> tuple[float, float] | None:
    """
    Free-energy correction for the flat-bottom restraint holding the bound state. The
    restraint biases the estimate whenever the ligand pushes against its wall, which is
    removed by reweighting the bound state to a square well of the same radius: discarding
    the samples at the wall shifts the free energy by kT ln P(inside).
    :param system_directory: Directory where the system files are located.
    :return: Correction in kcal/mol to be added to the free energy and its uncertainty,
        or None if it cannot be computed.
    """
    # Check if required files exist
    mdp_file = system_directory / "lambda-000" / "production.out.mdp"
    px_file = system_directory / "lambda-000" / "production.px.xvg"
    if not mdp_file.exists() or not px_file.exists():
        return None
    # Radius of the flat bottom, given by the unshifted reference of a distance restraint
    settings = {}
    for line in mdp_file.read_text(encoding="utf-8").splitlines():
        key, separator, value = line.split(";")[0].partition("=")
        if separator:
            settings[key.strip().lower().replace("_", "-")] = value.strip().lower()
    if (
        settings.get("pull-coord1-type") != "flat-bottom"
        or settings.get("pull-coord1-geometry") != "distance"
        or settings.get("pull-coord1-start", "no") != "no"
        or settings.get("pull-print-ref-value", "no") != "no"
        or settings.get("pull-print-components", "no") != "no"
    ):
        logger.info(
            "Unsupported pull setup in %s, no flat-bottom correction",
            local_path(mdp_file),
        )
        return None
    radius = float(settings["pull-coord1-init"])
    # Fraction of the equilibrated bound-state samples that stay inside the flat bottom
    pull = np.loadtxt(px_file, comments=["#", "@"], ndmin=2)
    if pull.shape[0] < 2 or pull.shape[1] < 2:
        logger.warning("No pull coordinate found in %s", local_path(px_file))
        return None
    indicator = (pull[int(pull.shape[0] * 0.05) :, 1] <= radius).astype(float)
    inside = float(np.mean(indicator))
    if inside == 0.0:
        logger.warning(
            "No sample in %s stays inside the flat bottom", local_path(px_file)
        )
        return None
    if inside == 1.0:
        return 0.0, 0.0
    # Error of kT ln(P) from the binomial variance of P, using the effective number of samples
    n_eff = len(indicator) / float(statistical_inefficiency(indicator, fast=True))
    uncertainty = np.sqrt((1 - inside) / (inside * n_eff))
    logger.info(
        "Flat-bottom correction for %s: %d/%d samples at the wall, %.0f effective",
        local_path(system_directory),
        round((1 - inside) * len(indicator)),
        len(indicator),
        n_eff,
    )
    # Convert to kcal/mol
    return float(np.log(inside)) * kT_to_kcal, float(uncertainty) * kT_to_kcal


def run_mbar_calculations(
    system_directory: Path,
) -> tuple[float, float, np.ndarray] | None:
    """
    Run mbar free energy calculations.
    :param system_directory: Directory where the system files are located.
    :return: Free energy, its uncertainty and changes per lambda step, or None if not all
        lambda simulations are finished.
    """
    logger.info(
        "Running mbar calculations in directory: %s", local_path(system_directory)
    )
    # Get list of xvg files and count of finished lambda simulations
    xvg_files = []
    count = 0
    for directory in system_directory.glob("lambda-*"):
        if directory.is_dir() and (directory / "production.tpr").exists():
            count += 1
        if (directory / "production.xvg").exists():
            xvg_files.append(directory / "production.xvg")
        elif (directory / "production.pkl.gz").exists():
            xvg_files.append(directory / "production.pkl.gz")
    if count == 0 or len(xvg_files) != count:
        logger.info(
            "Skipping free energy calculation for %s, found only %d/%d xvg files",
            local_path(system_directory),
            len(xvg_files),
            count,
        )
        return
    # Load u_nk from xvg files or pickles
    u_nk_list = []
    xvg_files = sorted(xvg_files)
    for f in xvg_files:
        logger.debug("Processing file for mbar: %s", local_path(f))
        if f.suffix == ".xvg":
            u_nk_list.append(extract_u_nk(f, T=300))
        else:
            u_nk_list.append(pd.read_pickle(f))
    # Decorrelate each window by the ligand pocket distance if a pocket dummy bead is present
    usim = mda.Universe(str(system_directory / "lambda-000" / "production.tpr"))
    dummy = usim.select_atoms("resname DUM")
    ligand = usim.select_atoms("resname LIG")
    u_nk_list_red = []
    for xvg_file, u_nk in zip(xvg_files, u_nk_list):
        equilibrated = u_nk[int(len(u_nk) * 0.05) :]
        xtc = xvg_file.parent / "production.xtc"
        if not xtc.exists() or len(dummy) == 0 or len(ligand) == 0:
            u_nk_list_red.append(decorrelate_u_nk(equilibrated))
            continue
        usim.load_new(str(xtc))
        distances = np.empty(len(usim.trajectory))
        for i, _ in enumerate(usim.trajectory):
            box = usim.dimensions
            vectors = minimize_vectors(ligand.positions - dummy.positions[0], box=box)
            distances[i] = np.linalg.norm(vectors.mean(axis=0))
        distances = distances[int(len(distances) * 0.05) :]
        pose_tau = statistical_inefficiency(distances, fast=True) * usim.trajectory.dt
        dt = np.median(np.diff(equilibrated.index.get_level_values("time").values))
        stride = int(np.clip(round(pose_tau / dt), 1, max(1, len(equilibrated) // 10)))
        u_nk_list_red.append(equilibrated.iloc[::stride])
    # Combine all windows and run MBAR
    u_nk_combined = pd.concat(u_nk_list_red)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mbar = MBAR().fit(u_nk_combined)
    free_energy = float(mbar.delta_f_.iloc[0, -1]) * kT_to_kcal  # Convert to kcal/mol
    d_free_energy = float(mbar.d_delta_f_.iloc[0, -1]) * kT_to_kcal
    changes = np.diag(mbar.delta_f_.values, k=1) * kT_to_kcal
    # Save the reduced u_nk to pickles and remove the xvg files to save disk space
    for xvg_file, data in zip(xvg_files, u_nk_list):
        if xvg_file.suffix == ".xvg":
            data.to_pickle(xvg_file.with_suffix(".pkl.gz"))
            xvg_file.unlink()
    return free_energy, d_free_energy, changes


def calculate_free_energies(ligand_directory: Path, systems: list[str]):
    """
    Calculate free energies for ligand in specified system. Expects that all lambda
    simulations have been completed.
    :param ligand_directory: Directory where the ligand files are located.
    :param systems: List of systems for which to calculate free energies.
    """
    ligand_info_path = ligand_directory / "info.json"
    info = read_json_file(ligand_info_path)
    process_systems = []
    for system in systems:
        if system not in info and (ligand_directory / system).is_dir():
            process_systems.append(system)
    try:
        for system in process_systems:
            # Run MBAR calculations for the system
            dg_result = run_mbar_calculations(ligand_directory / system)
            if dg_result is None:
                raise RuntimeError(
                    f"Free energy calculation for {info['name']} in {system} failed."
                )
            info[system] = {
                "dG": dg_result[0],
                "ddG": dg_result[1],
                "dG_mbar": dg_result[0],
                "ddG_mbar": dg_result[1],
                "changes": dg_result[2].tolist(),
            }
            # Determine if a flat-bottom restraint correction is needed and apply it
            correction = flat_bottom_correction(ligand_directory / system)
            if correction is not None:
                info[system]["dG_restraint"] = correction[0]
                info[system]["ddG_restraint"] = correction[1]
                info[system]["dG"] += correction[0]
                info[system]["ddG"] = np.sqrt(
                    info[system]["ddG"] ** 2 + correction[1] ** 2
                )
            # Calculate analytical standard state correction
            stst_correction = standard_state_correction(ligand_directory / system)
            if stst_correction is not None:
                info[system]["dG_standard_state"] = stst_correction
                info[system]["dG"] += stst_correction
            logger.info(
                "Calculated free energy for %s in %s: %.3f kcal/mol",
                info["name"],
                system,
                dg_result[0],
            )
    finally:
        if len(process_systems) > 0:
            write_json_file(info, ligand_info_path, compact_lists=True)


def _mean_and_error(values: list[float], errors: list[float]) -> tuple[float, float]:
    """
    Average free energies of independent repeats.
    :param values: Free energy of every repeat.
    :param errors: MBAR uncertainty of every repeat.
    :return: Mean and its standard error, or the MBAR uncertainty for a single repeat.
    """
    if len(values) == 1:
        return values[0], errors[0]
    return float(np.mean(values)), float(np.std(values) / np.sqrt(len(values)))


def binding_free_energy(
    ligand_directory: Path, bound: str = "protein", unbound: str = "water"
) -> dict | None:
    """
    Combine the decoupling free energies of all repeats into the binding free energy,
    dG_bind = dG_unbound - dG_bound, where the bound leg already contains the restraint
    and standard state corrections.
    :param ligand_directory: Directory where the ligand files are located.
    :param bound: Name of the bound leg in the configuration.
    :param unbound: Name of the unbound leg in the configuration.
    :return: Free energies of both legs and of binding in kcal/mol, or None if a leg is
        missing.
    """
    info = read_json_file(ligand_directory / "info.json")
    result = {"name": info["name"]}
    for leg in (bound, unbound):
        keys = [k for k in info if k == leg or k.startswith(f"{leg}-r")]
        keys = [k for k in keys if isinstance(info[k], dict) and "dG" in info[k]]
        if len(keys) == 0:
            return None
        dG, ddG = _mean_and_error(
            [info[k]["dG"] for k in keys], [info[k]["ddG"] for k in keys]
        )
        result[leg] = {"dG": dG, "ddG": ddG, "n_repeats": len(keys)}
    result["dG_bind"] = result[unbound]["dG"] - result[bound]["dG"]
    result["ddG_bind"] = float(
        np.sqrt(result[unbound]["ddG"] ** 2 + result[bound]["ddG"] ** 2)
    )
    result["units"] = "kcal/mol"
    write_json_file(result, ligand_directory / "binding.json")
    return result
