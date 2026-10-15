"""Perform simulations for protein-ligand and water-ligand systems."""

import asyncio
import logging
import shutil
import warnings
from pathlib import Path
from subprocess import CalledProcessError
from time import monotonic

import mdtraj
import numpy as np
from omegaconf import DictConfig, OmegaConf

from .coby import setup_protein_box, setup_solvent_box
from .free_energies import calculate_free_energies
from .simulation import Simulation
from .utils import (
    check_file_exists,
    create_gromacs_index,
    get_n_lambda_steps_from_mdp,
    local_path,
    read_json_file,
)

logger = logging.getLogger(__name__)


def rerun_keys(sim_config: OmegaConf, name: str) -> list[str]:
    """
    List the directory keys of a system, one per repeat if it is simulated several times.
    :param sim_config: Simulation configuration.
    :param name: Name of the system in the configuration.
    :return: One key per repeat, or just the system name if it is simulated once.
    """
    n_reruns = sim_config[name].get("n_reruns", 1)
    if n_reruns > 1:
        return [f"{name}-r{rerun + 1}" for rerun in range(n_reruns)]
    return [name]


def _format_counts(composition: dict[str, int]) -> str:
    """
    Format molecule counts for the log, e.g. 'W=1354, NA=15, CL=16'.
    :param composition: Mapping of residue name -> molecule count.
    :return: Comma separated counts.
    """
    return ", ".join(f"{res}={count}" for res, count in composition.items())


def _log_production_start(label: str, step: int, n_lambda: int, sim_config: OmegaConf):
    """
    Log a lambda window that is run locally, queued windows are only summarized.
    :param label: System and leg, e.g. '1_cis/protein-r1'.
    :param step: Index of the lambda window.
    :param n_lambda: Number of lambda windows.
    :param sim_config: Simulation configuration.
    """
    if sim_config.queue.type == "none":
        logger.info("%s: running lambda window %d/%d", label, step + 1, n_lambda)
    else:
        logger.debug("%s: queueing lambda window %d/%d", label, step + 1, n_lambda)


def _log_production_summary(
    label: str, n_lambda: int, window_states: dict[str, int], sim_config: OmegaConf
):
    """
    Log how many lambda windows were submitted, were already queued or are finished.
    :param label: System and leg, e.g. '1_cis/protein-r1'.
    :param n_lambda: Number of lambda windows.
    :param window_states: Number of windows per state.
    :param sim_config: Simulation configuration.
    """
    action = "run" if sim_config.queue.type == "none" else "queued"
    states = ", ".join(
        f"{count} {action if state == 'submitted' else state}"
        for state, count in window_states.items()
        if count > 0
    )
    logger.info("%s: %d lambda windows (%s)", label, n_lambda, states)


async def simulate_ligand_in_solvent(
    name: str, ligand_directory: Path, sim_config: OmegaConf, key: str = None
):
    """
    Simulate ligand in solvent.
    :param name: Name of the solvent (e.g. "water", "octanol").
    :param ligand_directory: Directory where the ligand files are located.
    :param sim_config: Simulation configuration.
    :param key: Key for the solvent in the info file and folder name, defaults to its name.
    :return: Number of lambda windows, or None if the simulation already exists.
    """
    # Check if structure and topology files exist
    key = name if key is None else key
    ligand_gro = ligand_directory / "ligand.gro"
    ligand_top = ligand_directory / "ligand.itp"
    ligand_info = ligand_directory / "info.json"
    check_file_exists(ligand_gro, ligand_top, ligand_info)
    # Read info file
    info = read_json_file(ligand_info)
    # Check if simulation has already been performed
    if key in info:
        logger.info("Skipping %s simulation for %s, already exists", key, info["name"])
        return
    label = f"{ligand_directory.name}/{key}"
    logger.info("%s: setting up (ligand charge %s)", label, info["charge"])
    # Setup system directory
    system_directory = ligand_directory / key
    system_directory.mkdir(parents=True, exist_ok=True)
    # Setup simulation box with COBY
    system_top = system_directory / "system.top"
    system_gro = system_directory / "system.gro"
    if not system_top.exists() or not system_gro.exists():
        include_lines = [
            f'#include "{sim_config.martini[imp]}"' for imp in sim_config.martini
        ]
        include_lines.append('#include "../ligand.itp"')
        mixture = sim_config[name].get("composition", {"W": 1.0})
        charge = info["charge"] if sim_config[name].get("neutralize", True) else 0
        logger.info("%s: building solvent box with COBY", label)
        composition = setup_solvent_box(
            ligand_gro=ligand_gro,
            ligand_top=ligand_top,
            system_gro=system_gro,
            system_top=system_top,
            box_margin=sim_config[name].box_margin,
            charge=charge,
            salt_concentration=sim_config[name].get("salt", 0.0),
            include_lines=include_lines,
            mixture=mixture,
        )
        logger.info("%s: box built (%s)", label, _format_counts(composition))
        check_file_exists(
            system_top,
            system_gro,
            message="COBY did not produce expected output files.",
        )
    # Create index file
    system_ndx = system_directory / "system.ndx"
    if not system_ndx.exists():
        create_gromacs_index(
            system_gro,
            system_ndx,
            selections={
                "system": lambda df: df,
                "ligand": lambda df: df[df["resname"] == "LIG"],
                "other": lambda df: df[df["resname"] != "LIG"],
            },
        )
        logger.debug("Created index file for %s: %s", info["name"], system_ndx)
    # Run minimization
    minimization = Simulation(
        sim_config[name].minimization,
        system_gro,
        system_top,
        system_directory / "minimization.tpr",
        system_directory / "minimization.run.log",
        sim_config=sim_config,
    )
    if not Path(minimization.out_coord).exists():
        logger.info(
            "%s: running minimization (%s)",
            label,
            local_path(minimization.input_files["log"]),
        )
        minimization.run(ignore_queue=True, extra_args=sim_config.get("gromacs", ""))
    # Run equilibration
    equilibration = Simulation(
        sim_config[name].equilibration,
        minimization.out_coord,
        system_top,
        system_directory / "equilibration.tpr",
        system_directory / "equilibration.run.log",
        system_ndx,
        sim_config=sim_config,
    )
    if not Path(equilibration.out_coord).exists():
        logger.info(
            "%s: running equilibration (%s)",
            label,
            local_path(equilibration.input_files["log"]),
        )
        equilibration.run(ignore_queue=True, extra_args=sim_config.get("gromacs", ""))
    # Determine number of lambda steps and copy simulation config file
    mdp_file = sim_config[name].production
    if info["charged"] and "production_charged" in sim_config[name]:
        mdp_file = sim_config[name].production_charged
    n_lambda = get_n_lambda_steps_from_mdp(mdp_file)
    production_mdp = system_directory / "production.mdp"
    shutil.copy(mdp_file, production_mdp)
    # Setup production for each lambda step
    window_states = {"submitted": 0, "already queued": 0, "finished": 0}
    for step in range(n_lambda):
        lambda_dir = system_directory / f"lambda-{step:03d}"
        lambda_dir.mkdir(parents=True, exist_ok=True)
        lambda_out_tpr = lambda_dir / "production.tpr"
        production = Simulation(
            production_mdp,
            equilibration.out_coord,
            system_top,
            lambda_out_tpr,
            lambda_dir / "production.run.log",
            system_ndx,
            sim_config=sim_config,
        )
        if Path(production.out_coord).exists():
            window_states["finished"] += 1
            continue
        if not lambda_out_tpr.exists() or sim_config.queue.readd:
            window_states["submitted"] += 1
            mdp_modifications = {"init-lambda-state": step}
            if sim_config[name].get(f"xout{step}", None) is not None:
                xout_value = sim_config[name][f"xout{step}"]
                mdp_modifications["nstxout-compressed"] = xout_value
            elif sim_config[name].get("xout", None) is not None:
                mdp_modifications["nstxout-compressed"] = sim_config[name].xout
            production.modify_mdp(mdp_modifications)
            _log_production_start(label, step, n_lambda, sim_config)
            # Submit to queue
            production.run(ignore_queue=False, extra_args=sim_config.get("gromacs", ""))
        else:
            window_states["already queued"] += 1
    _log_production_summary(label, n_lambda, window_states, sim_config)
    return n_lambda


async def simulate_ligand_with_protein(
    ligand_directory: Path, sim_config: OmegaConf, key: str = "protein"
):
    """
    Simulate ligand in protein pocket.
    :param ligand_directory: Directory where the ligand files are located.
    :param sim_config: Simulation configuration.
    :param key: Key for the protein in the info file and folder name.
    :return: Number of lambda windows, or None if the simulation already exists.
    """
    # Check if structure and topology files exist
    ligand_gro = ligand_directory / "ligand.gro"
    ligand_top = ligand_directory / "ligand.itp"
    ligand_info = ligand_directory / "info.json"
    check_file_exists(ligand_gro, ligand_top, ligand_info)
    # Read info file
    info = read_json_file(ligand_info)
    # Check if simulation has already been performed
    if key in info:
        logger.info("Skipping %s simulation for %s, already exists", key, info["name"])
        return
    label = f"{ligand_directory.name}/{key}"
    logger.info("%s: setting up (ligand charge %s)", label, info["charge"])
    # Read protein charge from file
    protein_info = read_json_file(Path(sim_config.directory) / "protein_info.json")
    total_charge = info["charge"] + protein_info["charge"]
    logger.debug(
        "Total system charge for %s with %s: %s", info["name"], key, total_charge
    )
    # Setup system directory
    system_directory = ligand_directory / key
    system_directory.mkdir(parents=True, exist_ok=True)
    # Place ligand in pocket
    ligand_mol = mdtraj.load(ligand_gro)
    protein_ligand_pdb = system_directory / "system.pdb"
    ligand_pdb = system_directory / "ligand.pdb"
    if not protein_ligand_pdb.exists():
        # The ligand structure is placed in the pocket during the preparation, optionally
        # only its centroid is moved to the position of the dummy atom
        if sim_config.protein.get("recenter_ligand", False):
            ligand_mol.xyz = (
                ligand_mol.xyz
                - ligand_mol.xyz.mean(axis=1)
                + np.array(protein_info["ligand_com"])
            )
        ligand_mol.xyz = ligand_mol.xyz * 10  # Convert to Angstrom
        ligand_writer = mdtraj.formats.PDBTrajectoryFile(ligand_pdb, "w")
        ligand_writer.write(ligand_mol.xyz[0], ligand_mol.topology)
        ligand_writer.close()
        logger.debug("Wrote ligand pdb file: %s", ligand_pdb)
        # Combine protein (including the dummy bead) and ligand into single pdb file, in
        # the order of the molecules in the topology
        protein_pdb = Path(sim_config.directory) / "protein-cg.pdb"
        with open(protein_pdb, "r", encoding="utf-8") as protein_pdb_file:
            protein_pdb_lines = [
                line
                for line in protein_pdb_file
                if line.startswith(("CRYST1", "ATOM", "TER"))
            ]
        with open(ligand_pdb, "r", encoding="utf-8") as ligand_pdb_file:
            ligand_pdb_lines = [
                line for line in ligand_pdb_file if line.startswith(("ATOM", "HETATM"))
            ]
        with open(protein_ligand_pdb, "w", encoding="utf-8") as protein_ligand_pdb_file:
            protein_ligand_pdb_file.writelines(protein_pdb_lines)
            protein_ligand_pdb_file.writelines(ligand_pdb_lines)
            protein_ligand_pdb_file.write("TER\nEND\n")
        logger.debug("Wrote combined protein-ligand pdb file: %s", protein_ligand_pdb)
    # Setup simulation box with COBY
    system_top = system_directory / "system.top"
    system_gro = system_directory / "system.gro"
    if not system_top.exists() or not system_gro.exists():
        include_lines = [
            f'#include "{sim_config.martini.topology}"',
            "\n[ atomtypes ]\nDUM 0.0 0.000 V 0.0 0.0\n",
        ]
        include_lines += [
            f'#include "{sim_config.martini[imp]}"'
            for imp in sim_config.martini
            if imp != "topology"
        ]
        include_lines.append('#include "../ligand.itp"')
        if (ligand_directory / "protein.itp").exists():
            include_lines.append('#include "../protein.itp"')
        else:
            include_lines.append('#include "../../protein.itp"')
        charge = total_charge if sim_config.protein.get("neutralize", True) else 0
        logger.info("%s: solvating protein-ligand complex with COBY", label)
        composition = setup_protein_box(
            protein_ligand_pdb=protein_ligand_pdb,
            system_gro=system_gro,
            system_top=system_top,
            box_margin=sim_config.protein.box_margin,
            charge=charge,
            include_lines=include_lines,
            protein_name=protein_info["name"],
            salt_concentration=sim_config.protein.get("salt", None) or 0.0,
        )
        logger.info("%s: box built (%s)", label, _format_counts(composition))
        check_file_exists(
            system_top,
            system_gro,
            message="COBY did not produce expected output files.",
        )
    # Create index file
    system_ndx = system_directory / "system.ndx"
    if not system_ndx.exists():
        create_gromacs_index(
            system_gro,
            system_ndx,
            selections={
                "ligand": lambda df: df[df["resname"] == "LIG"],
                "dummy": lambda df: df[df["resname"] == "DUM"],
                "Protein": lambda df: df[
                    (df["resname"] != "LIG")
                    & (df["resname"] != "W")
                    & (df["resname"] != "NA")
                    & (df["resname"] != "NA+")
                    & (df["resname"] != "CL")
                    & (df["resname"] != "CL-")
                ],
                "other": lambda df: df[
                    (df["resname"] == "W")
                    | (df["resname"] == "NA")
                    | (df["resname"] == "NA+")
                    | (df["resname"] == "CL")
                    | (df["resname"] == "CL-")
                ],
            },
        )
        logger.debug("Created index file for %s: %s", info["name"], system_ndx)
    # Determine bead index which is closest to the ligand COM (for GROMACS pbc treatment)
    system = mdtraj.load(system_gro)
    ligand_indices = system.topology.select("resname LIG")
    ligand_com = system.xyz[0, ligand_indices].mean(axis=0)
    ligand_bead_distances = np.linalg.norm(system.xyz[0] - ligand_com, axis=1)
    closest_bead_idx = np.argmin(ligand_bead_distances) + 1  # 1-indexed
    # Run minimization
    minimization = Simulation(
        sim_config.protein.minimization,
        system_gro,
        system_top,
        system_directory / "minimization.tpr",
        system_directory / "minimization.run.log",
        sim_config=sim_config,
    )
    if not Path(minimization.out_coord).exists():
        logger.info(
            "%s: running minimization (%s)",
            label,
            local_path(minimization.input_files["log"]),
        )
        minimization.run(ignore_queue=True, extra_args=sim_config.get("gromacs", ""))
    # Run equilibration, with a copy of the mdp file as it is modified for the system
    equilibration_mdp = system_directory / "equilibration.mdp"
    shutil.copy(sim_config.protein.equilibration, equilibration_mdp)
    equilibration = Simulation(
        equilibration_mdp,
        minimization.out_coord,
        system_top,
        system_directory / "equilibration.tpr",
        system_directory / "equilibration.run.log",
        system_ndx,
        sim_config=sim_config,
    )
    if not Path(equilibration.out_coord).exists():
        equilibration.modify_mdp("pull-group1-pbcatom", closest_bead_idx)
        logger.info(
            "%s: running equilibration (%s)",
            label,
            local_path(equilibration.input_files["log"]),
        )
        equilibration.run(ignore_queue=True, extra_args=sim_config.get("gromacs", ""))
    # Get production mdp file based on ligand properties
    mdp_file = sim_config.protein.production
    if info["charged"] and "production_charged" in sim_config.protein:
        mdp_file = sim_config.protein.production_charged
    large_ligands = sim_config.protein.get("large_ligands", None)
    if large_ligands is not None and len(info["beads"]) >= large_ligands:
        mdp_file = sim_config.protein.get("production_large", mdp_file)
        if info["charged"] and "production_charged_large" in sim_config.protein:
            mdp_file = sim_config.protein.production_charged_large
        if mdp_file == sim_config.protein.production:
            warnings.warn(
                f"Ligand {info['name']} classified as large ligand, but no "
                "alternative production mdp file specified in config."
            )
    # Determine number of lambda steps and copy simulation config file
    n_lambda = get_n_lambda_steps_from_mdp(mdp_file)
    production_mdp = system_directory / "production.mdp"
    shutil.copy(mdp_file, production_mdp)
    # Run production for each lambda step
    run_mode = sim_config.protein.get("run_mode", "serial")
    logger.debug("Running production for %s in %s mode", info["name"], run_mode)
    window_states = {"submitted": 0, "already queued": 0, "finished": 0}
    starting_config = equilibration.out_coord
    for step in range(n_lambda - 1, -1, -1):
        lambda_dir = system_directory / f"lambda-{step:03d}"
        lambda_dir.mkdir(parents=True, exist_ok=True)
        lambda_out_tpr = lambda_dir / "production.tpr"
        production = Simulation(
            production_mdp,
            starting_config,
            system_top,
            lambda_out_tpr,
            lambda_dir / "production.run.log",
            system_ndx,
            sim_config=sim_config,
        )
        if Path(production.out_coord).exists():
            window_states["finished"] += 1
        else:
            if not lambda_out_tpr.exists() or sim_config.queue.readd:
                window_states["submitted"] += 1
                production.modify_mdp(
                    {
                        "init-lambda-state": step,
                        "pull-group1-pbcatom": closest_bead_idx,
                        "pull-nstxout": 100 if step == 0 else 0,
                    }
                )
                _log_production_start(label, step, n_lambda, sim_config)
                # Submit to queue
                production.run(
                    ignore_queue=False, extra_args=sim_config.get("gromacs", "")
                )
            else:
                window_states["already queued"] += 1
            if run_mode == "serial":
                logger.info(
                    "%s: waiting for lambda window %d/%d before starting the next one",
                    label,
                    step + 1,
                    n_lambda,
                )
                # Wait for simulation to finish or complete required number of steps before
                # continuing with the next lambda step
                log_file = Path(production.out_coord).with_suffix(".log")
                n_wait_steps = sim_config.protein.get("n_steps_next_lambda", None)
                while True:
                    if Path(production.out_coord).exists():
                        # Simulation finished
                        break
                    elif log_file.exists() and n_wait_steps is not None:
                        # Check current step from log file
                        try:
                            with open(log_file, "r", encoding="utf-8") as lf:
                                log_content = lf.read()
                            if "Step           Time" not in log_content:
                                continue
                            current_sim_step = int(
                                log_content.split("Step           Time")[-1]
                                .splitlines()[1]
                                .split()[0]
                            )
                            if current_sim_step >= n_wait_steps:
                                break
                        except Exception:
                            logger.debug("Could not read log file %s", log_file)
                    await asyncio.sleep(5)
        if run_mode == "serial":
            # Create starting structure for next lambda step by selecting frame where
            # the ligand is closest to the pocket center reference
            starting_config = Path(production.out_coord).with_suffix(".ns.gro")
            if not Path(starting_config).exists() and step > 0:
                traj = mdtraj.load(
                    Path(production.out_coord).with_suffix(".xtc"),
                    top=equilibration.out_coord,
                )
                lig_indices = traj.topology.select("resname LIG")
                dum_indices = traj.topology.select("resname DUM")
                start = int(np.ceil(len(traj) * 0.2))
                dum_indices_rep = np.repeat(dum_indices, len(lig_indices))
                pair_indices = np.stack((lig_indices, dum_indices_rep), axis=-1)
                distances = mdtraj.compute_distances(traj[start:], pair_indices)
                distances = distances.mean(axis=1)
                coordinates = np.expand_dims(
                    traj.xyz[distances.argmin() + start], axis=0
                )
                out_traj = mdtraj.formats.GroTrajectoryFile(starting_config, "w")
                box = np.expand_dims(
                    traj.unitcell_vectors[distances.argmin() + start], 0
                )
                out_traj.write(coordinates, traj.topology, unitcell_vectors=box)
                out_traj.close()
                logger.debug(
                    "Created starting structure for next lambda step: %s",
                    starting_config,
                )
    _log_production_summary(label, n_lambda, window_states, sim_config)
    if run_mode == "serial":
        # Wait for all lambda simulations to finish
        for step in range(n_lambda):
            lambda_gro = system_directory / f"lambda-{step:03d}" / "production.gro"
            while not lambda_gro.exists():
                await asyncio.sleep(1)
    return n_lambda


async def simulate_ligand(
    ligand_directory: Path, config: OmegaConf, n_retries: int = 2
):
    """
    Simulate a ligand in the protein pocket and in all solvents of the configuration.
    Expects a ligand .gro and .itp file in the ligand directory, creates a subdirectory per
    system and calculates the free energies once all lambda simulations are finished.
    :param ligand_directory: Directory where the ligand files are located.
    :param config: Configuration object.
    :param n_retries: Number of retries for a simulation in case of failure.
    """
    n_lambdas = {}
    last_tried_system = None
    for attempt in range(1, n_retries + 1):
        try:
            if config.simulation.get("protein", None) is not None:
                for key in rerun_keys(config.simulation, "protein"):
                    last_tried_system = key
                    n_lambdas[key] = await simulate_ligand_with_protein(
                        ligand_directory=ligand_directory,
                        sim_config=config.simulation,
                        key=key,
                    )
            # Get list of solvents to simulate
            solvent_names = []
            for name in config.simulation:
                if (
                    name == "protein"
                    or (not isinstance(config.simulation[name], (dict, DictConfig)))
                    or (
                        not config.simulation[name].get("composition", None)
                        and name != "water"
                    )
                    or (not config.simulation[name].get("box_margin", None))
                ):
                    continue
                solvent_names.append(name)
            # Iterate over solvents and simulate ligand in each solvent
            for name in solvent_names:
                for key in rerun_keys(config.simulation, name):
                    last_tried_system = key
                    n_lambdas[key] = await simulate_ligand_in_solvent(
                        name=name,
                        ligand_directory=ligand_directory,
                        sim_config=config.simulation,
                        key=key,
                    )
            break  # If we reach this point, all simulations were successful
        except CalledProcessError as e:
            if attempt == n_retries:
                raise e
            logger.warning(
                "Simulation attempt %d/%d for %s in %s failed: %s",
                attempt,
                n_retries,
                local_path(ligand_directory),
                last_tried_system,
                str(e),
            )
            shutil.rmtree(ligand_directory / last_tried_system)
    # Wait for all lambda simulations to finish
    lambda_gro_files = []
    for key, n_lambda in n_lambdas.items():
        if n_lambda is not None:
            lambda_gro_files += [
                ligand_directory / key / f"lambda-{step:03d}" / "production.gro"
                for step in range(n_lambda)
            ]
    last_finished, last_report = None, 0.0
    while True:
        finished = sum(1 for f in lambda_gro_files if f.exists())
        if finished == len(lambda_gro_files):
            break
        # Report when a window finished, otherwise every 10 minutes
        if finished != last_finished or monotonic() - last_report > 600:
            logger.info(
                "%s: %d/%d lambda windows finished, waiting for the rest",
                ligand_directory.name,
                finished,
                len(lambda_gro_files),
            )
            last_finished, last_report = finished, monotonic()
        await asyncio.sleep(5)
    # Perform free energy calculations
    logger.info(
        "%s: all simulations finished, calculating free energies", ligand_directory.name
    )
    calculate_free_energies(ligand_directory, list(n_lambdas.keys()))
