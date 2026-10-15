"""Setup of the coarse-grained protein structure and its binding pocket dummy bead."""

import logging
import re
import shutil
from itertools import combinations
from math import comb
from pathlib import Path
from subprocess import run

import numpy as np
from omegaconf import ListConfig, OmegaConf
from scipy.optimize import minimize
from scipy.spatial.distance import cdist

from .utils import (
    FileUpdater,
    check_file_exists,
    clean_pdb_structure,
    get_charge_from_itp,
    get_molecule_name_from_itp,
    get_residue_com_from_pdb,
    write_json_file,
)

logger = logging.getLogger(__name__)


def select_reference_beads(
    bead_coords: np.ndarray,
    center: np.ndarray,
    n_beads: int,
    cutoff: float,
    max_cutoff: float = 2.5,
    max_combinations: int = 500000,
) -> tuple[np.ndarray, float]:
    """
    Select backbone beads around the pocket center that are spread as far apart as
    possible, i.e. maximize the sum of their pairwise distances. All combinations are
    tested if feasible, otherwise a greedy selection refined by single swaps is used. The
    cutoff is increased if it holds fewer than the requested number of beads.
    :param bead_coords: Coordinates of all backbone beads in nm.
    :param center: Pocket center in nm.
    :param n_beads: Number of beads to select.
    :param cutoff: Initial cutoff around the pocket center in nm.
    :param max_cutoff: Maximum cutoff in nm.
    :param max_combinations: Maximum number of combinations to test exhaustively.
    :return: Indices into bead_coords of the selected beads and the cutoff used.
    """
    distances_to_center = np.linalg.norm(bead_coords - center, axis=1)
    while np.sum(distances_to_center < cutoff) < n_beads:
        if cutoff >= max_cutoff:
            raise ValueError(
                f"Less than {n_beads} backbone beads within {max_cutoff} nm of the "
                "ligand, check the ligand position."
            )
        cutoff = min(cutoff + 0.1, max_cutoff)
        logger.warning("Increased the pocket cutoff to %.1f nm", cutoff)
    candidates = np.nonzero(distances_to_center < cutoff)[0]
    pairwise = cdist(bead_coords[candidates], bead_coords[candidates])
    n_candidates = len(candidates)
    n_combinations = comb(n_candidates, n_beads)
    logger.debug(
        "Found %d beads within %.2f nm cutoff, %d combinations",
        n_candidates,
        cutoff,
        n_combinations,
    )
    if n_combinations <= max_combinations:
        best_combination = None
        max_cum_distance = -1
        for combination in combinations(range(n_candidates), n_beads):
            idx = np.array(combination)
            cum_distance = np.sum(pairwise[np.ix_(idx, idx)]) / 2
            if cum_distance > max_cum_distance:
                max_cum_distance = cum_distance
                best_combination = combination
        return candidates[list(best_combination)], cutoff
    # Greedy selection starting from the most distant pair
    i, j = np.unravel_index(np.argmax(pairwise), pairwise.shape)
    selected = [int(i), int(j)]
    while len(selected) < n_beads:
        scores = pairwise[:, selected].sum(axis=1)
        scores[selected] = -1
        selected.append(int(np.argmax(scores)))
    # Improve by swapping single beads as long as the cumulative distance grows
    improved = True
    while improved:
        improved = False
        for pos in range(n_beads):
            others = selected[:pos] + selected[pos + 1 :]
            scores = pairwise[:, others].sum(axis=1)
            scores[others] = -1
            best = int(np.argmax(scores))
            if scores[best] > scores[selected[pos]] + 1e-9:
                selected[pos] = best
                improved = True
    return candidates[selected], cutoff


def add_dummy_atom_to_pocket(ligand_com: np.ndarray, sim_config: OmegaConf) -> dict:
    """
    Add a dummy atom to the pocket region of the protein, used as the reference point for
    restraining the ligand. The atom is a virtual site of several backbone beads.
    :param ligand_com: Center of mass of the ligand in nm.
    :param sim_config: Simulation configuration from the config file.
    :return: Indices of the backbone beads constructing the virtual site, its position,
        its distance to the ligand center of mass and the pocket cutoff used.
    """
    logger.info("Adding dummy atom to protein pocket at %s", ligand_com)
    # Check if coarse-grained protein files exist
    protein_pdb = Path(sim_config.directory) / "protein-cg.pdb"
    protein_itp = Path(sim_config.directory) / "protein.itp"
    check_file_exists(protein_pdb, protein_itp)
    # Add dummy bead to coarse-grained pdb file after the last protein bead
    with FileUpdater(protein_pdb) as pdb_file:
        pdb_lines = pdb_file.readlines()
        atom_lines = [i for i, line in enumerate(pdb_lines) if line.startswith("ATOM")]
        last_atom = pdb_lines[atom_lines[-1]]
        atom_id = len(atom_lines) + 1
        residue_id = (int(last_atom[22:26]) + 1) % 10000
        chain = last_atom[21]
        dummy_lines = [
            f"ATOM  {atom_id % 100000:>5} DUM  DUM {chain}{residue_id:>4}    "
            + f"{ligand_com[0] * 10:8.3f}{ligand_com[1] * 10:8.3f}{ligand_com[2] * 10:8.3f}"
            + "  1.00  0.00\n",
            f"TER   {(atom_id + 1) % 100000:>5}      DUM {chain}{residue_id:>4}\n",
        ]
        insert_at = atom_lines[-1] + 1
        if insert_at < len(pdb_lines) and pdb_lines[insert_at].startswith("TER"):
            insert_at += 1
        pdb_lines[insert_at:insert_at] = dummy_lines
        pdb_file.overwrite(pdb_lines)
    # Add dummy atom to itp file at the end of the atoms section
    with FileUpdater(protein_itp) as itp_file:
        itp_lines = itp_file.readlines()
        section = None
        last_atom_line = None
        bb_bead_indices = []
        for idx, line in enumerate(itp_lines):
            content = line.split(";")[0].strip()
            if not content or content.startswith("#"):
                continue
            match = re.match(r"\[\s*(\w+)\s*\]", content)
            if match:
                section = match.group(1)
            elif section == "atoms":
                last_atom_line = idx
                # The new Martini protein model uses CA, the old one uses BB
                if content.split()[4] in ("BB", "CA"):
                    bb_bead_indices.append(int(content.split()[0]))
        n_itp_atoms = int(itp_lines[last_atom_line].split()[0])
        if n_itp_atoms != atom_id - 1:
            raise ValueError(
                f"Protein itp has {n_itp_atoms} atoms, coarse-grained pdb {atom_id - 1}"
            )
        itp_lines.insert(
            last_atom_line + 1,
            f"{atom_id:>3} DUM  {residue_id:>3} DUM DUM {atom_id:>3}  0.0\n",
        )
        itp_file.overwrite(itp_lines)
    # Calculate virtual sites for dummy atom
    ## Get backbone beads with coordinates from pdb file
    cg_coords = np.array(
        [
            [float(line[30:38]), float(line[38:46]), float(line[46:54])]
            for line in (pdb_lines[i] for i in atom_lines)
        ]
    )
    bb_bead_indices = np.array(bb_bead_indices)
    bb_bead_coords = cg_coords[bb_bead_indices - 1] / 10  # Convert to nm
    ## Find combination of backbone beads (within cutoff) that maximizes the distance between them
    dummy_config = sim_config.protein.dummy_bead
    selection, cutoff = select_reference_beads(
        bb_bead_coords,
        ligand_com,
        dummy_config.n_virtual_sites,
        dummy_config.cutoff,
        max_cutoff=dummy_config.get("max_cutoff", 2.5),
    )
    best_bb_indices = bb_bead_indices[selection]
    best_coordinates = bb_bead_coords[selection]
    logger.debug("Best combination of beads for virtual site: %s", best_bb_indices)
    ## Calculate weights for each constituting bead to match the ligand COM
    n = best_coordinates.shape[0]
    lambda_reg = dummy_config.get("regularization", 1)
    logger.debug("Using regularization param %f for virtual site weights", lambda_reg)
    # Solve constrained QP: minimise regularized residual subject to
    # sum(p) = 1 and p >= 0 (no negative weights).
    M = best_coordinates @ best_coordinates.T
    M += lambda_reg * np.eye(n) - (1 / n) * np.ones((n, n))
    c = best_coordinates @ ligand_com
    result = minimize(
        fun=lambda p: np.dot(p, M @ p) - 2 * np.dot(p, c),
        x0=np.ones(n) / n,
        method="SLSQP",
        bounds=[(0.0, None)] * n,
        constraints={"type": "eq", "fun": lambda p: p.sum() - 1.0},
        options={"ftol": 1e-12},
    )
    if not result.success:
        logger.warning("Weight optimisation did not converge: %s", result.message)

    params = result.x
    logger.debug(
        "Weights for virtual site: %s", ", ".join([f"{p:.3f}" for p in params])
    )
    mask = params > 0.0001
    virtual_site_coord = np.sum(
        params[mask, np.newaxis] * best_coordinates[mask], axis=0
    ) / np.sum(params[mask])
    offset = float(np.linalg.norm(virtual_site_coord - ligand_com))
    logger.info(
        "Virtual site position: [%s], ligand COM: [%s], Distance: %.3f nm",
        ", ".join([f"{v:.3f}" for v in virtual_site_coord]),
        ", ".join([f"{v:.3f}" for v in ligand_com]),
        offset,
    )
    if offset > 0.1:
        logger.warning(
            "Virtual site is %.2f nm away from the ligand center of mass, the pocket "
            "may not be enclosed by the protein backbone.",
            offset,
        )
    n_indices = [i for i in range(n) if params[i] > 0.0001]
    vs_text = " ".join(f"{best_bb_indices[i]} {params[i]:.3f}" for i in n_indices)
    vmd = " or ".join([f"index {best_bb_indices[i]}" for i in n_indices])
    logger.debug("Virtual site definition: %s", vs_text)
    # Add virtual site to itp file
    with FileUpdater(protein_itp) as itp_file:
        itp_content = itp_file.readlines()
        for idx, line in enumerate(itp_content):
            if "virtual_sitesn" in line:
                itp_content.insert(idx + 1, f"; VMD: {vmd}\n{atom_id} 3 {vs_text}\n")
                break
        else:
            itp_content.append("\n[ virtual_sitesn ]\n")
            itp_content.append(f"; VMD: {vmd}\n{atom_id} 3 {vs_text}\n")
        itp_file.overwrite(itp_content)
    return {
        "beads": best_bb_indices.tolist(),
        "position": virtual_site_coord.tolist(),
        "offset": offset,
        "cutoff": float(cutoff),
    }


def _martinize_includes(top_file: Path) -> list[str]:
    """
    Get the molecule topologies included in a martinize2 topology file.
    :param top_file: Topology file written by martinize2.
    :return: Included itp files, without the force field itp.
    """
    includes = re.findall(r'#include\s+"([^"]+)"', top_file.read_text(encoding="utf-8"))
    return [inc for inc in includes if not Path(inc).name.startswith("martini")]


def setup_protein_structure(config: OmegaConf):
    """
    Setup the CG protein structure from the atomistic structure using martinize2 and
    place a dummy bead at the center of the binding pocket.
    :param config: Configuration object.
    """
    logger.info("Setting up protein structure with martinize2")
    sim_config = config.simulation
    # Setup input and output files
    input_file = sim_config.protein.file
    directory = Path(sim_config.directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    output_clean = directory / "protein-clean.pdb"
    output_itp = directory / "protein.itp"
    output_top = directory / "protein.top"
    output_pdb = directory / "protein-cg.pdb"
    check_file_exists(input_file)
    if (directory / "protein_info.json").exists():
        logger.info("Protein files already exist, skipping martinize step")
        return
    # Cleanup pdb structure
    chains = clean_pdb_structure(
        input_file, output_clean, chain=sim_config.protein.get("chain", None)
    )
    if len(chains) == 0:
        raise ValueError(f"No protein atoms found in {input_file}")
    logger.info("Protein chains: %s", ", ".join(c.strip() or "-" for c in chains))
    # Construct martinize2 command with extra arguments from config
    martinize_args = dict(sim_config.protein.get("martinize", {}))
    if len(chains) > 1 and "merge" not in martinize_args:
        # The dummy bead is a virtual site, so all chains have to be one molecule
        martinize_args["merge"] = "all"
    extra_args = ""
    for key, value in martinize_args.items():
        if isinstance(value, bool) and value:
            extra_args += f" -{key}"
        elif value is not None and not isinstance(value, bool):
            extra_args += f" -{key} {value}"
    martinize_cmd = (
        f"martinize2 -f {output_clean} -o {output_top} -x {output_pdb} {extra_args}"
    )
    # Run martinize2 command in the system directory, where it writes the itp file
    logger.debug("Running command: %s", martinize_cmd)
    with open(directory / "martinize.log", "w", encoding="utf-8") as log_file:
        run(
            martinize_cmd,
            shell=True,
            check=True,
            cwd=directory,
            stdout=log_file,
            stderr=log_file,
        )
    if not output_top.exists() or not output_pdb.exists():
        raise FileNotFoundError("Martinize2 did not produce expected output files.")
    # Move itp file to target location
    itp_files = _martinize_includes(output_top)
    if len(itp_files) != 1:
        raise ValueError(
            f"Martinize2 produced {len(itp_files)} molecules ({', '.join(itp_files)}), "
            "merge all chains into a single molecule."
        )
    itp_file = directory / itp_files[0]
    if not itp_file.exists():
        raise FileNotFoundError(f"Included itp file {itp_file} not found.")
    if itp_file.resolve() != output_itp:
        shutil.move(itp_file, output_itp)
        logger.debug("Moved itp file to %s", output_itp)
    # Run OLIVES if specified in config
    olives = sim_config.protein.get("olives", None)
    if olives:
        if not Path(olives).exists():
            raise FileNotFoundError(f"OLIVES script {olives} not found.")
        olives_cmd = f"python {olives} -c {output_pdb} -i {output_itp}"
        logger.debug("Running OLIVES command: %s", olives_cmd)
        with open(directory / "olives.log", "w", encoding="utf-8") as log_file:
            run(
                olives_cmd,
                shell=True,
                check=True,
                cwd=directory,
                stdout=log_file,
                stderr=log_file,
            )
    elif "elastic" not in martinize_args and "go" not in martinize_args:
        logger.warning(
            "Neither OLIVES nor an elastic network is configured, the tertiary "
            "structure of the protein is not stabilized."
        )
    # Delete useless top file
    output_top.unlink()
    logger.debug("Deleted temporary top file %s", output_top)
    # Get ligand COM from config or reference pdb file
    if isinstance(sim_config.protein.ligand, str):
        ligand_com = get_residue_com_from_pdb(input_file, sim_config.protein.ligand)
    elif isinstance(sim_config.protein.ligand, (list, ListConfig)):
        ligand_com = np.array(sim_config.protein.ligand)
    else:
        raise ValueError("The ligand position has to be given in the configuration")
    # Add dummy atom to pdb at the position of the ligand
    dummy = add_dummy_atom_to_pocket(ligand_com, sim_config)
    # Create protein information file
    protein_info = {
        "name": get_molecule_name_from_itp(output_itp),
        "charge": get_charge_from_itp(output_itp),
        "chains": [c.strip() for c in chains],
        "ligand_com": ligand_com.tolist(),
        "dummy_reference_beads": dummy["beads"],
        "dummy_position": dummy["position"],
        "dummy_offset": dummy["offset"],
        "pocket_cutoff": dummy["cutoff"],
    }
    write_json_file(protein_info, directory / "protein_info.json")
