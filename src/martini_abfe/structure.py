"""Handling of the input complex and placement of the coarse-grained ligand in the pocket."""

import logging
import re
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)

ATOM_MASSES = {
    "H": 1.008,
    "D": 2.014,
    "B": 10.81,
    "C": 12.011,
    "N": 14.007,
    "O": 15.999,
    "F": 18.998,
    "SI": 28.085,
    "P": 30.974,
    "S": 32.06,
    "CL": 35.45,
    "SE": 78.971,
    "BR": 79.904,
    "I": 126.90,
}
LIGAND_MOLECULE_NAME = "ligand"
LIGAND_RESIDUE_NAME = "LIG"


# --- PDB handling ---


def _element(line: str) -> str:
    """
    Get the element of a pdb atom line, guessed from the atom name if the column is empty.
    :param line: ATOM or HETATM line of a pdb file.
    :return: Element symbol in upper case.
    """
    element = line[76:78].strip().upper() if len(line) >= 78 else ""
    if element:
        return element
    name = re.sub(r"[^A-Za-z]", "", line[12:16]).upper()
    if name[:2] in ATOM_MASSES and name[:2] not in ("CA", "CD", "NE", "HG"):
        return name[:2]
    return name[:1]


def read_pdb_atoms(pdb_file: Path) -> list[dict]:
    """
    Read the atoms of the first model of a pdb file.
    :param pdb_file: Path to the pdb file.
    :return: One dictionary per atom with the record line and its parsed fields,
        coordinates in nm.
    """
    atoms = []
    with open(pdb_file, "r", encoding="utf-8") as file:
        for line in file:
            if line.startswith("ENDMDL"):
                break
            if not line.startswith(("ATOM", "HETATM")):
                continue
            atoms.append(
                {
                    "line": line.rstrip("\n"),
                    "name": line[12:16].strip(),
                    "altloc": line[16].strip(),
                    "resname": line[17:21].strip(),
                    "chain": line[21].strip(),
                    "resid": line[22:27].strip(),
                    "xyz": np.array(
                        [float(line[30:38]), float(line[38:46]), float(line[46:54])]
                    )
                    / 10.0,
                    "element": _element(line),
                }
            )
    return atoms


def split_complex(
    complex_pdb: Path,
    ligand_resname: str,
    protein_pdb: Path,
    ligand_pdb: Path,
    ligand_chain: str = None,
    ligand_resid: str | int = None,
) -> list[dict]:
    """
    Split a protein-ligand complex into the protein and the bound ligand. All lines of the
    selected ligand residue are removed from the protein file, everything else is kept and
    cleaned up during the protein setup.
    :param complex_pdb: Atomistic structure of the ligand bound to the protein.
    :param ligand_resname: Residue name of the ligand in the complex.
    :param protein_pdb: Output pdb file of the protein.
    :param ligand_pdb: Output pdb file of the atomistic ligand.
    :param ligand_chain: Chain of the ligand, required if the residue occurs more than once.
    :param ligand_resid: Residue number of the ligand, required if the residue occurs more
        than once in the selected chain.
    :return: Atoms of the selected ligand as returned by read_pdb_atoms.
    """
    atoms = read_pdb_atoms(complex_pdb)
    ligand_atoms = [a for a in atoms if a["resname"] == ligand_resname]
    if ligand_chain is not None:
        ligand_atoms = [a for a in ligand_atoms if a["chain"] == ligand_chain]
    if ligand_resid is not None:
        ligand_atoms = [a for a in ligand_atoms if a["resid"] == str(ligand_resid)]
    if len(ligand_atoms) == 0:
        raise ValueError(
            f"No ligand with residue name {ligand_resname} (chain {ligand_chain}, "
            f"resid {ligand_resid}) found in {complex_pdb}"
        )
    copies = sorted({(a["chain"], a["resid"]) for a in ligand_atoms})
    if len(copies) > 1:
        logger.warning(
            "Found %d copies of ligand %s (chain, resid): %s. Using the first one, "
            "select another one with the ligand chain and resid options.",
            len(copies),
            ligand_resname,
            copies,
        )
        ligand_atoms = [a for a in ligand_atoms if (a["chain"], a["resid"]) == copies[0]]
    # Keep only the first alternate location of every ligand atom
    altlocs = sorted({a["altloc"] for a in ligand_atoms if a["altloc"]})
    if len(altlocs) > 1:
        logger.warning("Ligand has alternate locations %s, using %s", altlocs, altlocs[0])
        ligand_atoms = [a for a in ligand_atoms if a["altloc"] in ("", altlocs[0])]
    selected = (ligand_atoms[0]["chain"], ligand_atoms[0]["resid"])
    logger.info(
        "Selected ligand %s in chain '%s' with resid %s (%d atoms)",
        ligand_resname,
        *selected,
        len(ligand_atoms),
    )
    # Write protein file without the selected ligand
    with open(complex_pdb, "r", encoding="utf-8") as infile:
        lines = infile.readlines()
    with open(protein_pdb, "w", encoding="utf-8") as outfile:
        for line in lines:
            if (
                line.startswith(("ATOM", "HETATM", "ANISOU"))
                and line[17:21].strip() == ligand_resname
                and (line[21].strip(), line[22:27].strip()) == selected
            ):
                continue
            outfile.write(line)
    # Write ligand file
    with open(ligand_pdb, "w", encoding="utf-8") as outfile:
        for atom in ligand_atoms:
            outfile.write(atom["line"] + "\n")
        outfile.write("END\n")
    return ligand_atoms


def center_of_mass(atoms: list[dict]) -> np.ndarray:
    """
    Calculate the center of mass of a list of atoms.
    :param atoms: Atoms as returned by read_pdb_atoms.
    :return: Center of mass in nm.
    """
    masses = []
    for atom in atoms:
        if atom["element"] not in ATOM_MASSES:
            logger.warning(
                "Unknown element %s of atom %s, using carbon mass",
                atom["element"],
                atom["name"],
            )
        masses.append(ATOM_MASSES.get(atom["element"], ATOM_MASSES["C"]))
    masses = np.array(masses)
    coords = np.array([a["xyz"] for a in atoms])
    return np.sum(coords * masses[:, None], axis=0) / np.sum(masses)


# --- Coarse-grained ligand files ---


def read_itp_atoms(itp_file: Path) -> tuple[str, list[list[str]]]:
    """
    Read the molecule name and the atom lines of a single-molecule itp file.
    :param itp_file: Path to the itp file.
    :return: Name of the molecule and the split lines of its atoms section.
    """
    molecule_names = []
    atoms = []
    section = None
    with open(itp_file, "r", encoding="utf-8") as file:
        for line in file:
            content = line.split(";")[0].strip()
            if not content or content.startswith("#"):
                continue
            match = re.match(r"\[\s*(\w+)\s*\]", content)
            if match:
                section = match.group(1)
            elif section == "moleculetype":
                molecule_names.append(content.split()[0])
                section = None
            elif section == "atoms":
                atoms.append(content.split())
    if len(molecule_names) != 1:
        raise ValueError(
            f"{itp_file} must define exactly one molecule, found {len(molecule_names)}"
        )
    return molecule_names[0], atoms


def normalize_ligand_topology(itp_in: Path, itp_out: Path) -> list[list[str]]:
    """
    Copy a ligand topology, renaming its molecule to 'ligand' and its residues to 'LIG',
    the names the simulation setup and the mdp files refer to.
    :param itp_in: Input topology of the coarse-grained ligand.
    :param itp_out: Output topology.
    :return: Split lines of the atoms section.
    """
    molecule_name, atoms = read_itp_atoms(itp_in)
    if len(atoms) == 0:
        raise ValueError(f"No atoms found in {itp_in}")
    if any(len(atom) < 7 for atom in atoms):
        raise ValueError(f"Every atom in {itp_in} needs an explicit charge column")
    output = []
    section = None
    with open(itp_in, "r", encoding="utf-8") as file:
        for line in file:
            content = line.split(";")[0].strip()
            match = re.match(r"\[\s*(\w+)\s*\]", content)
            if match:
                section = match.group(1)
            elif content and not content.startswith("#"):
                if section == "moleculetype":
                    nrexcl = content.split()[1]
                    line = f"{LIGAND_MOLECULE_NAME} {nrexcl}\n"
                    section = None
                elif section == "atoms":
                    parts = content.split()
                    parts[3] = LIGAND_RESIDUE_NAME
                    comment = line.split(";", 1)[1] if ";" in line else None
                    line = " ".join(parts)
                    line += f" ;{comment}" if comment else "\n"
            output.append(line)
    with open(itp_out, "w", encoding="utf-8") as file:
        file.writelines(output)
    logger.info(
        "Wrote ligand topology %s (molecule %s renamed to %s)",
        itp_out,
        molecule_name,
        LIGAND_MOLECULE_NAME,
    )
    return atoms


def read_gro_coordinates(gro_file: Path) -> np.ndarray:
    """
    Read the coordinates of a gro file.
    :param gro_file: Path to the gro file.
    :return: Coordinates in nm.
    """
    with open(gro_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
    n_atoms = int(lines[1].strip())
    return np.array(
        [
            (float(ln[20:28]), float(ln[28:36]), float(ln[36:44]))
            for ln in lines[2 : 2 + n_atoms]
        ]
    )


def write_ligand_gro(
    gro_file: Path, coordinates: np.ndarray, atom_names: list[str], title: str
):
    """
    Write a ligand gro file with a single residue named 'LIG'.
    :param gro_file: Output gro file.
    :param coordinates: Coordinates in nm.
    :param atom_names: Atom names, taken from the topology so that they match.
    :param title: Title line of the file.
    """
    with open(gro_file, "w", encoding="utf-8") as file:
        file.write(f"{title}\n{len(coordinates)}\n")
        for i, (name, xyz) in enumerate(zip(atom_names, coordinates), start=1):
            file.write(
                f"{1:>5}{LIGAND_RESIDUE_NAME:<5}{name[:5]:>5}{i % 100000:>5}"
                f"{xyz[0]:8.3f}{xyz[1]:8.3f}{xyz[2]:8.3f}\n"
            )
        # Box large enough to hold the ligand, only used by the solvent setup as reference
        extent = np.ptp(coordinates, axis=0).max() + 2.0
        file.write(f"{extent:10.5f}{extent:10.5f}{extent:10.5f}\n")


def gro_to_pdb(gro_file: Path, pdb_file: Path, chain: str = "A") -> int:
    """
    Convert a gro file into a pdb file with a single chain, keeping atom and residue
    names and residue numbers.
    :param gro_file: Input gro file.
    :param pdb_file: Output pdb file.
    :param chain: Chain identifier of all atoms.
    :return: Number of atoms.
    """
    with open(gro_file, "r", encoding="utf-8") as file:
        lines = file.readlines()
    n_atoms = int(lines[1].strip())
    box = [float(v) * 10 for v in lines[2 + n_atoms].split()[:3]]
    pdb_lines = []
    if any(box):
        pdb_lines.append(
            f"CRYST1{box[0]:9.3f}{box[1]:9.3f}{box[2]:9.3f}"
            "  90.00  90.00  90.00 P 1           1\n"
        )
    for i, line in enumerate(lines[2 : 2 + n_atoms], start=1):
        resid = int(line[0:5])
        resname = line[5:10].strip()
        name = line[10:15].strip()
        # Atom names shorter than four characters start in the second column
        name = name[:4] if len(name) >= 4 else f" {name}"
        resname = f"{resname:>3} " if len(resname) <= 3 else resname[:4]
        x, y, z = (float(line[s : s + 8]) * 10 for s in (20, 28, 36))
        pdb_lines.append(
            f"ATOM  {i % 100000:>5} {name:<4} {resname}{chain}{resid % 10000:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00\n"
        )
    pdb_lines.append(f"TER   {(n_atoms + 1) % 100000:>5}\nEND\n")
    with open(pdb_file, "w", encoding="utf-8") as file:
        file.writelines(pdb_lines)
    return n_atoms


def read_mapping(ndx_file: Path) -> list[list[int]]:
    """
    Read a bead mapping in GROMACS index format, one group per bead in topology order
    holding the 1-based indices of the atomistic ligand atoms it represents.
    :param ndx_file: Path to the mapping file.
    :return: Zero-based atom indices of every bead.
    """
    mapping = []
    with open(ndx_file, "r", encoding="utf-8") as file:
        for line in file:
            content = line.split(";")[0].strip()
            if not content:
                continue
            if content.startswith("["):
                mapping.append([])
            elif mapping:
                mapping[-1] += [int(i) - 1 for i in content.split()]
    return mapping


# --- Placement of the coarse-grained ligand ---


def _kabsch(mobile: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Optimal rotation and translation superimposing mobile onto target points.
    :param mobile: Points to move, shape (n, 3).
    :param target: Reference points, shape (n, 3).
    :return: Rotation matrix and translation, applied as x @ R.T + t.
    """
    mobile_center = mobile.mean(axis=0)
    target_center = target.mean(axis=0)
    h = (mobile - mobile_center).T @ (target - target_center)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rotation = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rotation, target_center - mobile_center @ rotation.T


def _align_with_mapping(
    cg_coords: np.ndarray, aa_coords: np.ndarray, mapping: list[list[int]]
) -> np.ndarray:
    """
    Superimpose the coarse-grained ligand onto the bead positions of the mapped
    atomistic ligand. Indices beyond the atomistic structure (e.g. hydrogens missing in a
    crystal structure) are ignored.
    :param cg_coords: Coarse-grained ligand coordinates in nm.
    :param aa_coords: Atomistic ligand coordinates in nm.
    :param mapping: Zero-based atom indices of every bead.
    :return: Aligned coarse-grained coordinates.
    """
    if len(mapping) != len(cg_coords):
        raise ValueError(
            f"Mapping defines {len(mapping)} beads, the ligand has {len(cg_coords)}"
        )
    beads, targets = [], []
    for bead, indices in enumerate(mapping):
        indices = [i for i in indices if 0 <= i < len(aa_coords)]
        if indices:
            beads.append(bead)
            targets.append(aa_coords[indices].mean(axis=0))
    if len(beads) < 3:
        raise ValueError("Less than three beads of the mapping match atomistic atoms")
    targets = np.array(targets)
    rotation, translation = _kabsch(cg_coords[beads], targets)
    aligned = cg_coords @ rotation.T + translation
    rmsd = np.sqrt(np.mean(np.sum((aligned[beads] - targets) ** 2, axis=1)))
    logger.info(
        "Aligned ligand onto %d mapped bead positions, RMSD %.3f nm", len(beads), rmsd
    )
    return aligned


def _fit_to_envelope(
    cg_coords: np.ndarray, aa_coords: np.ndarray, n_starts: int = 64, seed: int = 0
) -> np.ndarray:
    """
    Orient the coarse-grained ligand inside the atomistic ligand without a mapping, by
    minimizing the symmetric Chamfer distance between the beads and the atoms: every bead
    should sit close to an atom and every atom close to a bead.
    :param cg_coords: Coarse-grained ligand coordinates in nm.
    :param aa_coords: Atomistic (heavy) atom coordinates in nm.
    :param n_starts: Number of random starting orientations.
    :param seed: Random seed for the starting orientations.
    :return: Fitted coarse-grained coordinates.
    """
    aa_tree = cKDTree(aa_coords)
    cg_center = cg_coords.mean(axis=0)
    aa_center = aa_coords.mean(axis=0)
    centered = cg_coords - cg_center

    def transform(params: np.ndarray) -> np.ndarray:
        rotation = Rotation.from_rotvec(params[:3])
        return rotation.apply(centered) + aa_center + params[3:]

    def chamfer(params: np.ndarray) -> float:
        coords = transform(params)
        d_beads = aa_tree.query(coords)[0]
        d_atoms = cKDTree(coords).query(aa_coords)[0]
        return np.mean(d_beads**2) + np.mean(d_atoms**2)

    starts = [np.zeros(3)] + list(
        Rotation.random(n_starts, random_state=seed).as_rotvec()
    )
    best = None
    for start in starts:
        result = minimize(
            chamfer, np.concatenate([start, np.zeros(3)]), method="Powell"
        )
        if best is None or result.fun < best.fun:
            best = result
    logger.info(
        "Fitted ligand orientation to the atomistic ligand, Chamfer RMS %.3f nm",
        np.sqrt(best.fun / 2),
    )
    return transform(best.x)


def place_cg_ligand(
    cg_gro: Path,
    aa_atoms: list[dict],
    orient: str = "auto",
    mapping_file: Path = None,
) -> np.ndarray:
    """
    Place the coarse-grained ligand at the pose of the atomistic ligand.
    :param cg_gro: Structure of the coarse-grained ligand.
    :param aa_atoms: Atoms of the bound atomistic ligand as returned by read_pdb_atoms.
    :param orient: 'mapping' to superimpose the beads onto the mapped atom groups,
        'auto' to fit the orientation to the shape of the atomistic ligand, or 'none' to
        only move the centroid of the ligand to the atomistic center of mass.
    :param mapping_file: Bead mapping in index format, required for 'mapping'.
    :return: Coordinates of the placed coarse-grained ligand in nm.
    """
    cg_coords = read_gro_coordinates(cg_gro)
    aa_coords = np.array([a["xyz"] for a in aa_atoms])
    if orient == "mapping":
        if mapping_file is None:
            raise ValueError("Orientation 'mapping' requires a mapping file")
        return _align_with_mapping(cg_coords, aa_coords, read_mapping(mapping_file))
    if orient == "auto":
        if len(cg_coords) < 3:
            logger.info("Ligand has less than three beads, only translating it")
        else:
            heavy = np.array([a["element"] not in ("H", "D") for a in aa_atoms])
            return _fit_to_envelope(cg_coords, aa_coords[heavy] if heavy.any() else aa_coords)
    elif orient != "none":
        raise ValueError(f"Unknown orientation mode {orient}")
    return cg_coords - cg_coords.mean(axis=0) + center_of_mass(aa_atoms)
