"""
Empirical correction of the systematic underbinding of Martini 3, from Duve et al.,
"Predictable Accuracy of Martini 3 Absolute Protein-Ligand Binding Free Energies across
15 Targets" (arXiv:2610.11714). Martini underbinds by a target-dependent offset that grows
linearly with the number of charged residues lining the pocket,

    dG_CG - dG_ref = 0.507 N_charged + 3.265 kcal/mol,

fitted to the per-target mean offsets against FEP+ over 15 targets (R^2 = 0.78,
Q^2 = 0.72). The offset is the same for all ligands of a target, so it improves absolute
binding free energies but leaves the ranking within a target unchanged.
"""

import logging
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from .structure import read_pdb_atoms
from .utils import local_path

logger = logging.getLogger(__name__)

UNDERBINDING_SLOPE = 0.50677  # kcal/mol per charged pocket residue
UNDERBINDING_INTERCEPT = 3.26491  # kcal/mol
UNDERBINDING_UNCERTAINTY = 0.83  # kcal/mol, leave-one-target-out RMSE of the offset
POCKET_CUTOFF = 0.6  # nm, between heavy atoms of the pocket residues and the ligand

# Side chain charges by residue name, for structures without hydrogens these are the
# standard protonation states at pH 7 with neutral histidine
SIDE_CHAIN_CHARGES = {
    "ASP": -1, "GLU": -1, "LYS": 1, "ARG": 1,
    "ASH": 0, "GLH": 0, "LYN": 0, "CYM": -1,
    "HIP": 1, "HSP": 1, "HIH": 1, "HID": 0, "HIE": 0, "HSD": 0, "HSE": 0, "HIS": 0,
}  # fmt: skip
N_CAPS = {"ACE", "FOR"}
C_CAPS = {"NME", "NMA", "NH2", "NHE"}


def residue_charge(resname: str, atom_names: set[str]) -> int:
    """
    Side chain charge of a residue, from its name or, if it has hydrogens, from its
    protonation state.
    :param resname: Residue name, including protonation variants such as HIP or ASH.
    :param atom_names: Names of the atoms of the residue.
    :return: Integer charge of the side chain.
    """
    if not any(name.startswith("H") for name in atom_names):
        return SIDE_CHAIN_CHARGES.get(resname, 0)
    if resname == "ASP":
        return 0 if atom_names & {"HD1", "HD2"} else -1
    if resname == "GLU":
        return 0 if atom_names & {"HE1", "HE2"} else -1
    if resname == "HIS":
        return 1 if {"HD1", "HE2"} <= atom_names else 0
    if resname == "LYS":
        return 1 if {"HZ1", "HZ2", "HZ3"} <= atom_names else 0
    return SIDE_CHAIN_CHARGES.get(resname, 0)


def protein_residue_charges(protein_pdb: Path) -> dict:
    """
    Net charges of all residues of an atomistic protein, from the protonation states of
    the structure. The first and last residue of every chain carry the charge of the
    free termini, unless they are capped.
    :param protein_pdb: Atomistic structure of the protein.
    :return: Net charge by chain and residue number (with insertion code).
    """
    residues = {}
    for atom in read_pdb_atoms(protein_pdb):
        if atom["line"].startswith("ATOM"):
            key = (atom["chain"], atom["resid"])
            name = atom["name"]
            if name[0].isdigit():  # old hydrogen names, e.g. 1HZ for HZ1
                name = name[1:] + name[0]
            residues.setdefault(key, (atom["resname"], set()))[1].add(name)
    charges = {
        key: residue_charge(resname, names) for key, (resname, names) in residues.items()
    }
    chains = {}
    for chain, resid in residues:
        chains.setdefault(chain, []).append((chain, resid))
    for keys in chains.values():
        if residues[keys[0]][0] not in N_CAPS:
            charges[keys[0]] += 1
        if residues[keys[-1]][0] not in C_CAPS:
            charges[keys[-1]] -= 1
    return charges


def pocket_residues(
    protein_pdb: Path, ligand_pdb: Path, cutoff: float = POCKET_CUTOFF
) -> list[tuple[str, str, str]]:
    """
    Residues lining the pocket, i.e. holding a heavy atom within the cutoff of any heavy
    atom of the bound ligand.
    :param protein_pdb: Atomistic structure of the protein.
    :param ligand_pdb: Atomistic structure of the bound ligand.
    :param cutoff: Pocket radius in nm.
    :return: Chain, residue number (with insertion code) and residue name of every pocket
        residue, in order of appearance.
    """
    protein = [
        a
        for a in read_pdb_atoms(protein_pdb)
        if a["line"].startswith("ATOM") and a["element"] not in ("H", "D")
    ]
    ligand = [a for a in read_pdb_atoms(ligand_pdb) if a["element"] not in ("H", "D")]
    if len(protein) == 0 or len(ligand) == 0:
        raise ValueError(f"No heavy atoms found in {protein_pdb} or {ligand_pdb}")
    distances = cdist(
        np.array([a["xyz"] for a in protein]), np.array([a["xyz"] for a in ligand])
    )
    residues = {}
    for atom, near in zip(protein, distances.min(axis=1) < cutoff):
        if near:
            residues.setdefault((atom["chain"], atom["resid"]), atom["resname"])
    return [(chain, resid, resname) for (chain, resid), resname in residues.items()]


def underbinding_offset(n_charged: int) -> tuple[float, float]:
    """
    Predicted underbinding of Martini 3 for a pocket with the given number of charged
    residues.
    :param n_charged: Number of charged pocket residues.
    :return: Offset dG_CG - dG_ref and its uncertainty in kcal/mol.
    """
    return (
        UNDERBINDING_SLOPE * n_charged + UNDERBINDING_INTERCEPT,
        UNDERBINDING_UNCERTAINTY,
    )


def underbinding_correction(
    system_directory: Path, cutoff: float = POCKET_CUTOFF
) -> dict | None:
    """
    Count the charged residues lining the pocket of a prepared system and predict its
    underbinding. As in the fit of the model, the pocket and the protonation states are
    taken from the atomistic complex.
    :param system_directory: Prepared system directory.
    :param cutoff: Pocket radius in nm.
    :return: Charged pocket residues, their count and the predicted offset with its
        uncertainty in kcal/mol, which is subtracted from the binding free energy, or None
        if the atomistic structures are missing.
    """
    system_directory = Path(system_directory)
    protein_pdb = system_directory / "protein-clean.pdb"
    ligand_pdb = system_directory / "ligand-aa.pdb"
    if not protein_pdb.exists() or not ligand_pdb.exists():
        logger.warning(
            "Missing atomistic structures in %s, no underbinding correction",
            local_path(system_directory),
        )
        return None
    pocket = pocket_residues(protein_pdb, ligand_pdb, cutoff)
    charges = protein_residue_charges(protein_pdb)
    charged = [
        {"chain": chain, "resid": resid, "resname": resname, "charge": charges[(chain, resid)]}
        for chain, resid, resname in pocket
        if charges[(chain, resid)] != 0
    ]
    offset, uncertainty = underbinding_offset(len(charged))
    return {
        "cutoff": cutoff,
        "n_pocket": len(pocket),
        "n_charged": len(charged),
        "charged_residues": charged,
        "offset": offset,
        "uncertainty": uncertainty,
    }
