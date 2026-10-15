"""Build COBY solvent boxes around a ligand or a protein-ligand complex for Martini 3."""

import logging
import shutil
import tempfile
from pathlib import Path

import COBY
import numpy as np

from .utils import get_molecule_name_from_itp

logger = logging.getLogger(__name__)


# Definitions of solvents missing from the COBY library, e.g. octanol (OCO).
SOLVENT_LIBRARY = Path(__file__).parent / "coby_solvents.py"
# Average molecular volume (nm^3) per solvent molecule, used to estimate molecule counts.
MOLECULAR_VOLUMES = {"W": 0.120, "OCO": 0.262}
# Molecules per nm^3 at 1 mol/L (Avogadro * 1e-24 L/nm^3).
IONS_PER_NM3_PER_MOLAR = 0.6022


def _solute_diameter(structure_file: Path) -> float:
    """
    Calculate the solute diameter in the same way as insane does it.
    :param structure_file: Coordinate file of the solute (.gro or .pdb).
    :return: Diameter of the solute in nm.
    """
    with open(structure_file, "r", encoding="utf-8") as coord_file:
        lines = [line for line in coord_file if len(line.strip()) > 0]
    if Path(structure_file).suffix == ".pdb":
        coords = np.array(
            [
                (float(ln[30:38]), float(ln[38:46]), float(ln[46:54]))
                for ln in lines
                if ln.startswith(("ATOM", "HETATM"))
            ]
        )
        coords /= 10.0  # Convert to nm
    else:
        coords = np.array(
            [
                (float(ln[20:28]), float(ln[28:36]), float(ln[36:44]))
                for ln in lines[2:-1]
            ]
        )
    center = np.mean(coords, axis=0)
    return 2.0 * np.max(np.sum((coords - center) ** 2, axis=1)) ** 0.5


def _read_molecule_counts(top_file: Path) -> dict[str, int]:
    """
    Read the molecule counts from the [ molecules ] section of a topology.
    :param top_file: Topology file written by COBY.
    :return: Mapping of molecule name -> count, in topology order.
    """
    counts, in_molecules = {}, False
    with open(top_file, "r", encoding="utf-8") as top:
        for line in top:
            line = line.split(";")[0].strip()
            if line.startswith("["):
                in_molecules = line.replace(" ", "") == "[molecules]"
            elif in_molecules and line:
                name, count = line.split()[:2]
                counts[name] = counts.get(name, 0) + int(count)
    return counts


def _solvate(
    solute_file: Path,
    system_gro: Path,
    box_margin: float,
    charge: int,
    mixture: dict[str, float],
    salt_concentration: float,
    seed: int,
) -> dict[str, int]:
    """
    Place a solute at the center of a cubic box and fill it with solvent and ions using
    COBY. The solvent is inserted at its equilibrium density, but an equilibration is
    still required to relax the packing.
    :param solute_file: Coordinate file of the solute (.gro or .pdb).
    :param system_gro: Output system coordinate file (.gro).
    :param box_margin: Distance (nm) added to the solute diameter for the box edge.
    :param charge: Net solute charge; neutralizing counter-ions are added.
    :param mixture: Solvent composition as residue name to relative molecule fraction.
    :param salt_concentration: NaCl concentration (mol/L) added on top of the
        charge-neutralizing counter-ions, based on the box volume.
    :param seed: Random seed for reproducible insertion.
    :return: Mapping of solvent and ion residue name -> molecule count placed.
    """
    unknown = [res for res in mixture if res not in MOLECULAR_VOLUMES]
    if unknown:
        raise ValueError(f"No molecular volume defined for solvent {unknown}.")
    # Cubic box edge = solute diameter + box_margin.
    box_length = _solute_diameter(solute_file) + box_margin
    # Molarity of solvent molecules from the mean molecular volume of the mixture. The
    # mapping ratio is disabled, so COBY counts CG molecules and treats the fractions
    # as molecule ratios.
    total_fraction = sum(mixture.values())
    fractions = {res: frac / total_fraction for res, frac in mixture.items()}
    avg_volume = sum(frac * MOLECULAR_VOLUMES[res] for res, frac in fractions.items())
    solvent_molarity = 1.0 / (avg_volume * IONS_PER_NM3_PER_MOLAR)
    solvents = " ".join(f"solv:{res}:{frac:.6f}" for res, frac in fractions.items())
    logger.debug(
        "Solvent box for charge %s: L=%.2f nm, mixture=%s, salt=%s M",
        charge,
        box_length,
        fractions,
        salt_concentration,
    )
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        coby_gro, coby_top = tmp / "system.gro", tmp / "system.top"
        # The solute charge is given explicitly, so the counter-ions follow the
        # 'neutralize' setting instead of the topology
        COBY.COBY(
            box=[box_length] * 3,
            box_type="rectangular",
            import_library=str(SOLVENT_LIBRARY),
            protein=f"file:{solute_file} charge:{charge}",
            solvation=(
                f"{solvents} pos:NA neg:CL salt_molarity:{salt_concentration} "
                f"ionsvol:box mapping:False solv_molarity:{solvent_molarity:.4f}"
            ),
            out_gro=str(coby_gro),
            out_top=str(coby_top),
            rand=seed,
            verbose=0,
        )
        if not coby_gro.exists() or not coby_top.exists():
            raise RuntimeError(f"COBY did not produce expected output in {tmp}")
        shutil.copy(coby_gro, system_gro)
        counts = _read_molecule_counts(coby_top)
    # Keep only the inserted molecules, the solute is listed by the caller
    return {
        res: counts[res] for res in [*fractions, "NA", "CL"] if counts.get(res, 0) > 0
    }


def _write_topology(
    system_top: Path,
    title: str,
    include_lines: list[str],
    solute_molecules: list[str],
    composition: dict[str, int],
):
    """
    Write a system topology with the solute molecules followed by solvent and ions.
    :param system_top: Output system topology file (.top).
    :param title: Name of the system.
    :param include_lines: Formatted `#include "..."` lines for the topology.
    :param solute_molecules: Moleculetype names of the solute, once each, in the order
        of the coordinate file.
    :param composition: Mapping of solvent and ion residue name -> molecule count.
    """
    molecule_lines = [f"{name:<6} 1" for name in solute_molecules]
    molecule_lines += [f"{res:<6} {count}" for res, count in composition.items()]
    solvent_info = ", ".join(f"{res}={cnt}" for res, cnt in composition.items())
    topology = (
        f"; {title} ({solvent_info})\n"
        + "\n".join(include_lines)
        + f"\n\n[ system ]\n{title}\n\n"
        + "[ molecules ]\n"
        + "\n".join(molecule_lines)
        + "\n"
    )
    Path(system_top).write_text(topology, encoding="utf-8")
    logger.debug("Wrote system topology: %s", system_top)


def setup_solvent_box(
    ligand_gro: Path,
    ligand_top: Path,
    system_gro: Path,
    system_top: Path,
    box_margin: float,
    charge: int,
    include_lines: list[str],
    mixture: dict[str, float],
    salt_concentration: float = 0.0,
    seed: int = 12345,
) -> dict[str, int]:
    """
    Build a solvent box around a ligand at the box center with COBY and write its
    topology.
    :param ligand_gro: Ligand coordinate input file (.gro).
    :param ligand_top: Ligand topology (.itp), used to read the moleculetype name.
    :param system_gro: Output system coordinate file (.gro).
    :param system_top: Output system topology file (.top).
    :param box_margin: Distance (nm) added to the ligand diameter for the box edge.
    :param charge: Net ligand charge; neutralizing counter-ions are added.
    :param include_lines: Formatted `#include "..."` lines for the topology
    :param mixture: Solvent composition as residue name to relative molecule fraction,
        e.g. {"W": 1.0} for water or {"OCO": 0.92, "W": 0.08} for water-saturated octanol.
    :param salt_concentration: NaCl concentration (mol/L) added on top of the
        charge-neutralizing counter-ions, based on the box volume.
    :param seed: Random seed for reproducible insertion.
    :return: Mapping of residue name -> molecule count actually placed.
    """
    composition = _solvate(
        ligand_gro, system_gro, box_margin, charge, mixture, salt_concentration, seed
    )
    _write_topology(
        system_top,
        "Ligand in solvent box",
        include_lines,
        [get_molecule_name_from_itp(ligand_top)],
        composition,
    )
    return composition


def setup_protein_box(
    protein_ligand_pdb: Path,
    system_gro: Path,
    system_top: Path,
    box_margin: float,
    charge: int,
    include_lines: list[str],
    protein_name: str,
    salt_concentration: float = 0.0,
    seed: int = 12345,
) -> dict[str, int]:
    """
    Solvate the protein-ligand complex in water with COBY and write its topology.
    :param protein_ligand_pdb: Coordinates of the protein followed by the ligand (.pdb).
    :param system_gro: Output system coordinate file (.gro).
    :param system_top: Output system topology file (.top).
    :param box_margin: Distance (nm) added to the complex diameter for the box edge.
    :param charge: Net charge of the complex; neutralizing counter-ions are added.
    :param include_lines: Formatted `#include "..."` lines for the topology.
    :param protein_name: Moleculetype name of the protein.
    :param salt_concentration: NaCl concentration (mol/L) added on top of the
        charge-neutralizing counter-ions, based on the box volume.
    :param seed: Random seed for reproducible insertion.
    :return: Mapping of residue name -> molecule count actually placed.
    """
    composition = _solvate(
        protein_ligand_pdb,
        system_gro,
        box_margin,
        charge,
        {"W": 1.0},
        salt_concentration,
        seed,
    )
    _write_topology(
        system_top,
        "Ligand in protein-water system",
        include_lines,
        [protein_name, "ligand"],
        composition,
    )
    return composition
