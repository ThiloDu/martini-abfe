"""Build COBY solvent boxes (e.g. water or octanol/water) around a ligand for Martini 3."""

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


def _solute_diameter(gro_file: Path) -> float:
    """
    Calculate the solute diameter in the same way as insane does it.
    :param gro_file: Coordinate file of the solute.
    :return: Diameter of the solute in nm.
    """
    with open(gro_file, "r", encoding="utf-8") as coord_file:
        lines = [line for line in coord_file if len(line.strip()) > 0]
    coords = np.array(
        [(float(ln[20:28]), float(ln[28:36]), float(ln[36:44])) for ln in lines[2:-1]]
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
    topology. The solvent is inserted at its equilibrium density, but an equilibration is
    still required to relax the packing.
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
    unknown = [res for res in mixture if res not in MOLECULAR_VOLUMES]
    if unknown:
        raise ValueError(f"No molecular volume defined for solvent {unknown}.")
    # Cubic box edge = solute diameter + box_margin.
    box_length = _solute_diameter(ligand_gro) + box_margin
    # Molarity of solvent molecules from the mean molecular volume of the mixture. The
    # mapping ratio is disabled, so COBY counts CG molecules and treats the fractions
    # as molecule ratios.
    total_fraction = sum(mixture.values())
    fractions = {res: frac / total_fraction for res, frac in mixture.items()}
    avg_volume = sum(frac * MOLECULAR_VOLUMES[res] for res, frac in fractions.items())
    solvent_molarity = 1.0 / (avg_volume * IONS_PER_NM3_PER_MOLAR)
    solvents = " ".join(f"solv:{res}:{frac:.6f}" for res, frac in fractions.items())
    molecule_name = get_molecule_name_from_itp(ligand_top)
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
        # The ligand charge is given explicitly, so the counter-ions follow the
        # 'neutralize' setting instead of the topology
        COBY.COBY(
            box=[box_length] * 3,
            box_type="rectangular",
            import_library=str(SOLVENT_LIBRARY),
            protein=f"file:{ligand_gro} moleculetypes:{molecule_name} charge:{charge}",
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
        composition = _read_molecule_counts(coby_top)
    composition.pop(molecule_name, None)
    # Write the topology file with the included lines and molecule counts
    molecule_lines = [f"{molecule_name:<6} 1"]
    molecule_lines += [
        f"{resname:<6} {count}" for resname, count in composition.items() if count > 0
    ]
    solvent_info = ", ".join(f"{res}={cnt}" for res, cnt in composition.items())
    topology = (
        f"; Ligand in solvent box ({solvent_info})\n"
        + "\n".join(include_lines)
        + "\n\n[ system ]\nLigand in solvent box\n\n"
        + "[ molecules ]\n"
        + "\n".join(molecule_lines)
        + "\n"
    )
    Path(system_top).write_text(topology, encoding="utf-8")
    logger.debug("Wrote system topology: %s", system_top)

    return composition
