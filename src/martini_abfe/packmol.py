"""Build packmol-packed solvent boxes (e.g. octanol/water) around a ligand for Martini 3."""

import logging
import tempfile
from pathlib import Path
from subprocess import run

import numpy as np

from .utils import get_molecule_name_from_itp

logger = logging.getLogger(__name__)


# Single-molecule templates: residue name -> list of (atom name, xyz in Angstrom).
SOLVENT_TEMPLATES = {
    "W": [("W", (0.0, 0.0, 0.0))],
    "OCO": [("C1", (0.0, 0.0, 0.0)), ("C2", (3.9, 0.0, 0.0)), ("PC", (7.4, 0.0, 0.0))],
    "NA": [("NA", (0.0, 0.0, 0.0))],
    "CL": [("CL", (0.0, 0.0, 0.0))],
}
# Average molecular volume (nm^3) per solvent bead, used to estimate molecule counts.
MOLECULAR_VOLUMES = {"W": 0.120, "OCO": 0.262}
# Ion pairs per nm^3 at 1 mol/L (Avogadro * 1e-24 L/nm^3), for salt-concentration counts.
IONS_PER_NM3_PER_MOLAR = 0.6022


def _template_pdb(resname: str) -> str:
    """
    Build a one-molecule pdb string for a residue from its template.
    :param resname: Residue name of the solvent molecule.
    :return: Content of a pdb file holding the single molecule.
    """
    lines = []
    for serial, (name, (x, y, z)) in enumerate(SOLVENT_TEMPLATES[resname], start=1):
        name_field = f" {name:<3}" if len(name) < 4 else name[:4]
        lines.append(
            f"ATOM  {serial:>5} {name_field}{resname:>4} A{1:>4}    "
            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00"
        )
    return "\n".join(lines) + "\nEND\n"


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
    packmol_bin: str = "packmol",
    gmx_bin: str = "gmx",
    density_fill: float = 0.85,
    tolerance: float = 2.5,
    edge_buffer: float = 0.2,
    seed: int = 12345,
) -> dict[str, int]:
    """
    Build a solvent box around a ligand fixed at the box center and write its topology.
    Solvent counts are estimated to roughly fill the box, i.e. an equilibration is
    required to compress it to the correct density.
    :param ligand_gro: Ligand coordinate input file (.gro).
    :param ligand_top: Ligand topology (.itp), used to read the moleculetype name.
    :param system_gro: Output system coordinate file (.gro).
    :param system_top: Output system topology file (.top).
    :param box_margin: Minimum distance (nm) between ligand and box edge.
    :param charge: Net ligand charge; neutralizing counter-ions are added.
    :param include_lines: Formatted `#include "..."` lines for the topology
    :param mixture: Solvent composition as residue name to relative bead fraction, e.g.
        {"W": 1.0} for water or {"OCO": 0.92, "W": 0.08} for water-saturated octanol.
    :param salt_concentration: NaCl concentration (mol/L) added on top of the
        charge-neutralizing counter-ions, based on the final box volume.
    :param packmol_bin: Path to the packmol executable, defaults to "packmol".
    :param gmx_bin: Path to the GROMACS executable (for pdb<->gro conversion), defaults to "gmx".
    :param density_fill: Fraction of the equilibrium density to pack at (<1 leaves
        headroom so packmol converges and NPT can compress the box).
    :param tolerance: packmol minimum inter-molecular distance (Angstrom).
    :param edge_buffer: Margin (nm) kept between solvent and the box edges.
    :param seed: Random seed for reproducible packing.
    :return: Mapping of residue name -> molecule count actually placed.
    """
    # Cubic box edge = solute diameter + box_margin.
    box_length = _solute_diameter(ligand_gro) + box_margin
    # Normalize the bead fractions and estimate counts from box volume and mean bead volume.
    total_fraction = sum(mixture.values())
    fractions = {res: frac / total_fraction for res, frac in mixture.items()}
    avg_volume = sum(frac * MOLECULAR_VOLUMES[res] for res, frac in fractions.items())
    n_total = density_fill * box_length**3 / avg_volume
    composition = {res: round(frac * n_total) for res, frac in fractions.items()}
    # Add NaCl at the requested molarity, then neutralize the ligand charge.
    n_salt = round(salt_concentration * box_length**3 * IONS_PER_NM3_PER_MOLAR)
    n_na = n_salt + max(-charge, 0)
    n_cl = n_salt + max(charge, 0)
    if n_na > 0:
        composition["NA"] = n_na
    if n_cl > 0:
        composition["CL"] = n_cl
    logger.debug(
        "Solvent box for charge %s: L=%.2f nm, composition=%s",
        charge,
        box_length,
        composition,
    )
    # Determine the box limits for packmol in Angstrom
    length_a, buffer_a = box_length * 10.0, edge_buffer * 10.0
    lo, hi = buffer_a, length_a - buffer_a
    center = length_a / 2.0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        ligand_pdb, packed_pdb = tmp / "ligand.pdb", tmp / "packed.pdb"
        # packmol works with pdb files, so convert the ligand gro to pdb first
        run(
            f"{gmx_bin} editconf -f {ligand_gro} -o {ligand_pdb}",
            shell=True,
            check=True,
        )
        # Build the packmol input
        blocks = [
            f"tolerance {tolerance}",
            f"seed {seed}",
            f"output {packed_pdb}",
            "filetype pdb",
            "",
            f"structure {ligand_pdb}",
            "  number 1",
            "  center",
            f"  fixed {center:.3f} {center:.3f} {center:.3f} 0. 0. 0.",
            "end structure",
            "",
        ]
        for resname, count in composition.items():
            if count <= 0:
                continue
            if resname not in SOLVENT_TEMPLATES:
                raise ValueError(
                    f"No packmol template defined for residue '{resname}'."
                )
            template_pdb = tmp / f"{resname}.pdb"
            template_pdb.write_text(_template_pdb(resname), encoding="utf-8")
            blocks += [
                f"structure {template_pdb}",
                f"  number {count}",
                f"  inside box {lo:.3f} {lo:.3f} {lo:.3f} {hi:.3f} {hi:.3f} {hi:.3f}",
                "end structure",
                "",
            ]
        packmol_input = tmp / "packmol.inp"
        packmol_input.write_text("\n".join(blocks), encoding="utf-8")
        # packmol reads its directives from stdin.
        with open(packmol_input, "r", encoding="utf-8") as inp_file:
            run(packmol_bin, stdin=inp_file, shell=True, check=True)
        if not packed_pdb.exists():
            raise RuntimeError(f"packmol did not produce expected output: {packed_pdb}")
        # Convert the packed pdb back to gro format
        run(
            f"{gmx_bin} editconf -f {packed_pdb} -o {system_gro} "
            f"-box {box_length:.3f} {box_length:.3f} {box_length:.3f} -noc",
            shell=True,
            check=True,
        )
    # Write the topology file with the included lines and molecule counts
    molecule_lines = [f"{get_molecule_name_from_itp(ligand_top):<6} 1"]
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
