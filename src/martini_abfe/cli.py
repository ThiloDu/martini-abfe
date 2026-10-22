"""Command line interface: prepare, run and analyze binding free energy calculations."""

import asyncio
import json
import logging
import shutil
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from omegaconf import OmegaConf

from .config import load_config
from .correction import underbinding_correction
from .free_energies import binding_free_energy, calculate_free_energies
from .setup import GO_FILES, setup_protein_structure
from .simulate import rerun_keys, simulate_ligand
from .structure import (
    center_of_mass,
    normalize_ligand_topology,
    place_cg_ligand,
    read_gro_coordinates,
    split_complex,
    write_ligand_gro,
)
from .utils import local_path, read_json_file

logger = logging.getLogger("martini_abfe")

app = typer.Typer(
    help="Absolute protein-ligand binding free energies with Martini 3.",
    no_args_is_help=True,
)

PROTEIN_FILES = [
    "protein-clean.pdb",
    "protein-cg.pdb",
    "protein.itp",
    "protein_info.json",
    *GO_FILES,
]


class Orientation(str, Enum):
    auto = "auto"
    mapping = "mapping"
    none = "none"


class QueueType(str, Enum):
    file = "file"
    none = "none"


@app.callback()
def main(verbose: bool = typer.Option(False, "--verbose", "-v", help="Debug output")):
    """Absolute protein-ligand binding free energies with Martini 3."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for noisy in ("MDAnalysis", "matplotlib", "pymbar", "jax"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def write_ligand_info(ligand_dir: Path, name: str):
    """
    Write the info file of a ligand with its bead types and charge, read from its topology.
    :param ligand_dir: Directory containing the ligand topology.
    :param name: Name of the system.
    """
    beads = []
    charges = []
    with open(ligand_dir / "ligand.itp", "r", encoding="utf-8") as itp_file:
        read_mode = False
        for line in itp_file:
            line = line.split(";")[0].strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("["):
                read_mode = "atoms" in line
            elif read_mode:
                parts = line.split()
                beads.append(parts[1])
                charges.append(float(parts[6]))
    total_charge = sum(charges)
    if not np.isclose(total_charge, round(total_charge), atol=1e-3):
        raise ValueError(f"Total ligand charge {total_charge} is not an integer")
    ligand_info = {
        "name": name,
        "beads": beads,
        "charge": int(round(total_charge)),
        "charged": any(c != 0 for c in charges),
    }
    with open(ligand_dir / "info.json", "w", encoding="utf-8") as info_file:
        json.dump(ligand_info, info_file, indent=2)


@app.command()
def prepare(
    complex_pdb: Path = typer.Option(
        ..., "--complex", "-c", exists=True, help="Atomistic protein-ligand complex (pdb)"
    ),
    ligand_resname: str = typer.Option(
        ..., "--ligand-resname", "-r", help="Residue name of the ligand in the complex"
    ),
    ligand_itp: Path = typer.Option(
        ..., "--itp", exists=True, help="Coarse-grained ligand topology"
    ),
    ligand_gro: Path = typer.Option(
        ..., "--gro", exists=True, help="Coarse-grained ligand structure"
    ),
    output: Path = typer.Option(..., "--output", "-o", help="System directory"),
    mapping: Optional[Path] = typer.Option(
        None,
        "--mapping",
        "-m",
        exists=True,
        help="Bead mapping (ndx, one group per bead with 1-based atom indices of the "
        "ligand in the complex), used to superimpose the ligand onto the crystal pose",
    ),
    orient: Optional[Orientation] = typer.Option(
        None,
        help="Placement of the CG ligand: 'mapping' (default with --mapping), 'auto' "
        "(fit to the atomistic ligand shape, default otherwise) or 'none' (translate only)",
    ),
    ligand_chain: Optional[str] = typer.Option(None, help="Chain of the ligand"),
    ligand_resid: Optional[str] = typer.Option(None, help="Residue number of the ligand"),
    chain: Optional[str] = typer.Option(
        None, help="Protein chains to keep, comma separated (default: all)"
    ),
    protein_itp: Optional[Path] = typer.Option(
        None,
        exists=True,
        help="Predefined CG protein topology (single molecule), used with --protein-gro "
        "instead of martinize2 and OLIVES",
    ),
    protein_gro: Optional[Path] = typer.Option(
        None,
        exists=True,
        help="Predefined CG protein structure, in the frame of the complex",
    ),
    go_atomtypes: Optional[Path] = typer.Option(
        None,
        exists=True,
        help="go_atomtypes.itp of a predefined GoMartini protein, with --go-nbparams",
    ),
    go_nbparams: Optional[Path] = typer.Option(
        None,
        exists=True,
        help="go_nbparams.itp of a predefined GoMartini protein, with --go-atomtypes",
    ),
    config_file: Optional[Path] = typer.Option(
        None, "--config", exists=True, help="YAML file overriding the default settings"
    ),
    name: Optional[str] = typer.Option(None, help="System name, default: directory name"),
    force: bool = typer.Option(False, help="Overwrite a prepared system directory"),
):
    """
    Prepare a system directory from a protein-ligand complex and a coarse-grained ligand
    model: split the complex, place the CG ligand in the pocket, martinize the protein
    (or use a predefined CG protein) and add the pocket reference bead.
    """
    if (protein_itp is None) != (protein_gro is None):
        raise typer.BadParameter("Give both --protein-itp and --protein-gro, or neither")
    if (go_atomtypes is None) != (go_nbparams is None):
        raise typer.BadParameter("Give both --go-atomtypes and --go-nbparams, or neither")
    if go_atomtypes is not None and protein_itp is None:
        raise typer.BadParameter(
            "--go-atomtypes and --go-nbparams need --protein-itp, set martinize.go in "
            "the config to build a GoMartini model with martinize2"
        )
    output = output.resolve()
    if (output / "config.yaml").exists():
        if not force:
            raise typer.BadParameter(f"{output} is already prepared, use --force")
        for filename in PROTEIN_FILES + ["info.json"]:
            (output / filename).unlink(missing_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy(complex_pdb, output / "complex.pdb")
    # Split complex into protein and ligand
    ligand_atoms = split_complex(
        output / "complex.pdb",
        ligand_resname,
        output / "protein.pdb",
        output / "ligand-aa.pdb",
        ligand_chain=ligand_chain,
        ligand_resid=ligand_resid,
    )
    ligand_com = center_of_mass(ligand_atoms)
    logger.info("Atomistic ligand center of mass: %s nm", np.round(ligand_com, 3))
    # Copy coarse-grained ligand with normalized names and place it in the pocket
    itp_atoms = normalize_ligand_topology(ligand_itp, output / "ligand.itp")
    n_beads = len(read_gro_coordinates(ligand_gro))
    if n_beads != len(itp_atoms):
        raise ValueError(
            f"{ligand_gro} has {n_beads} beads, {ligand_itp} defines {len(itp_atoms)}"
        )
    if orient is None:
        orient = Orientation.mapping if mapping is not None else Orientation.auto
    if mapping is not None:
        shutil.copy(mapping, output / "mapping.ndx")
    coordinates = place_cg_ligand(ligand_gro, ligand_atoms, orient.value, mapping)
    write_ligand_gro(
        output / "ligand.gro",
        coordinates,
        [atom[4] for atom in itp_atoms],
        f"CG ligand placed at the {ligand_resname} pose of {complex_pdb.name}",
    )
    logger.info(
        "CG ligand centroid is %.3f nm from the atomistic center of mass",
        np.linalg.norm(coordinates.mean(axis=0) - ligand_com),
    )
    write_ligand_info(output, name or output.name)
    # Store the configuration of the system
    config = load_config(config_file)
    sim_config = config.simulation
    sim_config.directory = str(output)
    sim_config.protein.file = str(output / "protein.pdb")
    sim_config.protein.ligand = ligand_com.tolist()
    if chain is not None:
        sim_config.protein.chain = chain
    if protein_itp is not None:
        sim_config.protein.cg_itp = str(protein_itp)
        sim_config.protein.cg_gro = str(protein_gro)
    if go_atomtypes is not None:
        sim_config.protein.go_atomtypes = str(go_atomtypes)
        sim_config.protein.go_nbparams = str(go_nbparams)
    for key in ("cg_itp", "cg_gro", "go_atomtypes", "go_nbparams"):
        if sim_config.protein.get(key):
            sim_config.protein[key] = str(Path(sim_config.protein[key]).resolve())
    if sim_config.queue.get("path"):
        sim_config.queue.path = str(Path(sim_config.queue.path).resolve())
    OmegaConf.save(config, output / "config.yaml")
    # Coarse-grain the protein and add the pocket reference bead
    setup_protein_structure(config)
    protein_info = read_json_file(output / "protein_info.json")
    ligand_info = read_json_file(output / "info.json")
    underbinding = underbinding_correction(output)
    charged = ", ".join(
        f"{r['resname']}{r['resid']}{r['chain']}" for r in underbinding["charged_residues"]
    )
    typer.echo(
        f"Prepared {output}\n"
        f"  protein charge {protein_info['charge']:+d}, ligand charge "
        f"{ligand_info['charge']:+d} ({len(ligand_info['beads'])} beads)\n"
        f"  pocket reference bead {protein_info['dummy_offset']:.3f} nm from the "
        f"ligand COM, built from beads {protein_info['dummy_reference_beads']}\n"
        f"  {underbinding['n_charged']} charged pocket residues ({charged or 'none'}), "
        f"predicted underbinding {underbinding['offset']:.1f} ± "
        f"{underbinding['uncertainty']:.1f} kcal/mol\n"
        f"Check protein-cg.pdb and ligand.gro, then run: martini-abfe run {output}"
    )


def _system_config(system: Path, queue: Optional[QueueType], queue_file: Optional[Path]):
    """
    Load the stored configuration of a prepared system.
    :param system: System directory.
    :param queue: Queue type overriding the stored one.
    :param queue_file: Queue file overriding the stored one.
    :return: Configuration object.
    """
    if not (system / "config.yaml").exists():
        raise typer.BadParameter(f"{system} is not prepared, run 'prepare' first")
    config = load_config(system / "config.yaml")
    config.simulation.directory = str(system)
    if queue is not None:
        config.simulation.queue.type = queue.value
    if queue_file is not None:
        config.simulation.queue.path = str(queue_file.resolve())
    return config


@app.command()
def run(
    systems: list[Path] = typer.Argument(..., help="Prepared system directories"),
    queue: Optional[QueueType] = typer.Option(
        None,
        help="'file' appends production runs to a queue file processed by "
        "martini-abfe-queue workers, 'none' runs everything locally",
    ),
    queue_file: Optional[Path] = typer.Option(None, help="Queue file to use"),
    gromacs: Optional[str] = typer.Option(
        None, help="Extra mdrun arguments for minimization and equilibration"
    ),
    no_wait: bool = typer.Option(
        False,
        "--no-wait",
        help="Only set up the legs and queue their lambda windows, without waiting for "
        "them. Compute the free energies later with 'martini-abfe analyze'.",
    ),
):
    """
    Simulate the ligand in the protein pocket and in water, then compute the free
    energies. With the file queue, this waits until workers finished all lambda windows.
    """
    tasks = []
    for system in systems:
        system = system.resolve()
        config = _system_config(system, queue, queue_file)
        if gromacs is not None:
            config.simulation.gromacs = gromacs
        if no_wait and config.simulation.queue.type != "file":
            raise typer.BadParameter("--no-wait requires the file queue")
        if no_wait and config.simulation.protein.get("run_mode") == "serial":
            raise typer.BadParameter(
                "--no-wait is not possible with run_mode 'serial', where each lambda "
                "window starts from the previous one"
            )
        if config.simulation.queue.type == "file":
            queue_path = config.simulation.queue.path
            typer.echo(
                f"Queueing {system.name} in {local_path(queue_path)}, process it "
                f"with: martini-abfe-queue -f {local_path(queue_path)}"
            )
        tasks.append(
            simulate_ligand(ligand_directory=system, config=config, wait=not no_wait)
        )

    async def gather():
        await asyncio.gather(*tasks)

    asyncio.run(gather())
    if no_wait:
        names = " ".join(local_path(system) for system in systems)
        typer.echo(
            "All lambda windows are queued. Once the workers finished them, compute "
            f"the free energies with: martini-abfe analyze {names}"
        )
        return
    _report(systems)


def _report(systems: list[Path]):
    """
    Compute and print the binding free energies of finished systems.
    :param systems: System directories.
    """
    typer.echo(
        f"{'system':<30} {'dG_protein':>12} {'dG_water':>12} {'dG_bind':>18} "
        f"{'N_charged':>9} {'dG_bind_corr':>18}"
    )
    for system in systems:
        result = binding_free_energy(system.resolve())
        if result is None:
            typer.echo(f"{system.name:<30} {'not finished':>12}")
            continue
        line = (
            f"{result['name']:<30} {result['protein']['dG']:12.2f} "
            f"{result['water']['dG']:12.2f} "
            f"{result['dG_bind']:10.2f} ± {result['ddG_bind']:.2f}"
        )
        if "dG_bind_corrected" in result:
            line += (
                f" {result['underbinding']['n_charged']:9d} "
                f"{result['dG_bind_corrected']:10.2f} ± "
                f"{result['ddG_bind_corrected']:.2f}"
            )
        typer.echo(line)
    typer.echo(
        "Free energies in kcal/mol, dG_bind = dG_water - dG_protein. dG_bind_corr "
        "subtracts the underbinding of Martini 3 predicted from the N_charged charged "
        "pocket residues (arXiv:2610.11714), which shifts all ligands of a target equally."
    )


@app.command()
def analyze(
    systems: list[Path] = typer.Argument(..., help="Prepared system directories"),
):
    """Compute the free energies of finished simulations and the binding free energy."""
    for system in systems:
        system = system.resolve()
        config = _system_config(system, None, None)
        info = read_json_file(system / "info.json")
        legs = []
        for name in ("protein", "water"):
            legs += [k for k in rerun_keys(config.simulation, name) if k not in info]
        try:
            calculate_free_energies(system, legs)
        except RuntimeError as e:
            logger.warning(str(e))
    _report(systems)


@app.command()
def show_config(
    config_file: Optional[Path] = typer.Option(None, "--config", exists=True),
):
    """Print the default configuration, optionally merged with a YAML file."""
    typer.echo(OmegaConf.to_yaml(load_config(config_file)))


if __name__ == "__main__":
    app()
