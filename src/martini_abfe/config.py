"""Default configuration of the binding free energy protocol."""

import os
from importlib.resources import files
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

MDP_DIRECTORY = Path(str(files("martini_abfe") / "mdps"))
REPOSITORY_ROOT = Path(__file__).resolve().parent.parent.parent


def default_olives_script() -> str | None:
    """
    Locate the OLIVES script, from the environment variable MARTINI_ABFE_OLIVES or the
    OLIVES submodule of the repository.
    :return: Path to the OLIVES script, or None if it cannot be found.
    """
    if os.environ.get("MARTINI_ABFE_OLIVES"):
        return os.environ["MARTINI_ABFE_OLIVES"]
    script = REPOSITORY_ROOT / "external/OLIVES/OLIVES_v2.0.1_M3.0.0.py"
    return str(script) if script.exists() else None


def default_config() -> DictConfig:
    """
    Default configuration of the protocol: a bound leg ('protein') simulated in three and
    an unbound leg ('water') simulated in two independent repeats.
    :return: Configuration object.
    """
    return OmegaConf.create(
        {
            "simulation": {
                "directory": None,
                "gromacs": "-ntomp 2 -noddcheck -rdd 2.0",
                "queue": {
                    "type": "file",
                    "path": "simulation-queue.dat",
                    "readd": False,
                },
                "martini": {
                    "topology": "/mnt/plutus/tduve/forcefields/1_martini_v3.0.0/martini_v3.0.0.itp",
                    "solvents": "/mnt/plutus/tduve/forcefields/1_martini_v3.0.0/martini_v3.0.0_solvents_v1.itp",
                    "ions": "/mnt/plutus/tduve/forcefields/1_martini_v3.0.0/martini_v3.0.0_ions_v1.itp",
                },
                "protein": {
                    "file": None,
                    "ligand": None,
                    "chain": None,
                    "recenter_ligand": False,
                    "dummy_bead": {
                        "n_virtual_sites": 8,
                        "cutoff": 1.1,
                        "max_cutoff": 2.5,
                        "regularization": 0.005,
                    },
                    "run_mode": "parallel",
                    "martinize": {
                        "name": "protein",
                        "dssp": True,
                        "ff": "martini3001",
                        "ignore": "NME,ACE",
                        "maxwarn": 4,
                    },
                    "n_reruns": 3,
                    "olives": "auto",
                    "box_margin": 2.1,
                    "salt": 0.15,
                    "neutralize": True,
                    "minimization": str(MDP_DIRECTORY / "minimization.mdp"),
                    "equilibration": str(MDP_DIRECTORY / "protein-equilibration.mdp"),
                    "production": str(MDP_DIRECTORY / "protein-production.mdp"),
                    "production_charged": str(
                        MDP_DIRECTORY / "protein-production-charged.mdp"
                    ),
                },
                "water": {
                    "composition": {"W": 1.0},
                    "box_margin": 4.5,
                    "neutralize": True,
                    "salt": 0.15,
                    "n_reruns": 2,
                    "minimization": str(MDP_DIRECTORY / "minimization.mdp"),
                    "equilibration": str(MDP_DIRECTORY / "water-equilibration.mdp"),
                    "production": str(MDP_DIRECTORY / "water-production.mdp"),
                    "production_charged": str(
                        MDP_DIRECTORY / "water-production-charged.mdp"
                    ),
                },
            }
        }
    )


def load_config(*config_files: str | Path | None) -> DictConfig:
    """
    Load the default configuration, updated with the given configuration files.
    :param config_files: YAML files merged in order, None entries are skipped.
    :return: Configuration object.
    """
    configs = [default_config()]
    for config_file in config_files:
        if config_file is not None:
            configs.append(OmegaConf.load(config_file))
    return OmegaConf.merge(*configs)
