"""Setup and execution of GROMACS simulations."""

import logging
import os
import subprocess
from pathlib import Path
from time import sleep

from omegaconf import OmegaConf

from .file_queue import add_simulation_task_to_file_queue, gromacs_env

logger = logging.getLogger(__name__)


class Simulation:
    """
    Class for running GROMACS simulations using python.
    """

    def __init__(
        self, *input_files: str | Path, sim_config: OmegaConf, **simulation_kwargs
    ):
        """
        Initialize a simulation from its input files, whose role is determined by their
        file extension. The basename of the 'tpr' file is used for all output files.
        :param input_files: Input files, 'tpr', 'top', 'gro' and 'mdp' are required,
            'ndx' (index), 'res' (restraint reference) and 'log' (output log) are optional.
        :param sim_config: Simulation configuration from the config file.
        :param simulation_kwargs: Supported are 'restrain' to restrain the system with the
            coordinates of the 'gro' file and 'maxwarn' for the grompp warning limit.
        """
        # Store input files in a dictionary with their extensions as keys
        self.input_files = {
            str(filename).rsplit(".", 1)[-1]: str(filename) for filename in input_files
        }
        # Check for required input files
        required_extensions = ["tpr", "top", "gro", "mdp"]
        for ext in self.input_files.keys():
            if ext in required_extensions:
                required_extensions.remove(ext)
        if len(required_extensions) > 0:
            raise ValueError(
                f"Missing required input files with extensions: {', '.join(required_extensions)}"
            )
        # Set other arguments as attributes
        self.sim_config = sim_config
        self.simulation_kwargs = simulation_kwargs
        self.prepared = False
        # Setup output coordinate file path
        self.out_coord = os.path.splitext(self.input_files["tpr"])[0] + ".gro"

    def modify_mdp(self, key: str | dict, value: str = None, output_file: str = None):
        """
        Modify the value of a key in the MDP file.
        :param key: Key in the MDP file to modify. Alternatively, a dictionary of key-value
            pairs can be provided to modify multiple keys at once.
        :param value: New value for the key. Only if `key` is a string.
        :param output_file: If provided, the modified MDP file is written to this file.
            Otherwise, the original MDP file is modified in place.
        """
        if isinstance(key, str):
            if value is None:
                raise ValueError("Value must be provided if key is a string")
            key = {key: value}
        # Load original MDP file
        with open(self.input_files["mdp"], "r", encoding="utf-8") as f:
            lines = f.readlines()
        # Determine output file
        if output_file is None:
            output_file = self.input_files["mdp"]
        # Iterate over lines and modify if key is found
        with open(output_file, "w", encoding="utf-8") as f:
            for line in lines:
                found = False
                for k, v in key.items():
                    if line.startswith(k):
                        f.write(f"{line.split('=')[0]}= {v}\n")
                        found = True
                        break
                if not found:
                    f.write(line)

    def prepare(self):
        """
        Prepare the GROMACS simulation by creating the TPR file. This method runs the
        'gmx_mpi grompp' command with the provided input files and simulation parameters.
        """
        maxwarn = self.simulation_kwargs.get("maxwarn", 1)
        command = (
            f"gmx_mpi grompp -f {self.input_files['mdp']} -c {self.input_files['gro']} "
            + f"-p {self.input_files['top']} -o {self.input_files['tpr']} "
            + f"-po {self.input_files['tpr'].replace('tpr', 'out.mdp')} -maxwarn {maxwarn}"
        )
        if "ndx" in self.input_files.keys():
            command += f" -n {self.input_files['ndx']}"
        if "res" in self.input_files.keys():
            command += f" -r {self.input_files['res']}"
        elif self.simulation_kwargs.get("restrain", False):
            command += f" -r {self.input_files['gro']}"
        if "log" in self.input_files.keys() and ">>" not in command:
            command += f" >> {self.input_files['log']} 2>&1"
        subprocess.run(command, shell=True, check=True)
        self.prepared = True

    def run(
        self,
        ignore_queue: bool = False,
        blocking: bool | None = None,
        extra_args: str = "",
    ):
        """
        Run the simulation with 'gmx_mpi mdrun'. If a queue system is specified in the simulation
        configuration, the simulation is added to that queue instead of running locally.
        :param ignore_queue: If True, run locally even if a queue system is configured.
        :param blocking: If True, wait for the simulation to finish before returning. Only
            relevant if a queue system is used, local simulations are always blocking.
        :param extra_args: Additional command line arguments for the mdrun command.
        """
        if not self.prepared:
            self.prepare()
        filename = os.path.splitext(self.input_files["tpr"])[0]
        if "queue" not in self.sim_config or "type" not in self.sim_config.queue:
            ignore_queue = True
        if ignore_queue or self.sim_config.queue.type == "none":
            # Run simulation locally
            if blocking is not None and not blocking:
                logger.warning("Local simulations are always blocking.")
            # Pull output names as in the queue runner, read by the restraint correction
            command = (
                f"gmx_mpi mdrun -deffnm {filename} -px {filename}.px.xvg "
                f"-pf {filename}.pf.xvg {extra_args}"
            )
            if "log" in self.input_files.keys() and ">>" not in command:
                command += f" >> {self.input_files['log']} 2>&1"
            logger.debug("Running command: %s", command)
            subprocess.run(command, shell=True, check=True, env=gromacs_env(command))
        elif self.sim_config.queue.type == "file":
            # Add simulation to queue
            queue_filename = Path(self.sim_config.queue.path)
            add_simulation_task_to_file_queue(filename, queue_filename)
            logger.debug(
                "Added simulation task %s to queue %s", filename, queue_filename
            )
            if blocking is not None and blocking:
                wait_file = os.path.splitext(self.input_files["tpr"])[0] + ".gro"
                while not os.path.exists(wait_file):
                    sleep(0.5)
        else:
            raise ValueError(
                f"Unknown queue type {self.sim_config.queue.type}. Currently supported "
                "types are 'none' and 'file'."
            )
