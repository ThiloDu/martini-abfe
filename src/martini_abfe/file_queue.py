"""Simulator file for running GROMACS simulations from the simulation queue."""

import argparse
import os
import socket
import subprocess
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from time import sleep
from typing import Optional


@contextmanager
def _queue_lock(lock_file: str):
    """
    Wait until the lock file is removed.
    :param lock_file: Path to the lock file.
    """
    delay = 0.1
    while True:
        try:
            with open(lock_file, "xb"):
                pass
            break
        except FileExistsError:
            sleep(delay)
            delay = min(delay * 2, 1.0)
    try:
        yield
    finally:
        os.remove(lock_file)


def add_simulation_task_to_file_queue(task_name: str | list[str], queue_filename: Path):
    """
    Add a simulation task to the queue.
    :param task_name: Filename of the simulation task to add to the queue. The file should be a
       '.tpr' file, but without the extension. A list of files can also be provided, in which
       case all files are added to the queue.
    :param queue_filename: Path to the queue file.
    """
    if task_name is None:
        return
    if not isinstance(task_name, list):
        task_name = [task_name]
    # Acquire lock by creating a lock file
    lock_filename = queue_filename.with_suffix(queue_filename.suffix + ".lock")
    with _queue_lock(lock_filename):
        # Append the task to the queue file
        with open(queue_filename, "a", encoding="utf-8") as queue_list_file:
            for task in task_name:
                try:
                    task = Path(task).resolve()
                    queue_list_file.write(str(task) + "\n")
                except ValueError:
                    print(f"ValueError: {task} is not a valid path")


def get_simulation_task_from_file_queue(queue_filename: Path) -> tuple[str, str] | None:
    """
    Get the next simulation task from the queue.
    :param queue_filename: Path to the queue file.
    :return: Filename of the next simulation task to run. If the queue is empty
        or does not exist, None is returned.
    """
    # Check if the queue file exists
    if not os.path.exists(queue_filename):
        return None
    # Acquire lock by creating a lock file
    lock_filename = queue_filename.with_suffix(queue_filename.suffix + ".lock")
    with _queue_lock(lock_filename):
        # Read the queue file and get the first task
        with open(queue_filename, "r+", encoding="utf-8") as queue_list_file:
            task_list = queue_list_file.readlines()
            # Return if the queue is empty
            if len(task_list) == 0:
                return None
            # Sort the task list so that tasks with "protein" in their path are prioritized
            task_list.sort(key=lambda e: e.count("protein"), reverse=True)
            # Get the first task
            task_name = task_list[0].strip()
            # Write the remaining tasks back to the queue file
            queue_list_file.seek(0)
            queue_list_file.truncate()
            for task in task_list[1:]:
                queue_list_file.write(task)
    # Return the task directory and the task basename
    dirname = os.path.dirname(task_name)
    basename = os.path.basename(task_name)
    return dirname, basename


def get_simulation_batch_from_file_queue(
    queue_filename: Path,
) -> tuple[str, list[str], str] | None:
    """
    Get the next batch of simulation tasks for replica exchange from the queue. A batch consists
    of all tasks that share the same parent directory. This is used for running replica exchange
    simulations, where each batch corresponds to a set of replicas.
    :param queue_filename: Path to the queue file.
    :return: A tuple containing the parent directory, a list of task directories, and the common
        basename. If the queue is empty or does not exist, None is returned.
    """
    # Check if the queue file exists
    if not os.path.exists(queue_filename):
        return None
    # Acquire lock by creating a lock file
    lock_filename = queue_filename.with_suffix(queue_filename.suffix + ".lock")
    with _queue_lock(lock_filename):
        # Read the queue file and get the first task
        with open(queue_filename, "r+", encoding="utf-8") as queue_list_file:
            task_list = queue_list_file.readlines()
            # Return if the queue is empty
            if len(task_list) == 0:
                return None
            # Sort the task list so that tasks with "protein" in their path are prioritized
            task_list.sort(key=lambda e: e.count("protein"), reverse=True)
            # Get the first task
            task_name = task_list[0].strip()
            # Get number of tasks in the same batch
            mdp_path = f"{task_name}.out.mdp"
            if not os.path.exists(mdp_path):
                print(f"MDP file not found for task {task_name}. Skipping batch.")
                return None
            num_lambdas = 0
            with open(mdp_path, "r", encoding="utf-8") as mdp_file:
                mdp_lines = mdp_file.readlines()
                for line in mdp_lines:
                    if (
                        "=" in line
                        and "lambdas" in line.split("=")[0].strip()
                        and len(line.split("=")[1].strip()) > 0
                    ):
                        _num_lambdas = len(line.split("=")[1].strip().split())
                        if _num_lambdas > num_lambdas:
                            num_lambdas = _num_lambdas
            # Get all tasks in the same batch
            parent_dir = os.path.dirname(os.path.dirname(task_name))
            batch_tasks = []
            for task in task_list:
                if task.startswith(parent_dir + os.sep):
                    batch_tasks.append(task)
            # Check if the number of tasks in the batch matches the number of lambdas
            if len(batch_tasks) < num_lambdas:
                return None
            # Write the remaining tasks back to the queue file
            queue_list_file.seek(0)
            queue_list_file.truncate()
            for task in task_list:
                if not task.startswith(parent_dir + os.sep):
                    queue_list_file.write(task)
    # Return the task directory and the task basename
    target_dirs = [
        os.path.basename(os.path.dirname(task.strip())) for task in batch_tasks
    ]
    return parent_dir, target_dirs, os.path.basename(task_name.strip())


def _reset_task(
    task: tuple[str, str] | tuple[str, list[str], str],
    queue_filename: Path,
    delete_files: bool = False,
):
    """
    Add a task back to the queue. If delete_files is True, all files of the task are
    deleted except for the .tpr, .mdp and log files.
    :param task: The task to reset, as returned by the queue readers.
    :param queue_filename: Path to the queue file.
    :param delete_files: Whether to delete the output files of the task.
    """
    if task is None:
        return
    # Get the list of tasks to add back to the queue
    task_list = []
    if isinstance(task[1], list):
        for t in task[1]:
            task_list.append((os.path.join(task[0], t), task[2]))
    else:
        task_list.append(task)
    # Delete files associated with the task if requested
    if delete_files:
        for directory, basename in task_list:
            task_name = os.path.join(directory, basename)
            if os.path.exists(directory) and not os.path.exists(f"{task_name}.gro"):
                for file in os.listdir(directory):
                    if (
                        basename in file
                        and not file.endswith(".tpr")
                        and not file.endswith(".mdp")
                        and "log" not in file
                    ):
                        os.remove(os.path.join(directory, file))
    # Readd the task(s) to the queue
    task_names = [os.path.join(d, b) for d, b in task_list]
    add_simulation_task_to_file_queue(task_names, queue_filename)


def run_single_simulations(
    queue_filename: Path,
    auto_quit: bool = False,
    extra: str = "",
    max_tasks: Optional[int] = None,
    readd_failed: bool = False,
    stop_file: Optional[str] = None,
):
    """
    Run the queued simulations one after another until the queue is empty.
    :param queue_filename: Path to the queue file.
    :param auto_quit: Quit after five minutes without a task in the queue.
    :param extra: Extra arguments to pass to mdrun.
    :param max_tasks: Maximum number of tasks to run before returning.
    :param readd_failed: Add failed tasks back to the queue.
    :param stop_file: Stop after the current task once this file exists.
    """
    quit_counter = 0
    while max_tasks is None or max_tasks > 0:
        task = None
        try:
            if stop_file and os.path.exists(stop_file):
                print("Found stop file. Exiting...")
                break
            # Get the next task from the queue
            task = get_simulation_task_from_file_queue(queue_filename)
            # If the queue is empty, wait for a while before checking again
            if task is None:
                if auto_quit:
                    quit_counter += 1
                    if quit_counter > 60 * 5:
                        print("Queue is empty. Exiting...")
                        break
                sleep(2)
            # If a task was retrieved, run the simulation
            else:
                dirname, basename = task
                task_name = os.path.join(dirname, basename)
                quit_counter = 0
                time = str(datetime.now()).split(".", maxsplit=1)[0]
                # Write to log file and print message with timestamp
                with open(f"{task_name}.run.log", "a", encoding="utf-8") as log_file:
                    hostname = socket.gethostname()
                    log_file.write(f"Simulation started at {time} on host {hostname}\n")
                print(f"{time} Running task: {task_name}")
                # Setup GROMACS command
                command = (
                    f'gmx mdrun -deffnm "{basename}" -px "{basename}.px.xvg" '
                    + f'-pf "{basename}.pf.xvg" {extra} >> "{basename}.run.log" 2>&1'
                )
                # Run command
                subprocess.run(command, shell=True, check=True, cwd=dirname)
                if not os.path.exists(f"{task_name}.gro"):
                    raise RuntimeError("Task failed: Gro file not found.")
                # Print completion message with timestamp
                time = str(datetime.now()).split(".", maxsplit=1)[0]
                print(f"{time} Task finished: {task_name}")
                if max_tasks is not None:
                    max_tasks -= 1
        except KeyboardInterrupt:
            print("Exiting...")
            # Reset the task by adding it back to the queue
            _reset_task(task, queue_filename)
            break
        except Exception as e:
            print(e)
            if readd_failed and task is not None:
                if Path(os.path.join(*task)).with_suffix(".tpr").exists():
                    print("Re-adding failed task to the queue...")
                    _reset_task(task, queue_filename, delete_files=True)


def run_replica_exchange_simulations(
    queue_filename: Path,
    n_gpus: int,
    gpu_tasks_per_replica: int,
    replex: int = 1000,
    mpi_flags: str = "",
    auto_quit: bool = False,
    extra: str = "",
    max_tasks: Optional[int] = None,
    readd_failed: bool = False,
    stop_file: Optional[str] = None,
):
    """
    Run the queued simulations in batches of replicas that exchange during the run.
    :param queue_filename: Path to the queue file.
    :param n_gpus: Number of GPUs to distribute the replicas over.
    :param gpu_tasks_per_replica: Number of GPU tasks used per replica.
    :param replex: Number of steps between replica exchange attempts.
    :param mpi_flags: Extra flags to pass to mpirun.
    :param auto_quit: Quit after five minutes without a task in the queue.
    :param extra: Extra arguments to pass to mdrun.
    :param max_tasks: Maximum number of batches to run before returning.
    :param readd_failed: Add failed tasks back to the queue.
    :param stop_file: Stop after the current batch once this file exists.
    """
    quit_counter = 0
    while max_tasks is None or max_tasks > 0:
        task = None
        try:
            if stop_file and os.path.exists(stop_file):
                print("Found stop file. Exiting...")
                break
            # Get the next task from the queue
            task = get_simulation_batch_from_file_queue(queue_filename)
            # If the queue is empty, wait for a while before checking again
            if task is None:
                if auto_quit:
                    quit_counter += 1
                    if quit_counter > 60 * 5:
                        print("Queue is empty. Exiting...")
                        break
                sleep(2)
            # If a task was retrieved, run the simulation
            else:
                dirname, target_dirs, basename = task
                quit_counter = 0
                # Print message with timestamp
                time = str(datetime.now()).split(".", maxsplit=1)[0]
                n_replicas = len(target_dirs)
                print(f"{time} Running task: {dirname} with {n_replicas} replicas")
                # Setup GROMACS command
                lambda_dirs = " ".join(sorted(target_dirs))
                gputasks = "".join(
                    str((i * n_gpus) // n_replicas) * gpu_tasks_per_replica
                    for i in range(n_replicas)
                )
                command = (
                    f"mpirun {mpi_flags} -np {n_replicas} gmx_mpi mdrun -replex {replex} "
                    f'-multidir {lambda_dirs} -deffnm "{basename}" '
                    f'-gputasks {gputasks} -pf "{basename}.pf.xvg" -px "{basename}.px.xvg" '
                    f'{extra} >> "{os.path.join(dirname, basename)}.run.log" 2>&1'
                )
                # Run command
                subprocess.run(command, shell=True, check=True, cwd=dirname)
                output_path = f"{os.path.join(dirname, target_dirs[0], basename)}.gro"
                if not os.path.exists(output_path):
                    raise RuntimeError(f"Task failed: File {output_path} not found.")
                # Print completion message with timestamp
                time = str(datetime.now()).split(".", maxsplit=1)[0]
                print(f"{time} Task finished: {dirname} with {n_replicas} replicas")
                if max_tasks is not None:
                    max_tasks -= 1
        except KeyboardInterrupt:
            print("Exiting...")
            # Reset the task by adding it back to the queue
            _reset_task(task, queue_filename)
            break
        except Exception as e:
            print(e)
            if readd_failed and task is not None:
                paths = [os.path.join(task[0], t, task[2]) for t in task[1]]
                if all(Path(p).with_suffix(".tpr").exists() for p in paths):
                    print("Re-adding failed task to the queue...")
                    _reset_task(task, queue_filename, delete_files=True)


def run_simulations_from_queue():
    """Run GROMACS simulations from the simulation queue."""
    # Setup command line arguments
    ap = argparse.ArgumentParser(
        description="Run GROMACS simulations from the simulation queue."
    )
    ap.add_argument(
        "--queue-file",
        "-f",
        type=str,
        default="simulation-queue.dat",
    )
    ap.add_argument(
        "--extra", "-e", type=str, default="", help="Extra arguments to pass to mdrun."
    )
    ap.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Maximum number of tasks to run before exiting.",
    )
    ap.add_argument(
        "--auto-quit",
        "-q",
        action="store_true",
        help="Automatically quit if the queue is empty.",
    )
    ap.add_argument(
        "--readd-failed",
        action="store_true",
        help="Re-add failed tasks to the queue",
    )
    ap.add_argument(
        "--stop-file",
        type=str,
        default=".stop-file",
        help="Path to a file that, if it exists, will stop the simulations after the current task.",
    )
    ap.add_argument(
        "--replex",
        type=int,
        default=None,
        help=(
            "Number of steps between replica exchange attempts "
            "(don't specify to disable replica exchange)."
        ),
    )
    ap.add_argument(
        "--n-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for replica exchange simulations.",
    )
    ap.add_argument(
        "--gpu-tasks-per-replica",
        type=int,
        default=1,
        help="Number of GPU tasks to use per replica for replica exchange simulations.",
    )
    ap.add_argument(
        "--mpi-flags",
        type=str,
        default="",
        help="Extra MPI flags to pass to mpirun for replica exchange simulations.",
    )
    # Parse command line arguments
    args = ap.parse_args()
    queue_filename = Path(args.queue_file).expanduser().resolve()
    auto_quit = args.auto_quit
    max_tasks = args.max_tasks if args.max_tasks and args.max_tasks > 0 else None
    stop_file = Path(args.stop_file).expanduser().resolve() if args.stop_file else None
    # Run simulations
    if args.replex is not None and args.replex > 0:
        run_replica_exchange_simulations(
            queue_filename=queue_filename,
            n_gpus=args.n_gpus,
            gpu_tasks_per_replica=args.gpu_tasks_per_replica,
            replex=args.replex,
            mpi_flags=args.mpi_flags,
            auto_quit=auto_quit,
            extra=args.extra,
            max_tasks=max_tasks,
            readd_failed=args.readd_failed,
            stop_file=stop_file,
        )
    else:
        run_single_simulations(
            queue_filename=queue_filename,
            extra=args.extra,
            max_tasks=max_tasks,
            auto_quit=auto_quit,
            readd_failed=args.readd_failed,
            stop_file=stop_file,
        )
