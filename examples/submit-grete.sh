#!/bin/bash
# Queue worker for the GWDG Grete GPU partitions, processing the lambda windows queued by
# 'martini-abfe run' with replica exchange.
#
# Full node on grete (4x A100, 64 cores, billed as a whole node):
#   sbatch submit-grete.sh
# Part of a node on grete:shared (16 cores and ~120 GiB per GPU):
#   sbatch -p grete:shared -G A100:2 -c 32 --mem=240G submit-grete.sh
#
# Submit from the directory of the queue file, or set QUEUE_FILE. The queue holds absolute
# paths, so 'martini-abfe run' must have been started on a file system the nodes can see.
#
#SBATCH --job-name=martini-abfe
#SBATCH --partition=grete
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH -G A100:4
#SBATCH --mem=480G
#SBATCH --time=48:00:00
#SBATCH --output=martini-abfe-%j.log
# Send SIGINT 15 minutes before the time limit, so the worker returns the running leg to
# the queue before the job ends
#SBATCH --signal=B:INT@900
##SBATCH -A <project>

set -euo pipefail

QUEUE_FILE=${QUEUE_FILE:-simulation-queue.dat}
# Largest number of lambda windows of a leg, i.e. MPI ranks of one replica exchange run
MAX_REPLICAS=${MAX_REPLICAS:-23}

# Environment providing gmx_mpi (GROMACS built with MPI and CUDA) and martini-abfe
# module load <gromacs module>
# source <conda>/bin/activate martini-abfe

# One OpenMP thread per rank and core, so that the largest leg fits on the allocated cores
N_GPUS=${SLURM_GPUS_ON_NODE:-1}
NTOMP=$(( SLURM_CPUS_PER_TASK / MAX_REPLICAS ))
NTOMP=$(( NTOMP > 0 ? NTOMP : 1 ))

echo "Host $(hostname), ${SLURM_CPUS_PER_TASK} cores, ${N_GPUS} GPUs, -ntomp ${NTOMP}"

# Replicas are spawned by mpirun inside the single Slurm task, so allow more ranks than
# tasks and leave the thread placement to mdrun (-pin on)
exec martini-abfe-queue \
    -f "${QUEUE_FILE}" \
    --replex 1000 \
    --n-gpus "${N_GPUS}" \
    --mpi-flags "--oversubscribe --bind-to none" \
    --extra "-ntomp ${NTOMP}" \
    --auto-quit
