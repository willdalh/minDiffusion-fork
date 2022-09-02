#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:V100:1
#SBATCH --account=ie-idi
#SBATCH --time=38:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=12000
#SBATCH --job-name="sampling mindiffusion"
#SBATCH --output=sampling_output.out
#SBATCH --mail-user=williad@stud.ntnu.no
#SBATCH --mail-type=ALL

WORKDIR=${SLURM_SUBMIT_DIR}
cd ${WORKDIR}
echo "we are running from this directory: $SLURM_SUBMIT_DIR"
echo " the name of the job is: $SLURM_JOB_NAME"
echo "Th job ID is $SLURM_JOB_ID"
echo "The job was run on these nodes: $SLURM_JOB_NODELIST"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "We are using $SLURM_CPUS_ON_NODE cores"
echo "We are using $SLURM_CPUS_ON_NODE cores per node"
echo "Total of $SLURM_NTASKS cores"
echo "Starting script"
echo ""
module purge
module load Anaconda3/2020.07
python sampler.py
uname -a
