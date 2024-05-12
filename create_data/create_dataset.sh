#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other command. to ignore just add another # - like so ##SBATCH
#SBATCH --partition main ### specify partition name where to run a job. main - 7 days time limit
#SBATCH --time 2-01:30:00 ### limit the time of job running. Make sure it is not greater than the partition time limit!! Format: D-H:MM:SS
#SBATCH --job-name create_dataset ### name of the job. replace my_job with your desired job name
#SBATCH --output my_job-id-%J.out ### output log for running job - %J is the job number variable
#SBATCH --mail-user=user@post.bgu.ac.il ### users email for sending job status notifications
#SBATCH --mail-type=BEGIN,END,FAIL ### conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH --cpus-per-task=4 # 6 cpus per task � use for multithreading, usually with --tasks=1
#SBATCH --tasks=1 # 4 processes � use for multiprocessing
#SBATCH --tmp=200G             ### Asks to allocate enough space on /scratch
#SBATCH --mem=24G



module load anaconda              ### load anaconda module
source activate cell_generator_new         ### activating environment, environment must be configured before running the job

export STORAGE_DIR=/storage/users/assafzar/full_cells_fovs
export SLURM_SCRATCH_DIR=/scratch/lionb@auth.ad.bgu.ac.il/${SLURM_JOB_ID}/full_cells_fovs

mkdir $SLURM_SCRATCH_DIR
##cp -r $STORAGE_DIR/* $SLURM_SCRATCH_DIR

python -u segment_and_create_pertrub_dataset.py
# python create_metadata.py

##\cp -r $SLURM_SCRATCH_DIR/* $STORAGE_DIR


