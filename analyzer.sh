#!/bin/bash
### sbatch config parameters must start with #SBATCH and must precede any other commands. to ignore just add another # - like ##SBATCH
#SBATCH --partition main                         ### specify partition name where to run a job. debug: 2 hours; short: 7 days
#SBATCH --time 5-01:00:00                      ### limit the time of job running, partition limit can override this. Format: D-H:MM:SS
#SBATCH --output my_job-id-%J.out                ### output log for running job - %J for job number
##SBATCH --mail-user=gilba@post.bgu.ac.il      ### users email for sending job status
##SBATCH --mail-type=BEGIN,END,FAIL             ### conditions when to send the email. ALL,BEGIN,END,FAIL, REQUEU, NONE
#SBATCH  --gpus=1 ###rtx_3090:1                ### number of GPUs (can't exceed 8 gpus for now) ask for more than 1 only if you can parallelize your code for multi GPU
##SBATCH --exclude=dt-gpu-03,dt-gpu-04,dt-1080-01
#SBATCH --mem=32G
#SBATCH --tmp=200G             ### Asks to allocate enough space on /scratch
#SBATCH --cpus-per-task=6 # 6 cpus per task – use for multithreading, usually with --tasks=1
#SBATCH --tasks=1 # 4 processes – use for multiprocessing
### Start you code below ####
module load anaconda              ### load anaconda module
source activate cell_generator_new         ### activating environment, environment must be configured before running the job
###source activate fnet_v1         ### activating environment, environment must be configured before running the job
# export STORAGE_DIR=/storage/users/assafzar/single_cells_fovs
# export SLURM_SCRATCH_DIR=/scratch/lionb@auth.ad.bgu.ac.il/${SLURM_JOB_ID}/single_cells_fovs
# mkdir $SLURM_SCRATCH_DIR
# cp -r $STORAGE_DIR/* $SLURM_SCRATCH_DIR
# ###PYTHONPATH=`pwd` python  predict.py  --experiment_name $2  --save_base_dir $3
# python create_metadata.py
python -u /sise/home/lionb/cell_generator/mg_analyzer.py
##python /sise/home/lionb/cell_generator/test.py ### execute jupyter lab command – replace with your own command e.g. ‘srun --mem=24G python my.py my_arg’ . you may use multiple srun lines, they are the job steps. --mem - the memory to allocate: use 24G x number of allocated GPUs
### --mem=24G