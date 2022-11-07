#!/bin/bash --login

#SBATCH --job-name=cas_test
#SBATCH --time=0-00:20:00 ##0-00:20:00 # 2-00:00:00 1-00:00:00
#SBATCH --nodes=4
#SBATCH --tasks-per-node=64 ##128
#SBATCH --cpus-per-task=2 ##1
#SBATCH --account=e89-sur
#SBATCH --partition=standard
#SBATCH --qos=short ## short long standard
#SBATCH --reservation=shortqos
#SBATCH -o slurm.%N.%j.out 
#SBATCH -e slurm.%N.%j.err
#SBATCH --exclusive

cd $SLURM_SUBMIT_DIR
echo $SLURM_NODELIST

# Load the Python module
module load cray-python/3.9.7.1 # cray-python/3.8.5.0

# Load castep
module load castep/21.11 #/20.11 


export WORK=/mnt/lustre/a2fs-work3/work/e89/e89/louie/
# export WORK=/mnt/lustre/a2fs-work1/work/e280/e280-Sacchi/louie280/
# export WORK=/mnt/lustre/a2fs-work2/work/e05/e05/louiemcc/

export PYTHONUSERBASE=$WORK/.local
export PATH=$PYTHONUSERBASE/bin:$PATH
export PYTHONPATH=$PYTHONUSERBASE/lib/python3.9/site-packages:$PYTHONPATH
export MPLCONFIGDIR=$WORK/.config/matplotlib

export OMP_NUM_THREADS=2
export OMP_PLACES=cores

export CASTEP_PP_PATH="/mnt/lustre/a2fs-work3/work/e89/e89/louie/ps_pots/"
export CASTEP_COMMAND="srun --distribution=block:block --hint=nomultithread castep.mpi"

echo "Starting calculation at $(date)"
python3 graphene_project_v04.py >> py_ouput.out
echo "Calculation ended at $(date)"
