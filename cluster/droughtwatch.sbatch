#!/bin/bash
#SBATCH --qos=unkillable 
#SBATCH --job-name=mapping
#SBATCH --cpus-per-task= 4
#SBATCH --gres=gpu:1
#SBATCH --account=def-bengioy
#SBATCH --mem= 16G
#SBATCH --time=10:00:00
#SBATCH --output /home/bhowmicd/logs/segmentation/-%j.out
#SBATCH --error home/bhowmicd/logs/segmentation/-%j.err
#SBATCH --mail-user=debjani.ism@gmail.com


# 1.load singularity
module load singularity/3.4
# 1.load singularity
# 2. Copy your dataset on the compute node
rsync -avz /home/bhowmicd/projects/rpp-bengioy/droughtwatch/data/$SLURM_TMPDIR
rsync -avz /home/bhowmicd/scratch/images/droughtwatch.img $SLURM_TMPDIR

# 3. Executing your code with singularity
singularity exec --nv --writable-tmpfs \
 -H $HOME:/dataset \
 -B $HOME/codes:/home/codes
 -B $SLURM_TMPDIR:/dataset/ \
 -B $SLURM_TMPDIR:/tmp_log/ \
 -B $HOME/logs:/final_log/ \
 -B $SCRATCH/:/$SCRATCH/ \
 /home/bhowmicd/scratch/images/droughwatch.img \
 /home/bhowmicd/codes/droughtwatch/cluster/droughtwatch.sh

# 4. Copy whatever you want to save on $SCRATCH
rsync -avz $SLURM_TMPDIR/final_log/ $SCRATCH
rsync -avz $SLURM_TMPDIR/temp_log/ $SCRATCH

