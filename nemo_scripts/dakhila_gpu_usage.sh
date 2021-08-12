#!/bin/bash
#MSUB -N nemo_gpu_test
#MSUB -q gpu
#MSUB -l nodes=1:ppn=4:gpus=1
#MSUB -l walltime=0:12:00:00
#MSUB -l pmem=8000mb
#MSUB -d /work/ws/nemo/fr_ds565-transformer_work-0/lab_transformer/
#MSUB -o /work/ws/nemo/fr_ds565-transformer_work-0/lab_transformer/out_std_${MOAB_JOBID}.out
#MSUB -e /work/ws/nemo/fr_ds565-transformer_work-0/lab_transformer/error_std_${MOAB_JOBID}.err

source ${HOME}/.bashrc

conda activate pytorch_transformer

echo 'env:' $CONDA_DEFAULT_ENV
echo 'env:' $CONDA_PREFIX
echo 'pythonpath:' $PYTHONPATH
echo "path: $PATH"

echo 'which python:' $(which python)

# Print some information about the job to STDOUT
echo "Workingdir: $PWD";
echo "Started at $(date)";

# ================================================== #
# Begin actual Code

${HOME}/anaconda3/envs/pytorch_transformer/bin/python  /work/ws/nemo/fr_ds565-transformer_work-0/lab_transformer/training_script.py --num_of_optimizers 2 --pre_model_file_name experiments_save/20210702090823.954315/models/transformer_ckpt_epoch_10.pth --pre_event_file experiments_save/runs/20210702090823.954315/events.out.tfevents.1625209716.gpuv100.nemo.privat.3928.0

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";