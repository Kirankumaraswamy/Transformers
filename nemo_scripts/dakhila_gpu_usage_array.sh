#!/bin/bash
#MOAB -N nemo_gpu_test
#MOAB -t 1-12 # specifies array job indices
#MOAB -q gpu
#MOAB -l nodes=1:ppn=4:gpus=1
#MOAB -l walltime=0:12:00:00
#MOAB -l pmem=8000mb
#MOAB -d /work/ws/nemo/fr_ds565-transformer_work-0/lab_transformer/
#MOAB -o /work/ws/nemo/fr_ds565-transformer_work-0/lab_transformer/out_std_${MOAB_JOBID}.out
#MOAB -e /work/ws/nemo/fr_ds565-transformer_work-0/lab_transformer/error_std_${MOAB_JOBID}.err

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
STEPS=$((MOAB_JOBARRAYINDEX))
echo "STEPS: $STEPS"

${HOME}/anaconda3/envs/pytorch_transformer/bin/python  /work/ws/nemo/fr_ds565-transformer_work-0/lab_transformer/training_script.py --use_config_load True --array_number $STEPS --name dakhila --layer_scheme enc-dec
# later run: --layer_scheme ff-att

# Print some Information about the end-time to STDOUT
echo "DONE";
echo "Finished at $(date)";