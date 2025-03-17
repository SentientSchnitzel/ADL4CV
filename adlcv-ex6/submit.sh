#!/bin/sh 

### -- set the job Name --
#BSUB -J ex6
### -- specify files --
#BSUB -o /work3/s194262/GitHub/ADL4CV/adlcv-ex6/hpc_logs/ex6-%J.out
#BSUB -e /work3/s194262/GitHub/ADL4CV/adlcv-ex6/hpc_logs/ex6-%J.err

### General options
### –- specify queue --
# possible: gpuv100, gpua100, gpua10, gpua40
#BSUB -q gpua100
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 00:30
### -- request _ GB of system-memory --
#BSUB -R "rusage[mem=4GB]"
#BSUB -R "span[hosts=1]"


nvidia-smi
module load cuda/11.8

source /work3/s194262/adv_dl_cv/bin/activate

cd /work3/s194262/GitHub/ADL4CV/adlcv-ex6


# default train
python main.py --config
