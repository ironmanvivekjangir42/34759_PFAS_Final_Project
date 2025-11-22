#!/bin/sh 

### LSF Directives
# ---------------------------------------------------------------------------------------

### -- Specify Queue -- 
# Switch to the dedicated GPU V100 queue
#BSUB -q gpuv100 

### -- Set the Job Name -- 
#BSUB -J YOLO_Transfer_Train

### -- ask for number of CPU cores -- 
#BSUB -n 16

### -- specify that all cores must be on the same host -- 
#BSUB -R "span[hosts=1]"

### -- specify memory per core/slot -- 
#BSUB -R "rusage[mem=4GB]"

### -- set walltime limit: hh:mm -- 
#BSUB -W 12:00

###  CORRECTED: Specify GPU requirements using -gpu flag 
# Request 1 GPU in exclusive process mode (recommended practice on HPC)
#BSUB -gpu "num=1:mode=exclusive_process" 

### -- Specify the output and error files (%J is the job-id) -- 
#BSUB -oo transfer_Output_%J.out 
#BSUB -eo transfer_Output_%J.err 

# ---------------------------------------------------------------------------------------
### Execution Commands
# ---------------------------------------------------------------------------------------

# 1. Activate your Conda/Python environment
source /zhome/63/6/222806/DLforCV/bin/activate


# 2. Run the Python training script
# The script itself must be configured for Transfer Learning (e.g., MODEL_WEIGHTS = 'yolov8s.pt')
python train_yolov11.py