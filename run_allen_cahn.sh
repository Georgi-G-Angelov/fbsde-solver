#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL # required to send email notifcations
#SBATCH --mail-user=gga19 # required to send email notifcations - please replace <your_username> with your college login name or email address
export PATH=/vol/bitbucket/gga19/myvenv/bin/:$PATH
source /vol/cuda/11.0.3-cudnn8.0.5.39/setup.sh
source activate
python3 AllenCahn20D.py
