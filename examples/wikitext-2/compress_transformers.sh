#!/bin/bash
#SBATCH --job-name=condensa
#SBATCH --output=condensa-ngc-%j.out-%N  # %j is job id and %N is node 

#SBATCH --account=notchpeak-gpu 
#SBATCH --partition=notchpeak-gpu

#SBATCH --nodes=1  # usually always keep this at 1
#SBATCH --gres=gpu:v100:1   # allocate 1 GPU for this job (maximum 3 for this machine)

#SBATCH --time=2:00:00  # maximum time to run for

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=vinutah.soc@gmail.com

nvidia-smi
python -c "import time; print(time.strftime('%Y-%m-%d %H:%M'))"  # print time so I know when it started

# Setup python environment
module load python/3.6.3
module load cuda/10.1 cudnn/7.6.0
python3 -m venv $HOME/condensa/venv/
module unload python/3.6.3
source bin/activate
pip install --upgrade pip
pip3 install torch
pip3 install torchvision
pip3 install jupyterlab
pip3 install six bcolz scipy
pip3 install opencv-python pandas seaborn
#pip3 install graph_viz
pip3 install isoweek pandas_summary
pip3 install ipywidgets tqdm torchtext sklearn-pandas

# have your condensa-job here

# print time so I know when it finished
python -c "import time; print(time.strftime('%Y-%m-%d %H:%M'))"
