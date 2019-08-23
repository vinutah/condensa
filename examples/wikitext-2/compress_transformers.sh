#!/bin/bash
#SBATCH --job-name=condensa
#SBATCH --output=condensa-ngc-%j.out-%N  # %j is job id and %N is node 

#SBATCH --account=notchpeak-gpu 
#SBATCH --partition=notchpeak-gpu

#SBATCH --nodes=1  # usually always keep this at 1
#SBATCH --gres=gpu:v100:1   # allocate 1 GPU for this job (maximum 3 for this machine)

#SBATCH --time=20:00:00  # maximum time to run for

#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --mail-user=vinutah.soc@gmail.com

#nvidia-smi
python -c "import time; print(time.strftime('%Y-%m-%d %H:%M'))"  # print time so I know when it started

# Setup python environment
module load python/3.6.3
module load cuda/10.1 cudnn/7.6.0
python3 -m venv $HOME/rockbox/condensa/venv/
module unload python/3.6.3
source $HOME/rockbox/condensa/activate
pip install --upgrade pip
pip3 install torch
pip3 install torchvision
pip3 install jupyterlab
pip3 install pylint
pip3 install six bcolz scipy
pip3 install opencv-python pandas seaborn
#pip3 install graph_viz
pip3 install isoweek pandas_summary
pip3 install ipywidgets tqdm torchtext sklearn-pandas

#cd $HOME/condensa/examples/wikitext-2

# have your condensa-job here
#SCHEME=${1}
#DENSITY=${2}
#STEPS=${3}
#
#PREFIX=transformer_${SCHEME}_${DENSITY//[\.]/_}
#
#python compress.py\
#       --arch Transformer\
#       --model trained/transformer.pth\
#       --dataset ./data/wikitext-2\
#       --steps ${STEPS}\
#       --lr 5 --lr_decay 1e-4\
#       --weight_decay 0\
#       --momentum 0.95\
#       --mb_iterations_per_l 3000\
#       --mb_iterations_first_l 30000\
#       --mu_init 1e-3 --mu_multiplier 1.1\
#       --l_batch_size 128\
#       --scheme ${SCHEME}\
#       --density ${DENSITY}\
#       --out compressed/${PREFIX}.pth\
#       --csv results/${PREFIX}.csv\
#       --cuda \
#       -v

# print time so I know when it finished
python -c "import time; print(time.strftime('%Y-%m-%d %H:%M'))"
