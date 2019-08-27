#!/bin/bash

nvidia-smi
python -c "import time; print(time.strftime('%Y-%m-%d %H:%M'))"  # print time so I know when it started

SCHEME=${1}
DENSITY=${2}
STEPS=${3}

PREFIX=transformer_${SCHEME}_${DENSITY//[\.]/_}

python compress.py\
       --arch Transformer\
       --model trained/transformer.pth\
       --dataset ./data/wikitext-2\
       --steps ${STEPS}\
       --lr 5 \
       --lr_decay 1e-4\
       --weight_decay 0\
       --momentum 0.95\
       --mb_iterations_per_l 3000\
       --mb_iterations_first_l 30000\
       --mu_init 1e-3 --mu_multiplier 1.1\
       --l_batch_size 128\
       --eval_batch_size 10\
       --scheme ${SCHEME}\
       --density ${DENSITY}\
       --out compressed/${PREFIX}.pth\
       --csv results/${PREFIX}.csv\
       --cuda \
       -v

# print time so I know when it finished
python -c "import time; print(time.strftime('%Y-%m-%d %H:%M'))"
