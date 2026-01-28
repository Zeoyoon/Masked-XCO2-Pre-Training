#!/bin/bash

# Pre-training tasks with multi-masking strategy across different seeds
python train.py --mode 'Pre-train' --seed 0 --pt 0 --lr 0.01 --bs 32
python train.py --mode 'Pre-train' --seed 1 --pt 0 --lr 0.01 --bs 32
python train.py --mode 'Pre-train' --seed 2 --pt 0 --lr 0.01 --bs 32
python train.py --mode 'Pre-train' --seed 3 --pt 0 --lr 0.01 --bs 32
python train.py --mode 'Pre-train' --seed 4 --pt 0 --lr 0.01 --bs 32

# Downstream Plume Regression tasks with pre-trained model across different seeds
python train.py --mode 'PlumeCLS' --seed 0 --pt 1 --lr 0.01 --bs 32
python train.py --mode 'PlumeCLS' --seed 1 --pt 1 --lr 0.01 --bs 32
python train.py --mode 'PlumeCLS' --seed 2 --pt 1 --lr 0.01 --bs 32
python train.py --mode 'PlumeCLS' --seed 3 --pt 1 --lr 0.01 --bs 32
python train.py --mode 'PlumeCLS' --seed 4 --pt 1 --lr 0.01 --bs 32
