# DS-Healthcare-Final-Project
Course Project for DS-GA 3001 Data Science for Healthcare

## Installation 

1. Python Environment: version 3.9
```
conda create -n venv python=3.9
```

2. Packages Needed
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y
conda install pyg -c pyg -y
pip install torch-scatter==2.1.1 -f https://data.pyg.org/whl/torch-2.0.0%2Bcu117.html
pip install torch-sparse==0.6.17 -f https://data.pyg.org/whl/torch-2.0.0%2Bcu117.html 
conda install anndata -c conda-forge -y
conda install pandas -y
pip install python-louvain 
conda install -c conda-forge scanpy -y
conda install -c anaconda scikit-image -y
conda install -c conda-forge igraph -y
conda install -c conda-forge louvain -y
```

## Run 
An example of running the model with dataset (MouseLymphCrossInfection)
```
python run.py --dataset MouseLymphCrossInfection --num-heads 14 --num-seed-class 3 --wd 0 --lr 1e-3 --batch-size 32 --epoch 320 --distance_thres 30 --cnn_type gat
```
This example is set to run on GATV2. In case for running on original STELLAR model, remove the cnn_type argument from the command.


## Singularity Setting (NYU GREENE HPC)

1. Our Setup of CUDA image which ran successfully
```
cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif
```
2. Python 3.9
```
wget Miniconda3-py39_23.3.1-0-Linux-x86_64.sh
```
3. Singularity
```
singularity exec --overlay /scratch/yj2369/senv/overlay-15GB-500K.ext3:rw /scratch/work/public/singularity/cuda11.7.99-cudnn8.5-devel-ubuntu22.04.2.sif /bin/bash
```




