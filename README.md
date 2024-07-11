# Investigating Object Insertion into a Video Using Diffusion model

This repository contains technical explanation realting to investigation of using tuning method based on stable diffusion model in order to insert object into a human-object interaction video. 

first is a detail explanation on how to set the environmet for Tune-A-Video. 
you can use the original github or the forked one in my repo. 
```
git clone 
```

then create an environment, i used same python version as the authors mentioned. and then install the packages on the requirments file 
```
conda create -n tune-a-video python=3.10.6
conda activate tune-a-video
pip install -r requirements.txt
```

some of the requirements:
```
torch==1.12.1
torchvision==0.13.1
diffusers[torch]==0.11.1
transformers>=4.25.1
...
```

### shoter explanation about CUDA and PyTorch Compability:
CUDA is a general parallel computation architecture developed to NVIDIA's GPUs.  using CUDA, we can improve the preformance of training PyTorch models. 
the PyTorch code runs in parallel over the GPUs cores using CUDA. 
PyToech uses the package torch.cuda to execute CUDA operations. 

cuda and torch 1.12.1 compabilitym here the cuda is the nvcc one (higher level):
[https://pytorch.org/get-started/previous-versions/#v1121](https://pytorch.org/get-started/previous-versions/#v1121)
```
# CUDA 10.2
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
# CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```

CUDA has the runtime API and the driver API.
the runtime API installed by the CUDA toolkit installer and it's version can be obtained using ```nvcc --version```. 
nvcc stands for NVIDIA CUDA Complier. This command doesn't know anything about if and what GPU driver is installed. 
we get information about the driver using  ```nvidia-smi```, stands for NVIDIA System Management Interface. it gives us information about the driver version, GPU name, power usage and the processes that currently using the GPU. it also shows you the CUDA version that your driver supports.
please pay attention to the difference CUDA versions you might get when running ```nvcc --version``` or ```nvidia-smi```.
in my case ```nvcc --version``` is ```cuda 11.2``` and my ```nvidia-smi``` is ```cuda 12.4```. 
the nvidia-smi suppose to be equal or greater then the version reported in nvcc.

### shoter explanation about xformers:
we have to install xformers to use tune-a-video, but xformers works with pytorch 2.3.0 and tune-a-video needs torch 1.12.1.
so we need to install xformers from source. 
first install ninja, it makes the build much faster
```
pip install ninja
```
and then install xformers (might take few minutes):
```
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```
for more valuable information regarding installing xformers with cuda12 please take a look at these issues:
[https://github.com/facebookresearch/xformers/issues/842](https://github.com/facebookresearch/xformers/issues/842)
[https://github.com/facebookresearch/xformers/issues/960](https://github.com/facebookresearch/xformers/issues/960)

