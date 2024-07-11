# Investigating Object Insertion into a Video Using Diffusion model

This repository contains technical explanation realting to investigation of using tuning method based on stable diffusion model in order to insert object into a human-object interaction video. 
I struggeled a lot with the first installtion for the Tune-A-Video, and i would wish i had these kind of clearer guidelines, so i wrote what i learn with a lot of details and explanations to help with who ever wants to recover my results or use Tune-A-Video for other goals.
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
since this code uses xformers i couldn't download pytorch 1.12.1 for compability conflicts and i solved it by running:
```
pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
CUDA has the runtime API and the driver API.
the runtime API installed by the CUDA toolkit installer and it's version can be obtained using ```nvcc --version```. 
nvcc stands for NVIDIA CUDA Complier. This command doesn't know anything about if and what GPU driver is installed. 
we get information about the driver using  ```nvidia-smi```, stands for NVIDIA System Management Interface. it gives us information about the driver version, GPU name, power usage and the processes that currently using the GPU. it also shows you the CUDA version that your driver supports.
please pay attention to the difference CUDA versions you might get when running ```nvcc --version``` or ```nvidia-smi```.
in my case ```nvcc --version``` is ```cuda 11.2``` and my ```nvidia-smi``` is ```cuda 12.4```. 
the nvidia-smi suppose to be equal or greater then the version reported in nvcc.
This command should return True if CUDA is correctly set up with PyTorch.
```
python -c "import torch; print(torch.cuda.is_available())"
```

### shoter explanation about xformers:
xformers allows efficient running, with costomizable bulding blocks that not available on PyTorch, it contains it own CUDA kernels.
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
for me it didn't worked so i downloaded like this: 
```
pip install --root-user-action=ignore xformers==0.0.16
```
verify installation
```
python -c "import xformers; print(xformers.__version__)"
```
for more valuable information regarding installing xformers with cuda12 please take a look at these issues:
[https://github.com/facebookresearch/xformers/issues/842](https://github.com/facebookresearch/xformers/issues/842)
[https://github.com/facebookresearch/xformers/issues/960](https://github.com/facebookresearch/xformers/issues/960)

### shoter explanation about pretrained model installation:
tune-a-video uses text2image pretrained model, you can download the stable diffusion models from Hugging Face, 
Stable Diffusion v1 is a specific configuration of the model architecture that uses a downsampling-factor 8 autoencoder with an 860M UNet and CLIP ViT-L/14 text encoder for the diffusion model. The model was pretrained on 256x256 images and then finetuned on 512x512 images.
I used Stable Diffusion v1-4. it was initialized with the weights of Stable Diffusion v1-2 and was fine-tuned.

these .ckpt files might be too heavy for simple cloning, so we first need to install Git LFS. 
git LFS, stand for Large File Storage, helps to handle files larger then 10MB. if you dont have it please download from [https://git-lfs.com/](https://git-lfs.com/),
once downloaded install: 
```
git lfs install
```
at first i had issues to install it because of problems with running sudo, since i probably didnt have the super user elevated privileges, you can try 
Installing a new package: `sudo apt-get install package_name`
Updating the system: `sudo apt-get update && sudo apt-get upgrade`
at the end i solved it by running these commands: 
```
apt-get update
apt-get install curl
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs
git lfs install
```

after you have git LFS you can download the Stable Diffusion checkpoints by running: 
```
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4 checkpoints/CompVis/stable-diffusion-v1-4
```
i ran instead
```
cp -r ./checkpoints/CompVis/stable-diffusion-v1-4/ ./checkpoints/stable-diffusion-v1-4/ 
```
please pay attention, without git LFS this command won't raise a failure but the checkpoint may not been downloaded and it can raise errors later so don't skip it. 

### natsort 
this is needed for the inference
```
pip install natsort 8.4.0
```

### Training Tune-A-Video
finally, after succesful installation we can fine-tune the text-to-image diffusion models for text-to-video generation, run this command,
you can change the config file to your own dataset.
the command ```accelerate launch [arguments] {training_script}``` launches a specified script on a distributed system with the right parameters. 
```{training_script}```, in our case is the train_tuneavideo.py, is the script to be launched in parallel,  ```--confing_file CONFIG_FILE``` contains the default values in the launching script. more details [here](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch). 
```
accelerate launch train_tuneavideo.py --config="configs/man-skiing.yaml"
```

### important notes: 
n_sample_frames is the number of frames used for training the model
video_length is the number of frames used for inference (i.e., generating new videos)
please note that if you have 24 frames for traning, you need to write length 12 for the inference.

### conda list 
if you have any issues take a look at the packages list in my working environment at [https://github.com/shani1610/object-insertion-video-diffusion/blob/main/conda_list.md](https://github.com/shani1610/object-insertion-video-diffusion/blob/main/conda_list.md) 
