# Investigating Object Insertion into a Video Using Diffusion model

This repository contains technical explanations relating to the investigation of using tuning methods based on the stable diffusion model to insert objects into human-object interaction videos.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Installation
I struggled a lot with the first installation for Tune-A-Video, and I wish I had these kinds of clearer guidelines, so I wrote what I learned with a lot of details and explanations to help whoever wants to reproduce my results or use Tune-A-Video for other goals.

First is a detailed explanation of how to set up the environment for Tune-A-Video. You can use the original GitHub or the forked one in my repo.

```
git clone <repository-url>
```

Then create an environment. I used the same Python version as the authors mentioned and then installed the packages in the requirements file:

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

### CUDA and PyTorch Compatibility
CUDA is a general parallel computation architecture developed by NVIDIA for its GPUs. Using CUDA, we can improve the performance of training PyTorch models. The PyTorch code runs in parallel over the GPU cores using CUDA. PyTorch uses the torch.cuda package to execute CUDA operations.

For CUDA and PyTorch 1.12.1 compatibility, refer to the
[PyTorch documentation](https://pytorch.org/get-started/previous-versions/#v1121)
```
# CUDA 10.2
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
# CUDA 11.3
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
# CUDA 11.6
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.6 -c pytorch -c conda-forge
```
Since this code uses xformers, I couldn't download PyTorch 1.12.1 due to compatibility conflicts. I solved it by running:

```
pip install torch==1.13.1 torchvision==0.14.1 --extra-index-url https://download.pytorch.org/whl/cu117
```
CUDA has the runtime API and the driver API. The runtime API is installed by the CUDA toolkit installer, and its version can be obtained using ```nvcc --version```. 
nvcc stands for NVIDIA CUDA Complier. command stands for NVIDIA CUDA Compiler. This command doesn't know anything about whether a GPU driver is installed. We get information about the driver using  ```nvidia-smi```, which stands for NVIDIA System Management Interface. It gives us information about the driver version, GPU name, power usage, and the processes currently using the GPU. It also shows the CUDA version that your driver supports.
Please pay attention to the different CUDA versions you might get when running ```nvcc --version``` or ```nvidia-smi```.
In my case ```nvcc --version``` is ```cuda 11.2``` and my ```nvidia-smi``` is ```cuda 12.4```. 
the ```nvidia-smi``` version should be equal to or greater than the version reported by ```nvcc --version```.
This command should return True if CUDA is correctly set up with PyTorch.
```
python -c "import torch; print(torch.cuda.is_available())"
```

### xformers
xformers allows efficient running with customizable building blocks that are not available in PyTorch. It contains its own CUDA kernels. We have to install xformers to use Tune-A-Video, but xformers works with PyTorch 2.3.0, and Tune-A-Video needs torch 1.12.1. So we need to install xformers from the source.

First, install ninja; it makes the build much faster:
```
pip install ninja
```
Then install xformers (this might take a few minutes):

```
pip install -v -U git+https://github.com/facebookresearch/xformers.git@main#egg=xformers
```
If this doesn't work, try:

```
pip install --root-user-action=ignore xformers==0.0.16
```
Verify the installation:

```
python -c "import xformers; print(xformers.__version__)"
```
For more information regarding installing xformers with CUDA 12, please refer to these issues:

[Issue 842](https://github.com/facebookresearch/xformers/issues/842)

[Issue 960](https://github.com/facebookresearch/xformers/issues/960)

### Pretrained Model Installation

Tune-A-Video uses text-to-image pretrained models. You can download the Stable Diffusion models from Hugging Face. Stable Diffusion v1 is a specific configuration of the model architecture that uses a downsampling-factor 8 autoencoder with an 860M UNet and CLIP ViT-L/14 text encoder for the diffusion model. The model was pretrained on 256x256 images and then fine-tuned on 512x512 images. I used Stable Diffusion v1-4. It was initialized with the weights of Stable Diffusion v1-2 and was fine-tuned.

These .ckpt files might be too large for simple cloning, so we first need to install Git LFS. Git LFS stands for Large File Storage, which helps to handle files larger than 10MB. If you don't have it, please download it from [git-lfs.com](https://git-lfs.com/),
Once downloaded, install it:
 ```
git lfs install
```
Initially, I had issues installing it due to problems with running sudo, as I probably didn't have the superuser elevated privileges. You can try:
```
sudo apt-get install package_name
sudo apt-get update && sudo apt-get upgrade
```
Eventually, I solved it by running these commands:
```
apt-get update
apt-get install curl
curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | bash
apt-get install git-lfs
git lfs install
```

After you have Git LFS, you can download the Stable Diffusion checkpoints by running:

```
git clone https://huggingface.co/CompVis/stable-diffusion-v1-4 checkpoints/CompVis/stable-diffusion-v1-4
```
I ran instead:

```
cp -r ./checkpoints/CompVis/stable-diffusion-v1-4/ ./checkpoints/stable-diffusion-v1-4/ 
```
Please note, without Git LFS, this command won't raise a failure, but the checkpoint may not be downloaded, and it can raise errors later, so don't skip it.


### natsort 
This is needed for the inference:

```
pip install natsort 8.4.0
```

### Training Tune-A-Video
Finally, after a successful installation, we can fine-tune the text-to-image diffusion models for text-to-video generation. Run this command, and you can change the config file to your dataset. The command  ```accelerate launch [arguments] {training_script}``` launches a specified script on a distributed system with the right parameters. 
```{training_script}```, in our case is the train_tuneavideo.py, is the script to be launched in parallel,  ```--confing_file CONFIG_FILE``` contains the default values in the launching script. More details can be found [here](https://huggingface.co/docs/accelerate/package_reference/cli#accelerate-launch). 
```
accelerate launch train_tuneavideo.py --config="configs/man-skiing.yaml"
```

### Important Notes
* n_sample_frames is the number of frames used for training the model.
* video_length is the number of frames used for inference (i.e., generating new videos).
* Please note that if you have 24 frames for training, you need to set video_length to 12 for inference.

### conda list 
If you have any issues, refer to the packages list in my working environment at [https://github.com/shani1610/object-insertion-video-diffusion/blob/main/conda_list.md](https://github.com/shani1610/object-insertion-video-diffusion/blob/main/conda_list.md) 

## Usage
Details on how to use the project.

## Contributing
Guidelines for contributing to the project.

## License
Information about the project's license.

## Acknowledgements
Special thanks and acknowledgements.
