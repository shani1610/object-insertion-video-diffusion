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
packages in environment at /root/anaconda3/envs/tune-video:

#
# Name                    Version                   Build  Channel
_libgcc_mutex             0.1                        main  
_openmp_mutex             5.1                       1_gnu  
absl-py                   2.1.0                    pypi_0    pypi
accelerate                0.16.0                   pypi_0    pypi
aiofiles                  23.2.1                   pypi_0    pypi
altair                    5.3.0                    pypi_0    pypi
annotated-types           0.7.0                    pypi_0    pypi
antlr4-python3-runtime    4.9.3                    pypi_0    pypi
anyio                     4.4.0                    pypi_0    pypi
attrs                     23.2.0                   pypi_0    pypi
bitsandbytes              0.35.4                   pypi_0    pypi
bzip2                     1.0.8                h5eee18b_6  
ca-certificates           2024.3.11            h06a4308_0  
cachetools                5.3.3                    pypi_0    pypi
certifi                   2024.6.2                 pypi_0    pypi
charset-normalizer        3.3.2                    pypi_0    pypi
click                     8.1.7                    pypi_0    pypi
cmake                     3.29.5.1                 pypi_0    pypi
contourpy                 1.2.1                    pypi_0    pypi
cycler                    0.12.1                   pypi_0    pypi
decord                    0.6.0                    pypi_0    pypi
diffusers                 0.11.1                   pypi_0    pypi
dnspython                 2.6.1                    pypi_0    pypi
einops                    0.6.0                    pypi_0    pypi
email-validator           2.1.2                    pypi_0    pypi
exceptiongroup            1.2.1                    pypi_0    pypi
fastapi                   0.111.0                  pypi_0    pypi
fastapi-cli               0.0.4                    pypi_0    pypi
ffmpy                     0.3.2                    pypi_0    pypi
filelock                  3.15.1                   pypi_0    pypi
fonttools                 4.53.0                   pypi_0    pypi
fsspec                    2024.6.0                 pypi_0    pypi
ftfy                      6.1.1                    pypi_0    pypi
google-auth               2.30.0                   pypi_0    pypi
google-auth-oauthlib      0.4.6                    pypi_0    pypi
gradio                    4.36.1                   pypi_0    pypi
gradio-client             1.0.1                    pypi_0    pypi
grpcio                    1.64.1                   pypi_0    pypi
h11                       0.14.0                   pypi_0    pypi
httpcore                  1.0.5                    pypi_0    pypi
httptools                 0.6.1                    pypi_0    pypi
httpx                     0.27.0                   pypi_0    pypi
huggingface-hub           0.23.4                   pypi_0    pypi
idna                      3.7                      pypi_0    pypi
imageio                   2.25.0                   pypi_0    pypi
imageio-ffmpeg            0.5.1                    pypi_0    pypi
importlib-metadata        7.1.0                    pypi_0    pypi
importlib-resources       6.4.0                    pypi_0    pypi
jinja2                    3.1.4                    pypi_0    pypi
jsonschema                4.22.0                   pypi_0    pypi
jsonschema-specifications 2023.12.1                pypi_0    pypi
kiwisolver                1.4.5                    pypi_0    pypi
ld_impl_linux-64          2.38                 h1181459_1  
libffi                    3.3                  he6710b0_2  
libgcc-ng                 11.2.0               h1234567_1  
libgomp                   11.2.0               h1234567_1  
libstdcxx-ng              11.2.0               h1234567_1  
libuuid                   1.41.5               h5eee18b_0  
lit                       18.1.7                   pypi_0    pypi
markdown                  3.6                      pypi_0    pypi
markdown-it-py            3.0.0                    pypi_0    pypi
markupsafe                2.1.5                    pypi_0    pypi
matplotlib                3.9.0                    pypi_0    pypi
mdurl                     0.1.2                    pypi_0    pypi
modelcards                0.1.6                    pypi_0    pypi
mpmath                    1.3.0                    pypi_0    pypi
mypy-extensions           1.0.0                    pypi_0    pypi
ncurses                   6.4                  h6a678d5_0  
networkx                  3.3                      pypi_0    pypi
ninja                     1.11.1.1                 pypi_0    pypi
numpy                     1.21.2                   pypi_0    pypi
nvidia-cublas-cu11        11.10.3.66               pypi_0    pypi
nvidia-cublas-cu12        12.1.3.1                 pypi_0    pypi
nvidia-cuda-cupti-cu11    11.7.101                 pypi_0    pypi
nvidia-cuda-cupti-cu12    12.1.105                 pypi_0    pypi
nvidia-cuda-nvrtc-cu11    11.7.99                  pypi_0    pypi
nvidia-cuda-nvrtc-cu12    12.1.105                 pypi_0    pypi
nvidia-cuda-runtime-cu11  11.7.99                  pypi_0    pypi
nvidia-cuda-runtime-cu12  12.1.105                 pypi_0    pypi
nvidia-cudnn-cu11         8.5.0.96                 pypi_0    pypi
nvidia-cudnn-cu12         8.9.2.26                 pypi_0    pypi
nvidia-cufft-cu11         10.9.0.58                pypi_0    pypi
nvidia-cufft-cu12         11.0.2.54                pypi_0    pypi
nvidia-curand-cu11        10.2.10.91               pypi_0    pypi
nvidia-curand-cu12        10.3.2.106               pypi_0    pypi
nvidia-cusolver-cu11      11.4.0.1                 pypi_0    pypi
nvidia-cusolver-cu12      11.4.5.107               pypi_0    pypi
nvidia-cusparse-cu11      11.7.4.91                pypi_0    pypi
nvidia-cusparse-cu12      12.1.0.106               pypi_0    pypi
nvidia-nccl-cu11          2.14.3                   pypi_0    pypi
nvidia-nccl-cu12          2.20.5                   pypi_0    pypi
nvidia-nvjitlink-cu12     12.5.40                  pypi_0    pypi
nvidia-nvtx-cu11          11.7.91                  pypi_0    pypi
nvidia-nvtx-cu12          12.1.105                 pypi_0    pypi
oauthlib                  3.2.2                    pypi_0    pypi
omegaconf                 2.3.0                    pypi_0    pypi
openssl                   1.1.1w               h7f8727e_0  
orjson                    3.10.5                   pypi_0    pypi
packaging                 24.1                     pypi_0    pypi
pandas                    2.2.2                    pypi_0    pypi
peft                      0.11.1                   pypi_0    pypi
pillow                    10.3.0                   pypi_0    pypi
pip                       24.0            py310h06a4308_0  
protobuf                  3.20.3                   pypi_0    pypi
psutil                    5.9.8                    pypi_0    pypi
pyasn1                    0.6.0                    pypi_0    pypi
pyasn1-modules            0.4.0                    pypi_0    pypi
pydantic                  2.7.4                    pypi_0    pypi
pydantic-core             2.18.4                   pypi_0    pypi
pydub                     0.25.1                   pypi_0    pypi
pygments                  2.18.0                   pypi_0    pypi
pyparsing                 3.1.2                    pypi_0    pypi
pyre-extensions           0.0.23                   pypi_0    pypi
python                    3.10.6               haa1d7c7_1  
python-dateutil           2.9.0.post0              pypi_0    pypi
python-dotenv             1.0.1                    pypi_0    pypi
python-multipart          0.0.9                    pypi_0    pypi
pytz                      2024.1                   pypi_0    pypi
pyyaml                    6.0.1                    pypi_0    pypi
readline                  8.2                  h5eee18b_0  
referencing               0.35.1                   pypi_0    pypi
regex                     2024.5.15                pypi_0    pypi
requests                  2.32.3                   pypi_0    pypi
requests-oauthlib         2.0.0                    pypi_0    pypi
rich                      13.7.1                   pypi_0    pypi
rpds-py                   0.18.1                   pypi_0    pypi
rsa                       4.9                      pypi_0    pypi
ruff                      0.4.9                    pypi_0    pypi
safetensors               0.4.3                    pypi_0    pypi
semantic-version          2.10.0                   pypi_0    pypi
setuptools                69.5.1          py310h06a4308_0  
shellingham               1.5.4                    pypi_0    pypi
six                       1.16.0                   pypi_0    pypi
sniffio                   1.3.1                    pypi_0    pypi
sqlite                    3.45.3               h5eee18b_0  
starlette                 0.37.2                   pypi_0    pypi
sympy                     1.12.1                   pypi_0    pypi
tensorboard               2.11.2                   pypi_0    pypi
tensorboard-data-server   0.6.1                    pypi_0    pypi
tensorboard-plugin-wit    1.8.1                    pypi_0    pypi
tk                        8.6.14               h39e8969_0  
tokenizers                0.13.3                   pypi_0    pypi
tomlkit                   0.12.0                   pypi_0    pypi
toolz                     0.12.1                   pypi_0    pypi
torch                     1.13.1+cu117             pypi_0    pypi
torchvision               0.14.1+cu117             pypi_0    pypi
tqdm                      4.66.4                   pypi_0    pypi
transformers              4.26.0                   pypi_0    pypi
triton                    2.0.0                    pypi_0    pypi
typer                     0.12.3                   pypi_0    pypi
typing-extensions         4.12.2                   pypi_0    pypi
typing-inspect            0.9.0                    pypi_0    pypi
tzdata                    2024.1                   pypi_0    pypi
ujson                     5.10.0                   pypi_0    pypi
urllib3                   2.2.1                    pypi_0    pypi
uvicorn                   0.30.1                   pypi_0    pypi
uvloop                    0.19.0                   pypi_0    pypi
watchfiles                0.22.0                   pypi_0    pypi
wcwidth                   0.2.13                   pypi_0    pypi
websockets                11.0.3                   pypi_0    pypi
werkzeug                  3.0.3                    pypi_0    pypi
wheel                     0.43.0          py310h06a4308_0  
xformers                  0.0.16                   pypi_0    pypi
xz                        5.4.6                h5eee18b_1  
zipp                      3.19.2                   pypi_0    pypi
zlib                      1.2.13               h5eee18b_1  
