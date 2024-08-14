import torch
from torch import autocast
from diffusers import DDIMScheduler
from tuneavideo.pipelines.pipeline_tuneavideo import TuneAVideoPipeline
from tuneavideo.models.unet import UNet3DConditionModel
from tuneavideo.util import save_videos_grid
from natsort import natsorted
from glob import glob
import os
import argparse

parser = argparse.ArgumentParser(description="infer",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-o", "--outputdir", default="outputs/original/racket_2_start30", help="output directory")
parser.add_argument("-m", "--modelname", default="CompVis/stable-diffusion-v1-4", help="model name")
parser.add_argument("-p", "--prompt", default="spider man is swinging a green tennis racket", help="prompt for generating the video")
parser.add_argument("-s", "--savepath", default="./results/prompt.gif", help="prompt for generating the video")
args = parser.parse_args()


if args.outputdir:
	OUTPUT_DIR = args.outputdir
	
if args.modelname:
	MODEL_NAME = args.modelname

if args.prompt:
	prompt = args.prompt

if args.savepath:
	save_path = f"{args.savepath}"

unet = UNet3DConditionModel.from_pretrained(OUTPUT_DIR, subfolder='unet', torch_dtype=torch.float16).to('cuda')
scheduler = DDIMScheduler.from_pretrained(MODEL_NAME, subfolder='scheduler')
pipe = TuneAVideoPipeline.from_pretrained(MODEL_NAME, unet=unet, scheduler=scheduler, torch_dtype=torch.float16).to("cuda")
pipe.enable_xformers_memory_efficient_attention()
pipe.enable_vae_slicing()

g_cuda = None

# Set parameters for generating videos
negative_prompt = ""  #@param {type:"string"}
use_inv_latent = True  #@param {type:"boolean"}
inv_latent_path = ""  #@param {type:"string"}
num_samples = 1  #@param {type:"number"}
guidance_scale = 12.5  #@param {type:"number"}
num_inference_steps = 300  #@param {type:"number"}
video_length = 12  #@param {type:"number"}
height = 512  #@param {type:"number"}
width = 512  #@param {type:"number"}

ddim_inv_latent = None
if use_inv_latent and inv_latent_path == "":
    inv_latent_path = natsorted(glob(f"{OUTPUT_DIR}/inv_latents/*"))[-1]
    ddim_inv_latent = torch.load(inv_latent_path).to(torch.float16)
    print(f"DDIM inversion latent loaded from {inv_latent_path}")

# Verify the shape of the loaded latent tensor
if ddim_inv_latent is not None and ddim_inv_latent.shape[2] != video_length:
    raise ValueError(f"Loaded latent tensor has shape {ddim_inv_latent.shape}, but expected shape with video_length {video_length}.")

with autocast("cuda"), torch.inference_mode():
    videos = pipe(
        prompt,
        latents=ddim_inv_latent,
        video_length=video_length,
        height=height,
        width=width,
        negative_prompt=negative_prompt,
        num_videos_per_prompt=num_samples,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=g_cuda
    ).videos

#save_dir = "./results/original"  #@param {type:"string"}
#save_path = f"{save_dir}/{prompt}.gif"
save_videos_grid(videos, save_path)
