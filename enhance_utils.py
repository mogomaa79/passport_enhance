import os
import torch
import cv2
import numpy as np
import subprocess

from PIL import Image
from gfpgan.utils import GFPGANer
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from realesrgan.utils import RealESRGANer

def runcmd(cmd, verbose = False, *args, **kwargs):

    process = subprocess.Popen(
        cmd,
        stdout = subprocess.PIPE,
        stderr = subprocess.PIPE,
        text = True,
        shell = True
    )
    std_out, std_err = process.communicate()
    if verbose:
        print(std_out.strip(), std_err)
    pass

runcmd("pip freeze")
if not os.path.exists('GFPGANv1.4.pth'):
    runcmd("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P .")
if not os.path.exists('realesr-general-x4v3.pth'):
    runcmd("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P .")

model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
model_path = 'realesr-general-x4v3.pth'
half = True if torch.cuda.is_available() else False
upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)

face_enhancer = GFPGANer(model_path='GFPGANv1.4.pth', upscale=1, arch='clean', channel_multiplier=2)

def enhance_image(
    pil_image: Image,
    enhance_face: bool = False,
):
    img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    h, w = img.shape[0:2]
    if h < 300:
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    if enhance_face:
        _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=True, paste_back=True)
    else:
        output, _ = upsampler.enhance(img, outscale=2)
    pil_output = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))

    return pil_output