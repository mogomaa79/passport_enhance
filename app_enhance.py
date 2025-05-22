import os
import subprocess
import spaces
import torch
import cv2
import uuid
import gradio as gr
import numpy as np

from PIL import Image
from basicsr.archs.srvgg_arch import SRVGGNetCompact
from gfpgan.utils import GFPGANer
from realesrgan.utils import RealESRGANer

def runcmd(cmd, verbose = False):

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


if not os.path.exists('GFPGANv1.4.pth'):
    runcmd("wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P .")
if not os.path.exists('realesr-general-x4v3.pth'):
    runcmd("wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth -P .")



model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
model_path = 'realesr-general-x4v3.pth'
half = True if torch.cuda.is_available() else False
upsampler = RealESRGANer(scale=4, model_path=model_path, model=model, tile=0, tile_pad=10, pre_pad=0, half=half)


@spaces.GPU(duration=15)
def enhance_image(
    input_image: Image,
    scale: int,
    enhance_mode: str,
):
    only_face = enhance_mode == "Only Face Enhance"
    if enhance_mode == "Only Face Enhance":
        face_enhancer = GFPGANer(model_path='GFPGANv1.4.pth', upscale=scale, arch='clean', channel_multiplier=2)
    elif enhance_mode == "Only Image Enhance":
        face_enhancer = None
    else:
        face_enhancer = GFPGANer(model_path='GFPGANv1.4.pth', upscale=scale, arch='clean', channel_multiplier=2, bg_upsampler=upsampler)
    
    img = cv2.cvtColor(np.array(input_image), cv2.COLOR_RGB2BGR)

    h, w = img.shape[0:2]
    if h < 300:
        img = cv2.resize(img, (w * 2, h * 2), interpolation=cv2.INTER_LANCZOS4)
    
    if face_enhancer is not None:
        _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=only_face, paste_back=True)
    else:
        output, _ = upsampler.enhance(img, outscale=scale)

    # if scale != 2:
    #     interpolation = cv2.INTER_AREA if scale < 2 else cv2.INTER_LANCZOS4
    #     h, w = img.shape[0:2]
    #     output = cv2.resize(output, (int(w * scale / 2), int(h * scale / 2)), interpolation=interpolation)
    
    h, w = output.shape[0:2]
    max_size = 3480
    if h > max_size:
        w = int(w * max_size / h)
        h = max_size
    
    if w > max_size:
        h = int(h * max_size / w)
        w = max_size
    
    output = cv2.resize(output, (w, h), interpolation=cv2.INTER_LANCZOS4)

    enhanced_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
    tmpPrefix = "/tmp/gradio/"

    extension = 'png'

    targetDir = f"{tmpPrefix}output/"
    if not os.path.exists(targetDir):
        os.makedirs(targetDir)

    enhanced_path = f"{targetDir}{uuid.uuid4()}.{extension}"
    enhanced_image.save(enhanced_path, quality=100)
        
    return enhanced_image, enhanced_path


def create_demo() -> gr.Blocks:

    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                scale = gr.Slider(minimum=1, maximum=4, value=2, step=1, label="Scale")
            with gr.Column():
                enhance_mode = gr.Dropdown(
                    label="Enhance Mode",
                    choices=[
                        "Only Face Enhance",
                        "Only Image Enhance",
                        "Face Enhance + Image Enhance",
                    ],
                    value="Face Enhance + Image Enhance",
                )
                g_btn = gr.Button("Enhance Image")
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil")
            with gr.Column():
                output_image = gr.Image(label="Enhanced Image", type="pil", interactive=False)
                enhance_image_path = gr.File(label="Download the Enhanced Image", interactive=False)
                
        
        g_btn.click(
            fn=enhance_image,
            inputs=[input_image, scale, enhance_mode],
            outputs=[output_image, enhance_image_path],
        )

    return demo

if __name__ == "__main__":
    # run enhance_image without gradio just for test
    input_image = Image.open("/content/passport_enhance/test.jpg")
    output_image, enhance_image_path = enhance_image(input_image, 2, "Only Image Enhance")
    output_image.show()
    # save the output image
    output_image.save("output.png")
