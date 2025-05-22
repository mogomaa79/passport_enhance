import requests
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionUpscalePipeline
import torch
import gradio as gr
import time
import spaces

from segment_utils import(
    segment_image,
    restore_result,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f'{device} is available')

model_id = "stabilityai/stable-diffusion-x4-upscaler"
upscale_pipe = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
upscale_pipe = upscale_pipe.to(device)

DEFAULT_SRC_PROMPT = "a person with pefect face"
DEFAULT_CATEGORY = "face"


def create_demo() -> gr.Blocks:

    @spaces.GPU(duration=30)
    def upscale_image(
        input_image: Image,
        prompt: str,
        num_inference_steps: int = 10,
    ):
        time_cost_str = ''
        run_task_time = 0
        run_task_time, time_cost_str = get_time_cost(run_task_time, time_cost_str)
        upscaled_image = upscale_pipe(
            prompt=prompt, 
            image=input_image,
            num_inference_steps=num_inference_steps,
        ).images[0]
        run_task_time, time_cost_str = get_time_cost(run_task_time, time_cost_str)
        
        return upscaled_image, time_cost_str

    def get_time_cost(run_task_time, time_cost_str):
        now_time = int(time.time()*1000)
        if run_task_time == 0:
            time_cost_str = 'start'
        else:
            if time_cost_str != '': 
                time_cost_str += f'-->'
            time_cost_str += f'{now_time - run_task_time}'
        run_task_time = now_time
        return run_task_time, time_cost_str

    with gr.Blocks() as demo:
        croper = gr.State()
        with gr.Row():
            with gr.Column():
                input_image_prompt = gr.Textbox(lines=1, label="Input Image Prompt", value=DEFAULT_SRC_PROMPT)
            with gr.Column():
                num_inference_steps = gr.Number(label="Num Inference Steps", value=5)
                generate_size = gr.Number(label="Generate Size", value=512)
                g_btn = gr.Button("Upscale Image")
                
        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type="pil")
            with gr.Column():
                restored_image = gr.Image(label="Restored Image", format="png", type="pil", interactive=False)
                origin_area_image = gr.Image(label="Origin Area Image", format="png", type="pil", interactive=False, visible=False)
                upscaled_image = gr.Image(label="Upscaled Image", format="png", type="pil", interactive=False)
                download_path = gr.File(label="Download the output image", interactive=False)
                generated_cost = gr.Textbox(label="Time cost by step (ms):", visible=True, interactive=False)
                category = gr.Textbox(label="Category", value=DEFAULT_CATEGORY, visible=False)
                mask_expansion = gr.Number(label="Mask Expansion", value=20, visible=False)
                mask_dilation = gr.Slider(minimum=0, maximum=10, value=2, step=1, label="Mask Dilation", visible=False)

        g_btn.click(
            fn=segment_image,
            inputs=[input_image, category, generate_size, mask_expansion, mask_dilation],
            outputs=[origin_area_image, croper],
        ).success(
            fn=upscale_image,
            inputs=[origin_area_image, input_image_prompt, num_inference_steps],
            outputs=[upscaled_image, generated_cost],
        ).success(
            fn=restore_result,
            inputs=[croper, category, upscaled_image],
            outputs=[restored_image, download_path],
        )

    return demo
