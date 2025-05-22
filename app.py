import gradio as gr

from app_enhance import create_demo as create_demo_enhance

with gr.Blocks(css="style.css") as demo:
    with gr.Tabs():
        with gr.Tab(label="Enhance"):
            create_demo_enhance()

demo.launch()