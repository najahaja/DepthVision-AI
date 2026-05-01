import gradio as gr
import cv2
import matplotlib
import numpy as np
import os
from PIL import Image
try:
    import spaces
    HAS_SPACES = True
except ImportError:
    HAS_SPACES = False
    class spaces:
        @staticmethod
        def GPU(fn):
            return fn

import torch
import tempfile
from huggingface_hub import hf_hub_download

from depth_anything_v2.dpt import DepthAnythingV2

css = """
#img-display-input { max-height: 80vh; }
#img-display-output { max-height: 80vh; }
.download-file { height: 62px; }
"""
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}
encoder2name = {
    'vits': 'Small',
    'vitb': 'Base',
    'vitl': 'Large',
    'vitg': 'Giant',
}
encoder = 'vitl'
model_name = encoder2name[encoder]
model = DepthAnythingV2(**model_configs[encoder])
filepath = hf_hub_download(repo_id=f"depth-anything/Depth-Anything-V2-{model_name}", filename=f"depth_anything_v2_{encoder}.pth", repo_type="model")
state_dict = torch.load(filepath, map_location="cpu")
model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()

title = "# Depth Vision AI"
description = """Official demo for **Depth Vision AI**.
Please refer to our [github](https://github.com/DepthAnything/Depth-Anything-V2) for more details."""

@spaces.GPU
def predict_depth(image):
    return model.infer_image(image)

with gr.Blocks(css=css) as demo:
    gr.Markdown(title)
    gr.Markdown(description)
    gr.Markdown("### Depth Prediction demo")

    with gr.Row():
        input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
        depth_image_output = gr.Image(label="Depth Map", elem_id='img-display-output')
    submit = gr.Button(value="Compute Depth")
    gray_depth_file = gr.File(label="Grayscale depth map", elem_id="download-gray", elem_classes="download-file")
    raw_file = gr.File(label="16-bit raw output (can be considered as disparity)", elem_id="download-raw", elem_classes="download-file")

    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    def on_submit(image):
        if image is None:
            return [None, None, None]
        
        original_image = image.copy()
        h, w = image.shape[:2]
        depth = predict_depth(image[:, :, ::-1])

        raw_depth = Image.fromarray(depth.astype('uint16'))
        tmp_raw_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        raw_depth.save(tmp_raw_depth.name)

        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        colored_depth = (cmap(depth)[:, :, :3] * 255).astype(np.uint8)

        gray_depth = Image.fromarray(depth)
        tmp_gray_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        gray_depth.save(tmp_gray_depth.name)

        print(f"Depth prediction complete. Min: {depth.min()}, Max: {depth.max()}")
        return [colored_depth, tmp_gray_depth.name, tmp_raw_depth.name]

    submit.click(on_submit, inputs=[input_image], outputs=[depth_image_output, gray_depth_file, raw_file])

if __name__ == '__main__':
    # Use share=False for maximum local stability
    demo.queue().launch(share=False, server_name="127.0.0.1")
