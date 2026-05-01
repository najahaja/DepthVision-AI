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

# --- Premium UI Styling ---
css = """
#container {
    max-width: 1100px;
    margin: auto;
    padding-top: 2rem;
}
.header {
    text-align: center;
    margin-bottom: 2rem;
}
.footer {
    text-align: center;
    margin-top: 3rem;
    padding: 2rem;
    border-top: 1px solid #e5e7eb;
    color: #6b7280;
}
.group-members {
    font-weight: 600;
    color: #374151;
}
#img-display-input, #img-display-output {
    border-radius: 12px;
    overflow: hidden;
    box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1);
}
.download-file { 
    height: 62px; 
}
"""

# --- Model Loading (Backend remains same) ---
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}
encoder = 'vitl'
model = DepthAnythingV2(**model_configs[encoder])
filepath = hf_hub_download(repo_id=f"depth-anything/Depth-Anything-V2-Large", filename=f"depth_anything_v2_{encoder}.pth", repo_type="model")
state_dict = torch.load(filepath, map_location="cpu")
model.load_state_dict(state_dict)
model = model.to(DEVICE).eval()

@spaces.GPU
def predict_depth(image):
    return model.infer_image(image)

# --- App Structure ---
with gr.Blocks(css=css, title="Semester Project") as demo:
    with gr.Div(elem_id="container"):
        # Header
        gr.Markdown(
            """
            <div class="header">
                <h1>📐 Depth Vision AI</h1>
                <p>Advanced Monocular Depth Estimation using Depth-Anything-V2</p>
                <p style="font-size: 0.9rem; color: #6b7280;">Neural Networks & Image Processing | Semester Project</p>
            </div>
            """
        )

        with gr.Row():
            with gr.Column():
                input_image = gr.Image(label="Input Image", type='numpy', elem_id='img-display-input')
                submit = gr.Button(value="Estimate Depth", variant="primary")
            
            with gr.Column():
                depth_image_output = gr.Image(label="Estimated Depth Map", elem_id='img-display-output')
        
        with gr.Row():
            gray_depth_file = gr.File(label="Download Grayscale (PNG)", elem_id="download-gray", elem_classes="download-file")
            raw_file = gr.File(label="Download Raw Disparity (16-bit)", elem_id="download-raw", elem_classes="download-file")

        # Examples Section (Re-enabled for professional look)
        example_dir = 'assets/examples'
        if os.path.exists(example_dir) and os.listdir(example_dir):
            gr.Markdown("### 🖼️ Try with Examples")
            example_files = [os.path.join(example_dir, f) for f in os.listdir(example_dir) if f.endswith(('.jpg', '.png'))]
            gr.Examples(
                examples=example_files[:6], 
                inputs=[input_image], 
                outputs=[depth_image_output, gray_depth_file, raw_file], 
                fn=None, # User clicks button themselves
                cache_examples=False
            )

        # Footer
        gr.Markdown(
            """
            <div class="footer">
                <p>Project Supervisor: <b>Engr. Faheem Ul Rehman Siddiqi</b></p>
                <p>Developed by: <span class="group-members">Ahamed Najah</span> & <span class="group-members">Abdul Rahman Al tahir</span></p>
                <p>© 2024 Depth Vision AI | Semester Project Submission</p>
            </div>
            """
        )

    # --- Interaction Logic ---
    cmap = matplotlib.colormaps.get_cmap('Spectral_r')

    def on_submit(image):
        if image is None: return [None, None, None]
        
        depth = predict_depth(image[:, :, ::-1])

        # Prepare Downloads
        raw_depth = Image.fromarray(depth.astype('uint16'))
        tmp_raw_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        raw_depth.save(tmp_raw_depth.name)

        depth_norm = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_norm = depth_norm.astype(np.uint8)
        colored_depth = (cmap(depth_norm)[:, :, :3] * 255).astype(np.uint8)

        gray_depth = Image.fromarray(depth_norm)
        tmp_gray_depth = tempfile.NamedTemporaryFile(suffix='.png', delete=False)
        gray_depth.save(tmp_gray_depth.name)

        return [colored_depth, tmp_gray_depth.name, tmp_raw_depth.name]

    submit.click(on_submit, inputs=[input_image], outputs=[depth_image_output, gray_depth_file, raw_file])

if __name__ == '__main__':
    demo.queue().launch(server_name="0.0.0.0", share=False)
