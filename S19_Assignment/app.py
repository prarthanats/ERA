# -*- coding: utf-8 -*-
"""
Created on Fri Oct  6 17:53:27 2023
@author: prarthana.ts
"""

from ultralytics import YOLO
import gradio as gr
import torch
from utils.gradio_tools import fast_process
from utils.tools import format_results, box_prompt, point_prompt, text_prompt
from PIL import ImageDraw
import numpy as np

# Load the pre-trained model
model = YOLO('./weights/FastSAM.pt')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


icon_html = '<i class="fas fa-chart-bar" style="font-weight: bold;"></i>'
title = f"""
<div style="background-color: #f5f1f2; padding: 10px; display: flex; align-items: center; justify-content: center;">
    {icon_html} <span style="margin-left: 10px; font-size: 50px;">Fast Segment Anything</span>
</div>
"""

description = f"""
<div style="background-color: #f1f1f5; padding: 10px; display: flex; align-items: center;">
    {icon_html}
    <span style="margin-left: 10px;">
        <p><strong>Pre trained Implementation of fast sam</strong></p>
        <p>Note:Please provide the text with respect to what needs to be segmented</p>
    </span>
</div>
"""
css = "h1 { text-align: center } .about { text-align: justify; padding-left: 10%; padding-right: 10%; }"

def segment_everything(
    input,
    input_size=1024, 
    iou_threshold=0.7,
    conf_threshold=0.25,
    better_quality=False,
    withContours=True,
    use_retina=True,
    text="",
    wider=False,
    mask_random_color=True,
):
    input_size = int(input_size)  
    w, h = input.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input = input.resize((new_w, new_h))

    results = model(input,
                    device=device,
                    retina_masks=True,
                    iou=iou_threshold,
                    conf=conf_threshold,
                    imgsz=input_size,)

    if len(text) > 0:
        results = format_results(results[0], 0)
        annotations, _ = text_prompt(results, text, input, device=device, wider=wider)
        annotations = np.array([annotations])
    else:
        annotations = results[0].masks.data
    
    fig = fast_process(annotations=annotations,
                       image=input,
                       device=device,
                       scale=(1024 // input_size),
                       better_quality=better_quality,
                       mask_random_color=mask_random_color,
                       bbox=None,
                       use_retina=use_retina,
                       withContours=withContours,)
    return fig


def segment_with_points(
    input,
    input_size=1024, 
    iou_threshold=0.7,
    conf_threshold=0.25,
    better_quality=False,
    withContours=True,
    use_retina=True,
    mask_random_color=True,
):
    global global_points
    global global_point_label
    
    input_size = int(input_size)
    w, h = input.size
    scale = input_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    input = input.resize((new_w, new_h))
    
    scaled_points = [[int(x * scale) for x in point] for point in global_points]

    results = model(input,
                    device=device,
                    retina_masks=True,
                    iou=iou_threshold,
                    conf=conf_threshold,
                    imgsz=input_size,)
    
    results = format_results(results[0], 0)
    annotations, _ = point_prompt(results, scaled_points, global_point_label, new_h, new_w)
    annotations = np.array([annotations])

    fig = fast_process(annotations=annotations,
                       image=input,
                       device=device,
                       scale=(1024 // input_size),
                       better_quality=better_quality,
                       mask_random_color=mask_random_color,
                       bbox=None,
                       use_retina=use_retina,
                       withContours=withContours,)

    global_points = []
    global_point_label = []
    return fig, None


def get_points_with_draw(image, label, evt: gr.SelectData):
    global global_points
    global global_point_label

    x, y = evt.index[0], evt.index[1]
    point_radius, point_color = 15, (255, 255, 0) if label == 'Add Mask' else (255, 0, 255)
    global_points.append([x, y])
    global_point_label.append(1 if label == 'Add Mask' else 0)
    
    print(x, y, label == 'Add Mask')
    
    draw = ImageDraw.Draw(image)
    draw.ellipse([(x - point_radius, y - point_radius), (x + point_radius, y + point_radius)], fill=point_color)
    return image


cond_img_t = gr.Image(label="Input with text", value="examples/dogs.jpg", type='pil')
segm_img_t = gr.Image(label="Segmented Image with text", interactive=False, type='pil')

global_points = []
global_point_label = []

input_size_slider = gr.components.Slider(minimum=512,
                                         maximum=1024,
                                         value=1024,
                                         step=64,
                                         label='Input_size',
                                         info='The model was trained on a size of 1024')

with gr.Blocks(css=css, title='Fast Segment Anything') as demo:
    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(title)

    with gr.Row():
        with gr.Column(scale=1):
            # Title
            gr.Markdown(description)

    with gr.Tab("Text mode"):
        # Images
        with gr.Row(variant="panel"):
            with gr.Column(scale=1):
                cond_img_t.render()

            with gr.Column(scale=1):
                segm_img_t.render()

        # Submit & Clear
        with gr.Row():
            with gr.Column():
                input_size_slider_t = gr.components.Slider(minimum=512,
                                                           maximum=1024,
                                                           value=1024,
                                                           step=64,
                                                           label='Input_size',
                                                           info='Our model was trained on a size of 1024')
                with gr.Row():
                    with gr.Column():
                        contour_check = gr.Checkbox(value=True, label='withContours', info='draw the edges of the masks')
                        text_box = gr.Textbox(label="text prompt", value="a black dog")

                    with gr.Column():
                        segment_btn_t = gr.Button("Segment with text", variant='primary')
                        clear_btn_t = gr.Button("Clear", variant="secondary")

                gr.Markdown("Click on the examples below")
                gr.Examples(examples=[["examples/dogs.jpg"],["examples/cat.jpg"],["examples/fruit.jpg"],["examples/flower.jpg"],["examples/boat.jpg"],["examples/fruits.jpg"],],
                            inputs=[cond_img_t],
                            examples_per_page=6)
            
        with gr.Column():
                with gr.Accordion("Advanced options", open=False):
                    iou_threshold = gr.Slider(0.1, 0.9, 0.7, step=0.1, label='iou', info='iou threshold for filtering the annotations')
                    conf_threshold = gr.Slider(0.1, 0.9, 0.25, step=0.05, label='conf', info='object confidence threshold')
                    with gr.Row():
                        mor_check = gr.Checkbox(value=False, label='better_visual_quality', info='better quality using morphologyEx')
                        retina_check = gr.Checkbox(value=True, label='use_retina', info='draw high-resolution segmentation masks')
                        wider_check = gr.Checkbox(value=False, label='wider', info='wider result')
    
    segment_btn_t.click(segment_everything,
                        inputs=[
                            cond_img_t,
                            input_size_slider_t,
                            iou_threshold,
                            conf_threshold,
                            mor_check,
                            contour_check,
                            retina_check,
                            text_box,
                            wider_check,
                        ],
                        outputs=segm_img_t)

    def clear():
        return None, None
    
    def clear_text():
        return None, None, None

    # clear_btn_p.click(clear, outputs=[cond_img_p, segm_img_p])
    clear_btn_t.click(clear_text, outputs=[text_box])

demo.queue()
demo.launch()