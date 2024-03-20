import cv2
import random
import gradio as gr
from colorsys import rgb_to_hls, hls_to_rgb


markdown_default = """
<link href="https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap" rel="stylesheet">
<style>
        .highlighted-text {
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
            font-size: 14px;
            color: rgb(255, 255, 239);
            background-color: rgb(225, 231, 254);
            border-radius: 7px;
            padding: 5px 7px;
            display: inline-block;
        }
        .regular-text {
            font-family: 'Montserrat', sans-serif;
            font-weight: 400;
            font-size: 14px;
        }
        .highlighted-response {
            font-family: 'Montserrat', sans-serif;
            font-weight: 600;
            font-size: 14px;
            border-radius: 6px;
            padding: 3px 4px;
            display: inline-block;
        }
</style>
<span class="highlighted-text" style='color:rgb(107, 100, 239)'>GLaMM</span>

"""

examples = [
    ["Could you provide me with a detailed analysis of this photo? Please output with interleaved segmentation masks"
     " for the corresponding parts of the answer.", "./images/demo_resources/tokyo.jpg", ],
    ["Could you give a comprehensive explanation of what can be found within this picture? "
     "Please output with interleaved segmentation masks for the corresponding phrases.",
        "./images/demo_resources/mansion.jpg", ],
    ["Can you please segment the yacht ?", "./images/demo_resources/yacht.jpg", ],
    ["Can you segment the hot air balloon ?", "./images/demo_resources/balloon.jpg", ],
    ["Could you please give me a detailed description of the image ?", "./images/demo_resources/beetle.jpg", ],
    ["Could you provide me with a detailed analysis of this photo? "
     "Please output with interleaved segmentation masks for the corresponding parts of the answer.",
        "./images/demo_resources/joker.png", ],
    ["Can you segment what the person is using to ride ?", "./images/demo_resources/surfer.jpg", ],
    ["Can you segment the water around the person ?", "./images/demo_resources/paddle.jpg", ],
    ["Could you provide me with a detailed analysis of this photo? "
     "Please output with interleaved segmentation masks for the corresponding parts of the answer.",
        "./images/demo_resources/snow.png", ],
    ["What is she doing in this image ?", "./images/demo_resources/japan.jpg", ], ]

title = "GLaMM : Grounding Large Multimodal Model"

description = """
[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://mbzuai-oryx.github.io/groundingLMM)

**Usage** : <br>
&ensp;(1) For **Grounded Caption Generation** Interleaved Segmentation, input prompt like: *"Could you provide me with a detailed analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer."* <br>
&ensp;(2) For **Segmentation Output**, input prompt like: *"Can you please segment xxx in the given image"* <br>
&ensp;(3) To **Input Regions** : Draw boudning boxes over the uploaded image and input prompt like: *"Can you please describe this region &lt;bbox&gt;"* Need to give &lt;bbox&gt; identifier <br>
&ensp;(4) For **Image Captioning** VQA, input prompt like: *"Could you please give me a detailed description of the image?"* <br>
&ensp;(5) For **Conditional Generation** Image manipulation, first perform (2) then select generate and input prompt which describes the new image to be generated <br>
"""

article = """
<center> This is the online demo of GLaMM from MBZUAI. \n </center>
"""

colors = [
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [0, 255, 255],    # Cyan
        [255, 0, 255],    # Magenta
        [255, 192, 203],  # Pink
        [165, 42, 42],    # Brown
        [255, 165, 0],    # Orange
        [128, 0, 128],     # Purple
        [0, 0, 128],       # Navy
        [128, 0, 0],      # Maroon
        [128, 128, 0],    # Olive
        [70, 130, 180],   # Steel Blue
        [173, 216, 230],  # Light Blue
        [255, 192, 0],    # Gold
        [255, 165, 165],  # Light Salmon
        [255, 20, 147],   # Deep Pink
    ]


def process_markdown(output_str, color_history):
    markdown_out = output_str.replace('[SEG]', '')
    markdown_out = markdown_out.replace(
        "<p>", "<span class='highlighted-response' style='background-color:rgb[COLOR]'>"
    )
    markdown_out = markdown_out.replace("</p>", "</span>")

    for color in color_history:
        markdown_out = markdown_out.replace("[COLOR]", str(desaturate(tuple(color))), 1)

    markdown_out = f""" 
    <br>
    {markdown_out}

    """
    markdown_out = markdown_default + "<p><span class='regular-text'>" + markdown_out + '</span></p>'
    return markdown_out


def desaturate(rgb, factor=0.65):
    """
    Desaturate an RGB color by a given factor.

    :param rgb: A tuple of (r, g, b) where each value is in [0, 255].
    :param factor: The factor by which to reduce the saturation.
                   0 means completely desaturated, 1 means original color.
    :return: A tuple of desaturated (r, g, b) values in [0, 255].
    """
    r, g, b = [x / 255.0 for x in rgb]
    h, l, s = rgb_to_hls(r, g, b)
    l = factor
    new_r, new_g, new_b = hls_to_rgb(h, l, s)
    return (int(new_r * 255), int(new_g * 255), int(new_b * 255))


def draw_bbox(image, boxes, color_history=[]):

    colors = [
        [255, 0, 0],      # Red
        [0, 255, 0],      # Green
        [0, 0, 255],      # Blue
        [255, 255, 0],    # Yellow
        [0, 255, 255],    # Cyan
        [255, 0, 255],    # Magenta
        [255, 192, 203],  # Pink
        [165, 42, 42],    # Brown
        [255, 165, 0],    # Orange
        [128, 0, 128],     # Purple
        [0, 0, 128],       # Navy
        [128, 0, 0],      # Maroon
        [128, 128, 0],    # Olive
        [70, 130, 180],   # Steel Blue
        [173, 216, 230],  # Light Blue
        [255, 192, 0],    # Gold
        [255, 165, 165],  # Light Salmon
        [255, 20, 147],   # Deep Pink
    ]
    new_image = image
    text = '<region_1>'
    font = cv2.FONT_HERSHEY_TRIPLEX
    font_scale = 1.0
    thickness = 4
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)
    for bbox_id, box in enumerate(boxes):
        if len(color_history) == 0 :
            color = tuple(random.choice(colors))
        else :
            color = color_history[bbox_id]

        start_point = int(box[0]), int(box[1])
        end_point = int(box[2]), int(box[3])
        new_image = cv2.rectangle(new_image, start_point, end_point, color, thickness)
        if len(color_history) == 0 :
            new_image = cv2.putText(new_image,
                                    f'<region {bbox_id + 1}>',
                                    (int(box[0]), int(box[1]) + text_size[1]), font, font_scale, color, thickness)

    return new_image


class ImageSketcher(gr.Image):
    """Code is from https://github.com/ttengwang/Caption-
    Anything/blob/main/app.py#L32.

    Fix the bug of gradio.Image that cannot upload with tool == 'sketch'.
    """

    is_template = True  # Magic to make this work with gradio.Block, don't remove unless you know what you're doing.

    def __init__(self, **kwargs):
        super().__init__(tool='boxes', **kwargs)

    def preprocess(self, x):
        if x is None:
            return x
        self.DEFAULT_TEMP_DIR = 'gradio_out/' #Make Dir if required
        if self.tool == 'boxes' and self.source in ['upload', 'webcam']:
            if isinstance(x, str):
                x = {'image': x, 'boxes': []}
            else:
                assert isinstance(x, dict)
                assert isinstance(x['image'], str)
                assert isinstance(x['boxes'], list)
        x = super().preprocess(x)
        return x