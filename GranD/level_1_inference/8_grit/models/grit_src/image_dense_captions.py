import sys
from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image

sys.path.insert(0, 'models/grit_src/third_party/CenterNet2/projects/CenterNet2/')

from centernet.config import add_centernet_config
from models.grit_src.grit.config import add_grit_config

from models.grit_src.grit.predictor import VisualizationDemo
from utils.util import resize_long_edge_cv2

# constants
WINDOW_NAME = "GRiT"


def dense_pred_to_caption(predictions):
    boxes = predictions["instances"].pred_boxes if predictions["instances"].has("pred_boxes") else None
    object_description = predictions["instances"].pred_object_descriptions.data
    new_caption = ""
    for i in range(len(object_description)):
        new_caption += (object_description[i] + ": " + str(
            [int(a) for a in boxes[i].tensor.cpu().detach().numpy()[0]])) + "; "
    return new_caption


def dense_pred_dict(predictions):
    boxes = predictions["instances"].pred_boxes if predictions["instances"].has("pred_boxes") else None
    scores = predictions["instances"].scores if predictions["instances"].has("scores") else None
    object_description = predictions["instances"].pred_object_descriptions.data

    prediction_list = []
    for i in range(len(object_description)):
        bbox = [round(float(a), 2) for a in boxes[i].tensor.cpu().detach().numpy()[0]]
        score = round(float(scores[i]), 2) if scores is not None else None

        prediction_dict = {
            'bbox': bbox,
            'score': score,
            'description': object_description[i],
        }
        prediction_list.append(prediction_dict)

    return prediction_list


def setup_cfg(args):
    cfg = get_cfg()
    if args["cpu"]:
        cfg.MODEL.DEVICE = "cpu"
    add_centernet_config(cfg)
    add_grit_config(cfg)
    cfg.merge_from_file(args["config_file"])
    cfg.merge_from_list(args["opts"])
    # Set score_threshold for builtin models
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args["confidence_threshold"]
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args["confidence_threshold"]
    if args["test_task"]:
        cfg.MODEL.TEST_TASK = args["test_task"]
    cfg.MODEL.BEAM_SIZE = 1
    cfg.MODEL.ROI_HEADS.SOFT_NMS_ENABLED = False
    cfg.USE_ACT_CHECKPOINT = False
    cfg.freeze()
    return cfg


def get_parser(device):
    arg_dict = {'config_file': "blip_grit_tags/models/grit_src/configs/GRiT_B_DenseCap_ObjectDet.yaml", 'cpu': False,
                'confidence_threshold': 0.7, 'test_task': 'DenseCap',
                'opts': ["MODEL.WEIGHTS", "grit_b_densecap_objectdet.pth"]}
    if device == "cpu":
        arg_dict["cpu"] = True
    return arg_dict


def image_caption_api(image_src, device):
    args2 = get_parser(device)
    cfg = setup_cfg(args2)
    demo = VisualizationDemo(cfg)
    if image_src:
        img = read_image(image_src, format="BGR")
        img = resize_long_edge_cv2(img, 384)
        predictions, visualized_output = demo.run_on_image(img)
        new_caption = dense_pred_to_caption(predictions)
    return new_caption


def image_caption_dict(image_src, device):
    args2 = get_parser(device)
    cfg = setup_cfg(args2)
    demo = VisualizationDemo(cfg)
    if image_src:
        img = read_image(image_src, format="BGR")
        predictions, visualized_output = demo.run_on_image(img)
        dense_pred = dense_pred_dict(predictions)
    return dense_pred


def setup_grit(device):
    args2 = get_parser(device)
    cfg = setup_cfg(args2)
    demo = VisualizationDemo(cfg)
    return demo
