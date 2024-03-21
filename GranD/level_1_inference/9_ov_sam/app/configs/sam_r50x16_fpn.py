from mmcv.ops import RoIAlign
from mmdet.models import FPN, SingleRoIExtractor

from app.models.model import SAMSegmentor
from app.models.openclip_backbone import OpenCLIPBackbone
from app.models.ovsam_head import OVSAMHead
from app.models.sam_pe import SAMPromptEncoder
from app.models.transformer_neck import MultiLayerTransformerNeck

model = dict(
    type=SAMSegmentor,
    data_preprocessor=None,
    enable_backbone=True,
    backbone=dict(
        type=OpenCLIPBackbone,
        model_name='RN50x16',
        fix=True,
        init_cfg=dict(
            type='clip_pretrain',
            checkpoint='openai'
        )
    ),
    neck=dict(
        type=MultiLayerTransformerNeck,
        input_size=(1024, 1024),
        in_channels=[384, 768, 1536, 3072],
        strides=[4, 8, 16, 32],
        layer_ids=(0, 1, 2, 3),
        embed_channels=1280,
        out_channels=256,
        fix=True,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./sam2clip_vith_rn50.pth',
            prefix='neck_student',
        )
    ),
    prompt_encoder=dict(
        type=SAMPromptEncoder,
        model_name='vit_h',
        fix=True,
        init_cfg=dict(
            type='sam_pretrain',
            checkpoint='vit_h'
        )
    ),
    fpn_neck=dict(
        type=FPN,
        in_channels=[384, 768, 1536, 3072],
        out_channels=256,
        num_outs=4,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./models/R50x16_fpn_lvis_norare_v3det.pth',
            prefix='fpn_neck',
        ),
    ),
    mask_decoder=dict(
        type=OVSAMHead,
        model_name='vit_h',
        with_label_token=True,
        gen_box=True,
        ov_classifier_name='RN50x16_LVISV1Dataset',
        roi_extractor=dict(
            type=SingleRoIExtractor,
            roi_layer=dict(type=RoIAlign, output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]
        ),
        fix=False,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='./models/ovsam_R50x16_lvisnorare.pth',
            prefix='mask_decoder',
        ),
        load_roi_conv=dict(
            checkpoint='./models/R50x16_fpn_lvis_norare_v3det.pth',
            prefix='roi_conv',
        )
    )
)
