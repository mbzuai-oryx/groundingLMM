from app.models.last_layer import LastLayerNeck
from app.models.model import SAMSegmentor
from app.models.sam_backbone import SAMBackbone
from app.models.sam_mask_decoder import SAMMaskDecoder
from app.models.sam_pe import SAMPromptEncoder

model = dict(
    type=SAMSegmentor,
    data_preprocessor=None,
    backbone=dict(
        type=SAMBackbone,
        model_name='vit_h',
        fix=True,
        init_cfg=dict(
            type='sam_pretrain',
            checkpoint='vit_h'
        )
    ),
    neck=dict(type=LastLayerNeck),
    prompt_encoder=dict(
        type=SAMPromptEncoder,
        model_name='vit_h',
        fix=True,
        init_cfg=dict(
            type='sam_pretrain',
            checkpoint='vit_h'
        )
    ),
    mask_decoder=dict(
        type=SAMMaskDecoder,
        model_name='vit_h',
        fix=True,
        init_cfg=dict(
            type='sam_pretrain',
            checkpoint='vit_h'
        )
    )
)