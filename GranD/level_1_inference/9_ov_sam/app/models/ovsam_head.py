import copy
import os
from typing import Literal, Tuple, List, Optional

import torch
from mmcv.cnn import ConvModule
from mmdet.structures.bbox import bbox2roi
from mmdet.structures.mask import mask2bbox
from torch import nn
import torch.nn.functional as F
from mmengine import MMLogger
from mmengine.model import BaseModule
from mmdet.registry import MODELS

from ext.sam import MaskDecoder
from ext.sam.mask_decoder import MLP as SAMMLP
from ext.meta.sam_meta import meta_dict, checkpoint_dict
from utils.load_checkpoint import load_checkpoint_with_prefix


@MODELS.register_module()
class OVSAMHead(BaseModule):

    def __init__(
            self,
            model_name: Literal['vit_h', 'vit_l', 'vit_b'] = 'vit_h',
            with_label_token: bool = False,
            ov_classifier_name: Optional[str] = None,
            logit: Optional[float] = None,
            roi_extractor=None,
            fix: bool = True,
            init_cfg=None,
            cur_mask=1,
            roi_extractor_single=None,
            load_roi_conv=None,
            gen_box=False,
    ):
        assert init_cfg is not None and \
               init_cfg['type'] in ['sam_pretrain', 'Pretrained'], f"{init_cfg['type']} is not supported."
        pretrained = init_cfg['checkpoint']
        super().__init__(init_cfg=None)
        self.init_cfg = init_cfg
        self.logger = MMLogger.get_current_instance()
        if roi_extractor_single is not None:
            self.roi_extractor_single = MODELS.build(roi_extractor_single)
            self.roi_merge_proj = nn.Linear(768 * 2, 768)
        else:
            self.roi_extractor_single = None
            self.roi_merge_proj = None

        mask_decoder = MaskDecoder(
            num_multimask_outputs=cur_mask - 1,
            transformer_dim=meta_dict[model_name]['prompt_embed_dim'],
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            with_iou=False
        )

        if self.init_cfg['type'] == 'sam_pretrain':
            raise NotImplementedError

        self.mask_decoder = mask_decoder

        self.with_label_token = with_label_token

        if self.with_label_token:
            ov_path = os.path.join(os.path.expanduser('./models/'), f"{ov_classifier_name}.pth")
            cls_embed = torch.load(ov_path)
            cls_embed_norm = cls_embed.norm(p=2, dim=-1)
            assert torch.allclose(cls_embed_norm, torch.ones_like(cls_embed_norm))

            _dim = cls_embed.size(2)
            _prototypes = cls_embed.size(1)
            back_token = torch.zeros(1, _dim, dtype=torch.float32, device='cpu')
            cls_embed = torch.cat([
                cls_embed, back_token.repeat(_prototypes, 1)[None]
            ], dim=0)
            self.register_buffer('cls_embed', cls_embed.permute(2, 0, 1).contiguous(), persistent=False)

            if logit is None:
                logit_scale = torch.tensor(4.6052, dtype=torch.float32)
            else:
                logit_scale = torch.tensor(logit, dtype=torch.float32)
            self.register_buffer('logit_scale', logit_scale, persistent=False)

            transformer_dim = self.mask_decoder.mask_tokens.weight.shape[1]
            self.label_token = nn.Embedding(1, transformer_dim)
            self.label_mlp = SAMMLP(transformer_dim, transformer_dim, _dim, 3)

        self.gen_box = gen_box

        if roi_extractor is not None:
            self.roi = MODELS.build(roi_extractor)
            self.roi_conv = nn.Sequential(*[
                ConvModule(in_channels=self.roi.out_channels, out_channels=_dim, kernel_size=1, bias=False)
            ])
        else:
            self.roi = None

        if self.init_cfg['type'] == 'Pretrained':
            checkpoint_path = pretrained
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix=self.init_cfg['prefix'])
            self.load_state_dict(state_dict, strict=True)
        if roi_extractor is not None and load_roi_conv is not None:
            checkpoint_path = load_roi_conv['checkpoint']
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix=load_roi_conv['prefix'])
            self.roi_conv.load_state_dict(state_dict, strict=True)

        self.fix = fix

        if self.fix:
            self.train(mode=False)
            for name, param in self.named_parameters():
                param.requires_grad = False

    def init_weights(self):
        self.logger.info(f"Init Config for {self.__class__.__name__}")
        self.logger.info(self.init_cfg)

    def forward_logit(self, cls_embd):
        cls_pred = torch.einsum('bnc,ckp->bnkp', F.normalize(cls_embd, dim=-1), self.cls_embed)
        cls_pred = cls_pred.max(-1).values
        cls_pred = self.logit_scale.exp() * cls_pred
        return cls_pred

    def predict_masks(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            fpn_feats: List[torch.Tensor],
            roi_list: Optional[List[torch.Tensor]],
            backbone_feature: torch.Tensor,
            backbone=None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        num_instances = int(sparse_prompt_embeddings.size(0))
        # Concatenate output tokens
        output_tokens = torch.cat([
            self.label_token.weight,
            self.mask_decoder.mask_tokens.weight], dim=0
        )
        output_tokens = output_tokens.unsqueeze(0).expand(num_instances, -1, -1)
        queries = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # image_embeddings = torch.repeat_interleave(image_embeddings, num_instances, dim=0)
        image_embeddings = image_embeddings + dense_prompt_embeddings
        pos_img = torch.repeat_interleave(image_pe, num_instances, dim=0)
        b, c, h, w = image_embeddings.shape

        # Run the transformer
        queries, mask_feats = self.mask_decoder.transformer(image_embeddings, pos_img, queries)
        label_query = queries[:, 0, :]
        mask_embeds = queries[:, 1:(1 + self.mask_decoder.num_mask_tokens), :]

        # Upscale mask embeddings and predict masks using the mask tokens
        mask_feats = mask_feats.transpose(1, 2).view(b, c, h, w)
        mask_feats = self.mask_decoder.output_upscaling(mask_feats)
        mask_queries_list: List[torch.Tensor] = []
        for i in range(self.mask_decoder.num_mask_tokens):
            mask_queries_list.append(self.mask_decoder.output_hypernetworks_mlps[i](mask_embeds[:, i, :]))
        mask_queries = torch.stack(mask_queries_list, dim=1)
        b, c, h, w = mask_feats.shape
        masks = (mask_queries @ mask_feats.view(b, c, h * w)).view(b, -1, h, w)

        # Generate class labels
        if self.with_label_token:
            cls_embed_list = []
            assert self.mask_decoder.num_mask_tokens == 1
            for i in range(self.mask_decoder.num_mask_tokens):
                cls_embed_list.append(self.label_mlp(label_query))
            cls_embed = torch.stack(cls_embed_list, dim=1)
            if self.gen_box:
                bboxes = mask2bbox(masks.sigmoid()[:, 0] > 0.5) * 4
                roi_list = bbox2roi([bboxes])
            roi_feats = self.roi(fpn_feats, roi_list)
            roi_feats = self.roi_conv(roi_feats)
            roi_feats = roi_feats.mean(dim=-1).mean(dim=-1)
            if self.roi_extractor_single:
                roi_feats_clip = self.roi_extractor_single(
                    backbone.get_clip_feature(backbone_feature[-1:]), roi_list
                )
                roi_feats_clip = backbone.forward_feat(roi_feats_clip)
                roi_feats = self.roi_merge_proj(torch.cat([roi_feats, roi_feats_clip], dim=-1))
            roi_feats = roi_feats[:, None] + 0 * cls_embed
            cls_pred = self.forward_logit(roi_feats)
        else:
            cls_pred = None
        return masks, None, cls_pred

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multi_mask_output: bool,
            data_samples=None,
            fpn_feats=None,
            backbone_feats=None,
            backbone=None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        num_prompts = len(sparse_prompt_embeddings)
        image_embeddings = torch.repeat_interleave(image_embeddings, num_prompts, dim=0)

        masks, _, cls_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
            fpn_feats=fpn_feats,
            roi_list=None,
            backbone_feature=backbone_feats,
            backbone=backbone,
        )

        # Select the correct mask or masks for output
        if multi_mask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]

        # Prepare output
        return masks, None, cls_pred
