from typing import Literal, Tuple, List

import torch
import torch.nn.functional as F
from mmdet.structures import SampleList
from mmengine import MMLogger
from mmengine.model import BaseModule
from mmdet.registry import MODELS

from ext.sam import MaskDecoder
from ext.meta.sam_meta import meta_dict, checkpoint_dict
from utils.load_checkpoint import load_checkpoint_with_prefix


@MODELS.register_module()
class SAMMaskDecoder(BaseModule):

    def __init__(
            self,
            model_name: Literal['vit_h', 'vit_l', 'vit_b'] = 'vit_h',
            fix: bool = True,
            init_cfg=None,
    ):
        assert init_cfg is not None and \
               init_cfg['type'] in ['sam_pretrain', 'Pretrained'], f"{init_cfg['type']} is not supported."
        pretrained = init_cfg['checkpoint']
        super().__init__(init_cfg=None)
        self.init_cfg = init_cfg
        self.logger = MMLogger.get_current_instance()

        mask_decoder = MaskDecoder(
            num_multimask_outputs=3,
            transformer_dim=meta_dict[model_name]['prompt_embed_dim'],
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        )
        if self.init_cfg['type'] == 'sam_pretrain':
            checkpoint_path = checkpoint_dict[pretrained]
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix='mask_decoder')
            mask_decoder.load_state_dict(state_dict, strict=True)

        self.mask_decoder = mask_decoder
        if self.init_cfg['type'] == 'Pretrained':
            checkpoint_path = pretrained
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix=self.init_cfg['prefix'])
            self.load_state_dict(state_dict, strict=True)

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
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predicts masks. See 'forward' for more details."""
        num_instances = int(sparse_prompt_embeddings.shape[0])
        # Concatenate output tokens
        output_tokens = torch.cat([self.mask_decoder.iou_token.weight, self.mask_decoder.mask_tokens.weight], dim=0)
        output_tokens = output_tokens.unsqueeze(0).expand(num_instances, -1, -1)
        queries = torch.cat((output_tokens, sparse_prompt_embeddings), dim=1)

        # image_embeddings = torch.repeat_interleave(image_embeddings, num_instances, dim=0)
        image_embeddings = image_embeddings + dense_prompt_embeddings
        pos_img = torch.repeat_interleave(image_pe, num_instances, dim=0)
        b, c, h, w = image_embeddings.shape

        # Run the transformer
        queries, mask_feats = self.mask_decoder.transformer(image_embeddings, pos_img, queries)
        iou_query = queries[:, 0, :]
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

        # Generate mask quality predictions
        iou_pred = self.mask_decoder.iou_prediction_head(iou_query)

        return masks, iou_pred, None

    def forward(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            multi_mask_output: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_prompts = len(sparse_prompt_embeddings)
        image_embeddings = torch.repeat_interleave(image_embeddings, num_prompts, dim=0)
        masks, iou_pred, cls_pred = self.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_prompt_embeddings,
            dense_prompt_embeddings=dense_prompt_embeddings,
        )

        # Select the correct mask or masks for output
        if multi_mask_output:
            mask_slice = slice(1, None)
        else:
            mask_slice = slice(0, 1)
        masks = masks[:, mask_slice, :, :]
        iou_pred = iou_pred[:, mask_slice]

        # Prepare output
        return masks, iou_pred, cls_pred

    def forward_train(
            self,
            image_embeddings: torch.Tensor,
            image_pe: torch.Tensor,
            sparse_prompt_embeddings: torch.Tensor,
            dense_prompt_embeddings: torch.Tensor,
            batch_ind_list: List[int],
            data_samples: SampleList,
    ):
        raise NotImplementedError
