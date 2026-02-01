import torch
import torch.nn.functional as F
from torch import nn
import numpy.random as npr
from .inference import make_atss_postprocessor
from .loss import make_atss_loss_evaluator
from .anchor_generator import make_anchor_generator_complex

from maskrcnn_benchmark.structures.boxlist_ops import cat_boxlist, boxlist_iou
from maskrcnn_benchmark.layers import Scale, DYReLU, ModulatedDeformConv
from maskrcnn_benchmark.modeling.backbone.fbnet import *
from maskrcnn_benchmark.engine.inference import create_positive_map_label_to_token_from_positive_map
from maskrcnn_benchmark.modeling.make_layers import make_fc
from maskrcnn_benchmark.data import get_dataset_statistics
from ..utils import permute_and_flatten

from maskrcnn_benchmark.utils.fuse_helper import FeatureResizer, func_attention, _make_mlp, _make_conv, _make_coord, AttentionT2I, BiAttentionBlockForCheckpoint, BertLMPredictionHead
from transformers.models.bert.modeling_bert import BertConfig, BertAttention, BertIntermediate, BertOutput, \
    BertPreTrainedModel
from transformers.modeling_utils import apply_chunking_to_forward
import torch.utils.checkpoint as checkpoint
import numpy as np
import math

from maskrcnn_benchmark.modeling.language_backbone.clip_model import QuickGELU, LayerNorm, DropPath
from timm.models.layers import DropPath, trunc_normal_
from maskrcnn_benchmark.layers import SigmoidFocalLoss, IOULoss, TokenSigmoidFocalLoss
from maskrcnn_benchmark.modeling.utils import cat

from maskrcnn_benchmark.modeling.rpn.generate_text import Generate_with_T5
import clip

VG150_REL_CATEGORIES = ['__background__', 'above', 'across', 'against', 'along', 'and', 'at', 'attached to', 'behind', 'belonging to', 'between', 'carrying', 'covered in', 'covering', 'eating', 'flying in', 'for', 'from', 'growing on', 'hanging from', 'has', 'holding', 'in', 'in front of', 'laying on', 'looking at', 'lying on', 'made of', 'mounted on', 'near', 'of', 'on', 'on back of', 'over', 'painted on', 'parked on', 'part of', 'playing', 'riding', 'says', 'sitting on', 'standing on', 'to', 'under', 'using', 'walking in', 'walking on', 'watching', 'wearing', 'wears', 'with']

class h_sigmoid(nn.Module):
    def __init__(self, inplace=True, h_max=1):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
        self.h_max = h_max

    def forward(self, x):
        return self.relu(x + 3) * self.h_max / 6


class BoxCoder(object):

    def __init__(self, cfg):
        self.cfg = cfg

    def encode(self, gt_boxes, anchors):
        TO_REMOVE = 1  # TODO remove
        ex_widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        ex_heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ex_ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
        ex_ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

        gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + TO_REMOVE
        gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + TO_REMOVE
        gt_ctr_x = (gt_boxes[:, 2] + gt_boxes[:, 0]) / 2
        gt_ctr_y = (gt_boxes[:, 3] + gt_boxes[:, 1]) / 2

        wx, wy, ww, wh = (10., 10., 5., 5.)
        targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
        targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
        targets_dw = ww * torch.log(gt_widths / ex_widths)
        targets_dh = wh * torch.log(gt_heights / ex_heights)
        targets = torch.stack((targets_dx, targets_dy, targets_dw, targets_dh), dim=1)

        return targets

    def decode(self, preds, anchors):
        anchors = anchors.to(preds.dtype)

        TO_REMOVE = 1  # TODO remove
        widths = anchors[:, 2] - anchors[:, 0] + TO_REMOVE
        heights = anchors[:, 3] - anchors[:, 1] + TO_REMOVE
        ctr_x = (anchors[:, 2] + anchors[:, 0]) / 2
        ctr_y = (anchors[:, 3] + anchors[:, 1]) / 2

        wx, wy, ww, wh = (10., 10., 5., 5.)
        dx = preds[:, 0::4] / wx
        dy = preds[:, 1::4] / wy
        dw = preds[:, 2::4] / ww
        dh = preds[:, 3::4] / wh

        # Prevent sending too large values into torch.exp()
        dw = torch.clamp(dw, max=math.log(1000. / 16))
        dh = torch.clamp(dh, max=math.log(1000. / 16))

        pred_ctr_x = dx * widths[:, None] + ctr_x[:, None]
        pred_ctr_y = dy * heights[:, None] + ctr_y[:, None]
        pred_w = torch.exp(dw) * widths[:, None]
        pred_h = torch.exp(dh) * heights[:, None]

        pred_boxes = torch.zeros_like(preds)
        pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * (pred_w - 1)
        pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * (pred_h - 1)
        pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * (pred_w - 1)
        pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * (pred_h - 1)

        return pred_boxes


class Conv3x3Norm(torch.nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 groups=1,
                 deformable=False,
                 bn_type=None):
        super(Conv3x3Norm, self).__init__()

        if deformable:
            self.conv = ModulatedDeformConv(in_channels, out_channels, kernel_size=3, stride=stride, padding=1,
                                            groups=groups)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, groups=groups)

        if isinstance(bn_type, (list, tuple)):
            assert len(bn_type) == 2
            assert bn_type[0] == "gn"
            gn_group = bn_type[1]
            bn_type = bn_type[0]

        if bn_type == "bn":
            bn_op = nn.BatchNorm2d(out_channels)
        elif bn_type == "sbn":
            bn_op = nn.SyncBatchNorm(out_channels)
        elif bn_type == "nsbn":
            bn_op = NaiveSyncBatchNorm2d(out_channels)
        elif bn_type == "gn":
            bn_op = nn.GroupNorm(num_groups=gn_group, num_channels=out_channels)
        elif bn_type == "af":
            bn_op = FrozenBatchNorm2d(out_channels)
        if bn_type is not None:
            self.bn = bn_op
        else:
            self.bn = None

    def forward(self, input, **kwargs):
        x = self.conv(input, **kwargs)
        if self.bn:
            x = self.bn(x)
        return x


class DyConv(torch.nn.Module):
    def __init__(self,
                 in_channels=256,
                 out_channels=256,
                 conv_func=nn.Conv2d,
                 use_dyfuse=True,
                 use_dyrelu=False,
                 use_deform=False
                 ):
        super(DyConv, self).__init__()

        self.DyConv = nn.ModuleList()
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 1))
        self.DyConv.append(conv_func(in_channels, out_channels, 2))

        if use_dyfuse:
            self.AttnConv = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, 1, kernel_size=1),
                nn.ReLU(inplace=True))
            self.h_sigmoid = h_sigmoid()
        else:
            self.AttnConv = None

        if use_dyrelu:
            self.relu = DYReLU(in_channels, out_channels)
        else:
            self.relu = nn.ReLU()

        if use_deform:
            self.offset = nn.Conv2d(in_channels, 27, kernel_size=3, stride=1, padding=1)
        else:
            self.offset = None

        self.init_weights()

    def init_weights(self):
        for m in self.DyConv.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        if self.AttnConv is not None:
            for m in self.AttnConv.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight.data, 0, 0.01)
                    if m.bias is not None:
                        m.bias.data.zero_()

    def forward(self, inputs,single_feat=False):
        visual_feats = inputs["visual"]
        language_dict_features = inputs["lang"]
        single_feat=inputs["single_feat"]
        masks=inputs["masks"]
        eot_indices=inputs["eot_indices"]
        use_prompt=inputs["use_prompt"]

        next_x = []
        for level, feature in enumerate(visual_feats):

            conv_args = dict()
            if self.offset is not None:
                offset_mask = self.offset(feature)
                offset = offset_mask[:, :18, :, :]
                mask = offset_mask[:, 18:, :, :].sigmoid()
                conv_args = dict(offset=offset, mask=mask)

            temp_fea = [self.DyConv[1](feature, **conv_args)]

            if level > 0:
                temp_fea.append(self.DyConv[2](visual_feats[level - 1], **conv_args))
            if level < len(visual_feats) - 1:
                temp_fea.append(F.upsample_bilinear(self.DyConv[0](visual_feats[level + 1], **conv_args),
                                                    size=[feature.size(2), feature.size(3)]))
            mean_fea = torch.mean(torch.stack(temp_fea), dim=0, keepdim=False)

            if self.AttnConv is not None:
                attn_fea = []
                res_fea = []
                for fea in temp_fea:
                    res_fea.append(fea)
                    attn_fea.append(self.AttnConv(fea))

                res_fea = torch.stack(res_fea)
                spa_pyr_attn = self.h_sigmoid(torch.stack(attn_fea))

                mean_fea = torch.mean(res_fea * spa_pyr_attn, dim=0, keepdim=False)

            next_x.append(mean_fea)

        next_x = [self.relu(item) for item in next_x]

        features_dict = {"visual": next_x,
                         "lang": language_dict_features,
                         "masks": masks,
                         "use_prompt": use_prompt,
                         "single_feat": single_feat,
                         "eot_indices": eot_indices }


        return features_dict


class BertEncoderLayer(BertPreTrainedModel):
    def __init__(self, config,  clamp_min_for_underflow = False, clamp_max_for_overflow = False):
        super().__init__(config)
        self.config = config

        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1

        from maskrcnn_benchmark.modeling.rpn.modeling_bert import BertAttention, BertIntermediate, BertOutput

        self.attention = BertAttention(config,  clamp_min_for_underflow, clamp_max_for_overflow)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, inputs):
        language_dict_features = inputs["lang"]
        single_feat=inputs["single_feat"]
        masks=inputs["masks"]
        eot_indices=inputs["eot_indices"]
        use_prompt=inputs["use_prompt"]


        if single_feat or use_prompt:
            hidden_states = language_dict_features#4 310 768
        else:
            hidden_states = language_dict_features["hidden"]


        if single_feat:
            attention_mask = masks.unsqueeze(0)
        else:
            if use_prompt:
                attention_mask = masks
            else:
                attention_mask = language_dict_features["masks"]  # 4 310

        device = hidden_states.device
        input_shape = hidden_states.size()[:-1]
        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, device)#4 1 1 310

        self_attention_outputs = self.attention(
            hidden_states,#####25 77 768
            extended_attention_mask,
            None,
            output_attentions=False,
            past_key_value=None,
        )
        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights
        layer_output = apply_chunking_to_forward(
            self.feed_forward_chunk, self.chunk_size_feed_forward, self.seq_len_dim, attention_output
        )
        outputs = (layer_output,) + outputs
        hidden_states = outputs[0]

        if single_feat:
            language_dict_features = hidden_states
        else:
            if use_prompt:#################
                #language_dict_features = hidden_states[torch.arange(hidden_states.shape[0]), eot_indices]
                language_dict_features = hidden_states
            else:
                language_dict_features["hidden"] = hidden_states

        features_dict = {"visual": inputs["visual"],
                         "lang": language_dict_features,
                         "masks": masks,
                         "use_prompt": use_prompt,
                         "single_feat": single_feat,
                         "eot_indices": eot_indices }

        return features_dict

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output


class CLIPTransformerLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d_model = self.config.MODEL.CLIP.WIDTH
        n_head = self.config.MODEL.CLIP.HEADS
        drop_path = self.config.MODEL.CLIP.DROP_PATH
        self.context_length = self.config.MODEL.CLIP.CONTEXT_LENGTH
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)

    def attention(self, x: torch.Tensor, key_padding_mask: torch.Tensor = None):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask, key_padding_mask=key_padding_mask)[0]

    def forward(self, inputs):
        language_dict_features = inputs["lang"]
        x = language_dict_features["hidden"]
        mask = language_dict_features["masks"]
        # get extended attention mask for nn.MultiHeadAttention
        key_padding_mask = (1.0 - mask).to(torch.bool)

        x = x.permute(1, 0, 2)
        x = x + self.drop_path(self.attention(self.ln_1(x), key_padding_mask=key_padding_mask))
        x = x + self.drop_path(self.mlp(self.ln_2(x)))
        x = x.permute(1, 0, 2)

        language_dict_features["hidden"] = x
        features_dict = {"visual": inputs["visual"],
                         "lang": language_dict_features
                         }
        return features_dict


class DummyLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs):
        return inputs


class VLFuse(torch.nn.Module):
    """
    Early Fusion Module
    """

    def __init__(self, cfg):
        super(VLFuse, self).__init__()
        self.init_configs(cfg)
        self.cfg = cfg

        self.use_checkpoint = False
        if hasattr(cfg.MODEL.DYHEAD, 'USE_CHECKPOINT'):
            self.use_checkpoint = cfg.MODEL.DYHEAD.USE_CHECKPOINT
            self.dummy_tensor = torch.ones(1, dtype=torch.float32, requires_grad=True)

        # early fusion module
        print("EARLY FUSION ON, USING {}".format(cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE))
        if cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "MHA-S":
            # single-direction (text->image)
            # text -> image
            self.t2i_attn = AttentionT2I(q_dim=self.joint_embedding_size,
                                           k_dim=self.lang_dim,
                                           embed_dim=self.embed_dim,
                                           num_heads=self.n_head,
                                           hidden_dim=self.t2i_hidden_dim,
                                           dropout=0.1,
                                           drop_path=.0,
                                           init_values=1.0 / cfg.MODEL.DYHEAD.NUM_CONVS,
                                           mode="t2i",
                                           use_layer_scale=cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_LAYER_SCALE,
                                           clamp_min_for_underflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MIN_FOR_UNDERFLOW,
                                           clamp_max_for_overflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_MAX_FOR_OVERFLOW
                                           )

        elif cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "MHA-B":
            # bi-direction (text->image, image->text)
            self.b_attn = BiAttentionBlockForCheckpoint(v_dim=self.joint_embedding_size,
                        l_dim=self.lang_dim,
                        embed_dim=self.embed_dim,
                        num_heads=self.n_head,
                        hidden_dim=self.i2t_hidden_dim,
                        dropout=0.1,
                        drop_path=.0,
                        init_values=1.0 / cfg.MODEL.DYHEAD.NUM_CONVS,
                        cfg=cfg
                        )
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL and self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT:
                self.shrink_lang = FeatureResizer(self.lang_dim * 5,
                                self.lang_dim, 0.1)


        elif cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "SCAN":
            # single-direction (text->image)
            self.mapping_lang = _make_mlp(self.lang_dim,
                                          self.joint_embedding_size,
                                          self.joint_embedding_dropout)
            self.joint_fusion = nn.ModuleList([_make_conv(self.joint_inp_dim, self.joint_out_dim, 1) \
                                               for _ in range(5)])

        elif cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "FILM":
            # single-direction (text->image)
            self.mapping_lang = _make_mlp(self.lang_dim,
                                          self.joint_embedding_size,
                                          self.joint_embedding_dropout)
            self.gamma = nn.ModuleList(nn.Linear(self.joint_embedding_size, self.joint_inp_dim) for _ in range(5))
            self.beta = nn.ModuleList(nn.Linear(self.joint_embedding_size, self.joint_inp_dim) for _ in range(5))

            self.joint_fusion = nn.ModuleList([_make_conv(self.joint_inp_dim, self.joint_out_dim, 1) \
                                               for _ in range(5)])

        else:
            print("NO FUSION INVOLVED.")

    def init_configs(self, cfg):
        # common params
        self.lang_model = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        self.joint_embedding_size = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_SIZE
        self.joint_embedding_dropout = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_DROPOUT
        self.joint_mlp_layers = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_MLP_LAYERS

        self.max_query_len = cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN
        self.n_layers = cfg.MODEL.LANGUAGE_BACKBONE.N_LAYERS
        self.coord_dim = 8
        self.joint_inp_dim = self.coord_dim + self.joint_embedding_size
        self.joint_out_dim = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_OUT_SIZE

        # mha params
        self.n_head = 8
        self.embed_dim = 2048
        self.t2i_hidden_dim = 1024  # 256 * 4
        self.i2t_hidden_dim = 3072  # 768 * 4

        if self.lang_model in ["bert-base-uncased", "roberta-base", "clip"]:
            self.lang_dim = cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM
        else:
            self.lang_dim = 1024

    def forward(self, x,single_feat=False):
        visual_features = x["visual"]
        language_dict_features = x["lang"]
        single_feat=x["single_feat"]
        masks=x["masks"]
        eot_indices=x["eot_indices"]
        use_prompt=x["use_prompt"]


        #print("visual_features",type(visual_features))
        # if not isinstance(visual_features, tuple):
        if single_feat:
             masks = x["masks"]
        else:
            batch_size = visual_features[0].shape[0]
            device = visual_features[0].device

        fused_visual_features = None
        fused_language_dict_features = None

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "MHA-S":
            language_feature = language_dict_features['hidden']
            mask = language_dict_features['masks']
            # text -> image
            if self.use_checkpoint:
                q0, q1, q2, q3, q4 = checkpoint.checkpoint(
                    self.t2i_attn,
                    visual_features[0], visual_features[1],
                    visual_features[2], visual_features[3],
                    visual_features[4],
                    language_feature, language_feature,
                    mask,
                    self.dummy_tensor
                )
            else:
                q0, q1, q2, q3, q4 = self.t2i_attn(
                    visual_features[0], visual_features[1],
                    visual_features[2], visual_features[3],
                    visual_features[4],
                    language_feature, language_feature,
                    attention_mask=mask
                )

            fused_visual_features = [q0, q1, q2, q3, q4]
            fused_language_dict_features = language_dict_features

        elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "MHA-B":####################
            if self.use_checkpoint:
                if single_feat:
                    q0, q1, q2, q3, q4, l0, l1, l2, l3, l4 = checkpoint.checkpoint(self.b_attn,
                        visual_features, None,
                        None, None,
                        None,
                        language_dict_features,
                        masks,
                        self.dummy_tensor,
                        single_feat
                    )
                else:

                    q0, q1, q2, q3, q4, l0, l1, l2, l3, l4 = checkpoint.checkpoint(self.b_attn,
                                                                                   visual_features[0],
                                                                                   visual_features[1],
                                                                                   visual_features[2],
                                                                                   visual_features[3],
                                                                                   visual_features[4],
                                                                                   language_dict_features['hidden'],
                                                                                   language_dict_features['masks'],
                                                                                   self.dummy_tensor,
                                                                                   )
            else:
                if single_feat:
                    q0, q1, q2, q3, q4, l0, l1, l2, l3, l4 = self.b_attn(
                        visual_features, None,
                        None, None,
                        None,
                        language_dict_features,
                        masks,
                        self.dummy_tensor,
                        single_feat
                    )
                else:
                    q0, q1, q2, q3, q4, l0, l1, l2, l3, l4 = self.b_attn(
                        visual_features[0], visual_features[1],
                        visual_features[2], visual_features[3],
                        visual_features[4],
                        language_dict_features['hidden'],
                        language_dict_features['masks'],
                        self.dummy_tensor
                    )

            fused_visual_features = [q0, q1, q2, q3, q4]
            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SEPARATE_BIDIRECTIONAL and self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DO_LANG_PROJ_OUTSIDE_CHECKPOINT:
                language_features = self.shrink_lang(torch.cat([l0, l1, l2, l3, l4], dim = -1))
            else:#################
                language_features = l0
            if single_feat:
                language_features = l0
            else:
                language_dict_features['hidden'] = language_features
                fused_language_dict_features = language_dict_features

        elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "SCAN":
            # text -> image
            language_feature = language_dict_features['aggregate']
            language_feature = self.mapping_lang(language_feature)
            visu_feat = []
            for ii, feat in enumerate(visual_features):
                attn_feat = func_attention(feat, language_feature, smooth=1, raw_feature_norm="softmax")
                visu_feat.append(attn_feat)

            fused_visual_features = [fusion(feat) for feat, fusion in zip(visu_feat, self.joint_fusion)]
            fused_language_dict_features = language_dict_features

        elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TYPE == "FILM":
            # text -> image
            # relative position embedding
            coord_feats = [_make_coord(batch_size, x.shape[2], x.shape[3]) for x in visual_features]
            # I only use a global representation of language
            # you can also use more complex modeling using word-level representations
            # Usage: lang_feat = lang_feat['words'] shape [seq_len, dim]
            language_feature = language_dict_features['aggregate']
            language_feature = self.mapping_lang(language_feature)

            # attention mechanism for fusion
            gamma = [F.tanh(gamma(language_feature)) for gamma in self.gamma]
            beta = [F.tanh(beta(language_feature)) for beta in self.beta]

            visu_feat = []
            for ii, feat in enumerate(visual_features):
                coord_feat = coord_feats[ii].to(device)
                feat = torch.cat([feat, coord_feat], dim=1)
                b = beta[ii].view(batch_size, -1, 1, 1).expand_as(feat)
                g = gamma[ii].view(batch_size, -1, 1, 1).expand_as(feat)
                feat = F.relu(g * feat + b)
                visu_feat.append(feat)

            fused_visual_features = [fusion(feat) for feat, fusion in zip(visu_feat, self.joint_fusion)]
            fused_language_dict_features = language_dict_features

        else:
            fused_visual_features = visual_features
            fused_language_dict_features = language_dict_features

        if single_feat:
            features_dict = {"visual": q0,
                         "lang": language_features,
                             "masks": masks,
                             "use_prompt": use_prompt,
                             "single_feat": single_feat,
                             "eot_indices": eot_indices}
        else:
            features_dict = {"visual": fused_visual_features,
                         "lang": fused_language_dict_features,
                             "masks": masks,
                             "use_prompt": use_prompt,
                             "single_feat": single_feat,
                             "eot_indices": eot_indices}


        return features_dict


class VLDyHead(torch.nn.Module):
    def __init__(self, cfg):
        super(VLDyHead, self).__init__()
        self.cfg = cfg
        # bert_cfg = BertConfig.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE)
        if cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "bert-base-uncased":
            lang_cfg = BertConfig.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE)
        elif cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "clip":
            lang_cfg = cfg
        else:
            lang_cfg = None
            raise NotImplementedError

        num_classes = cfg.MODEL.DYHEAD.NUM_CLASSES - 1
        num_tokens = cfg.MODEL.LANGUAGE_BACKBONE.MAX_QUERY_LEN
        num_anchors = len(cfg.MODEL.RPN.ASPECT_RATIOS) * cfg.MODEL.RPN.SCALES_PER_OCTAVE
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        channels = cfg.MODEL.DYHEAD.CHANNELS

        if cfg.MODEL.DYHEAD.USE_GN:
            bn_type = ['gn', cfg.MODEL.GROUP_NORM.NUM_GROUPS]
        elif cfg.MODEL.DYHEAD.USE_NSYNCBN:
            bn_type = 'nsbn'
        elif cfg.MODEL.DYHEAD.USE_SYNCBN:
            bn_type = 'sbn'
        else:
            bn_type = None

        use_dyrelu = cfg.MODEL.DYHEAD.USE_DYRELU
        use_dyfuse = cfg.MODEL.DYHEAD.USE_DYFUSE
        use_deform = cfg.MODEL.DYHEAD.USE_DFCONV

        if cfg.MODEL.DYHEAD.CONV_FUNC:
            conv_func = lambda i, o, s: eval(cfg.MODEL.DYHEAD.CONV_FUNC)(i, o, s, bn_type=bn_type)
        else:
            conv_func = lambda i, o, s: Conv3x3Norm(i, o, s, deformable=use_deform, bn_type=bn_type)

        dyhead_tower = []
        for i in range(cfg.MODEL.DYHEAD.NUM_CONVS):#6
            if cfg.MODEL.DYHEAD.FUSE_CONFIG.EARLY_FUSE_ON:#################
                # cross-modality fusion
                dyhead_tower.append(
                    VLFuse(cfg)
                )
                # self language path
                if i < cfg.MODEL.DYHEAD.NUM_CONVS - 1 or cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_FUSED_FEATURES_DOT_PRODUCT:
                    # dyhead_tower.append(
                    #     BertEncoderLayer(
                    #     bert_cfg,
                    #     clamp_min_for_underflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW,
                    #     clamp_max_for_overflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW)
                    # )
                    if cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "bert-base-uncased":###################
                        dyhead_tower.append(
                            BertEncoderLayer(
                                lang_cfg,
                                clamp_min_for_underflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW,
                                clamp_max_for_overflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW)###true
                        )
                    elif cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "clip":
                        dyhead_tower.append(
                            CLIPTransformerLayer(lang_cfg)
                        )
                    else:
                        raise NotImplementedError

                else:###identify
                    dyhead_tower.append(
                        DummyLayer()
                    )

            # self vision path
            dyhead_tower.append(
                DyConv(
                    in_channels if i == 0 else channels,
                    channels,
                    conv_func=conv_func,
                    use_dyrelu=(use_dyrelu and in_channels == channels) if i == 0 else use_dyrelu,
                    use_dyfuse=(use_dyfuse and in_channels == channels) if i == 0 else use_dyfuse,
                    use_deform=(use_deform and in_channels == channels) if i == 0 else use_deform,
                )
            )

        self.add_module('dyhead_tower', nn.Sequential(*dyhead_tower))

        self.cls_logits = nn.Conv2d(channels, num_anchors * num_classes, kernel_size=1)
        self.bbox_pred = nn.Conv2d(channels, num_anchors * 4, kernel_size=1)
        self.centerness = nn.Conv2d(channels, num_anchors * 1, kernel_size=1)

        # initialize the bias for focal loss
        prior_prob = cfg.MODEL.DYHEAD.PRIOR_PROB
        bias_value = -math.log((1 - prior_prob) / prior_prob)

        log_scale = self.cfg.MODEL.DYHEAD.LOG_SCALE

        # soft token head
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            self.token_logits = nn.Conv2d(channels, num_anchors * num_tokens, kernel_size=1)
            # ABLATION
            # self.token_logits = nn.Conv2d(channels, num_anchors * num_tokens, kernel_size=1, bias=False)
            # self.bias = nn.Parameter(torch.zeros(channels), requires_grad=True)
            # self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)

        # contrastive alignment head
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            assert self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS == False
            contrastive_hdim = cfg.MODEL.DYHEAD.FUSE_CONFIG.CONTRASTIVE_HIDDEN_DIM
            self.contrastive_align_projection_image = nn.Conv2d(channels, num_anchors * contrastive_hdim, kernel_size=1)
            self.contrastive_align_projection_text = nn.Linear(channels, contrastive_hdim, bias=True)
            self.log_scale = nn.Parameter(torch.Tensor([log_scale]), requires_grad=True)

        # dot product soft token head
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:##############
            assert self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS == False
            self.dot_product_projection_image = nn.Identity()
            self.dot_product_projection_text = nn.Linear(self.cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM,
                                                         num_anchors * channels, bias=True)
            self.log_scale = nn.Parameter(torch.Tensor([log_scale]), requires_grad=True)
            # DEBUG
            # self.bias = nn.Parameter(torch.zeros(channels), requires_grad=True)
            self.bias_lang = nn.Parameter(torch.zeros(self.cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM), requires_grad=True)
            self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)

        # initialization
        for modules in [self.cls_logits, self.bbox_pred,
                        self.centerness]:
            for l in modules.modules():
                if isinstance(l, nn.Conv2d):
                    torch.nn.init.normal_(l.weight, std=0.01)
                    torch.nn.init.constant_(l.bias, 0)

        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(5)])

        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

        # if use soft token loss
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            for modules in [self.token_logits]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)

            torch.nn.init.constant_(self.token_logits.bias, bias_value)
            # print(torch.norm(self.token_logits.weight))

        # if use contrastive loss
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            for modules in [self.contrastive_align_projection_image]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)

        # if use dot product token loss
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:
            for modules in [self.dot_product_projection_image]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, bias_value)
        
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
            if cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "clip":
                lang_cfg = BertConfig.from_pretrained("bert-base-uncased")
                lang_cfg.hidden_size = cfg.MODEL.CLIP.WIDTH
                lang_cfg.vocab_size = cfg.MODEL.CLIP.VOCAB_SIZE
            self.mlm_head = BertLMPredictionHead(
                lang_cfg
            ) #nn.Linear(hidden_size, config.vocab_size, bias=False)

    def forward(self, x, language_dict_features=None, embedding=None, swint_feature_c4=None,language_dict_features_rel=None,embedding_rel=None,train_ov_relation=False,use_object_fuse=False):
        logits = []
        bbox_reg = []
        centerness = []

        feat_inputs = {"visual": x,
                       "lang": language_dict_features,
                       "masks": None,
                       "use_prompt": False,
                       "single_feat": False,
                       "eot_indices": None}###########图像级visual feat和text（全部object类）【融合】
        if train_ov_relation:
            feat_inputs_rel = {"visual": x,
                                "lang": language_dict_features_rel,
                                "masks": None,
                                "use_prompt": False,
                                "single_feat": False,
                                "eot_indices": None}


        dyhead_tower = self.dyhead_tower(feat_inputs) # through VL-fusion net VLFuse-text(hidden)+viusal   Bert-Encoder-text       DYconv-visual
        if train_ov_relation:
            if use_object_fuse:
                dyhead_tower_rel = self.dyhead_tower(feat_inputs_rel)
            else:
                aaaa=0

        # soft token
        t_logits = None
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            t_logits = []
        
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_FUSED_FEATURES_DOT_PRODUCT:############
            embedding = dyhead_tower["lang"]["hidden"]######编码全部token的embedding 4 310 768
            if train_ov_relation:
                if use_object_fuse:
                    embedding_rel = dyhead_tower_rel["lang"]["hidden"]
                else:
                    embedding_rel = language_dict_features_rel["hidden"]
                    #embedding_rel = None
        
        # MLM loss
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS:
            mlm_logits = self.mlm_head(embedding)
        else:
            mlm_logits = None

        # contrastive
        contrastive_logits = None
        proj_tokens = None
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            contrastive_logits = []
            # follow MDETR's way
            proj_tokens = F.normalize(
                self.contrastive_align_projection_text(embedding), p=2, dim=-1
            )

        # dot product soft token
        dot_product_logits = None
        dot_product_proj_tokens = None
        dot_product_proj_tokens_bias = None
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:###################
            dot_product_logits = []
            # norm
            embedding = F.normalize(embedding, p=2, dim=-1)
            dot_product_proj_tokens = self.dot_product_projection_text(embedding / 2.0)#4 310 256
            if train_ov_relation:
                #embedding_rel = F.normalize(embedding_rel, p=2, dim=-1)
                aaaa=1
            else:
                embedding_rel = None
            # dot_product_proj_tokens_rel = self.dot_product_projection_text(embedding_rel / 2.0)
            # w/o norm
            # dot_product_proj_tokens = self.dot_product_projection_text(embedding / 28.0)

            dot_product_proj_tokens_bias = torch.matmul(embedding, self.bias_lang) + self.bias0######4 310 Wx+b
            #dot_product_proj_tokens_bias_rel = torch.matmul(embedding_rel, self.bias_lang) + self.bias0

        # shallow contrastive (original feature from image & text encoder)
        shallow_img_emb_feats = None
        shallow_text_emb = None
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS \
                or self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS:
            shallow_img_emb_feats = []
            shallow_text_emb = embedding

        # print([v.shape for v in x])
        # shallow contrastive: use the feature from swint backbone
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS:
            for b, feature in enumerate(swint_feature_c4):
                # BF, CF, HF, WF = feat.shape
                # shallow_img_emb = permute_and_flatten(feat, BF, -1, CF, HF, WF)
                shallow_img_emb_feats.append(feature)

        fused_visual_features = None
        if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES or self.cfg.MODEL.DYHEAD.RELATION_CONSISTENCY_ON:#################
            fused_visual_features = []

        # use the feature from FPN
        for l, feature in enumerate(x):
            logits.append(self.cls_logits(dyhead_tower["visual"][l]))#4 150 h w

            bbox_pred = self.scales[l](self.bbox_pred(dyhead_tower["visual"][l]))
            bbox_reg.append(bbox_pred)#4 4 h w

            centerness.append(self.centerness(dyhead_tower["visual"][l]))#4 1 h w

            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
                t_logits.append(self.token_logits(dyhead_tower["visual"][l]))

                # ABLATION
                # b = self.bias.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                # x = dyhead_tower["visual"][l]
                # B, C, H, W = x.shape
                # bias = b.repeat(B, 1, H, W)
                # t_logits.append(self.token_logits(dyhead_tower["visual"][l] + bias) + self.bias0)

            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
                x = dyhead_tower["visual"][l]
                B, _, H, W = x.shape
                C = proj_tokens.shape[2]
                proj_queries = self.contrastive_align_projection_image(dyhead_tower["visual"][l])
                proj_queries = permute_and_flatten(proj_queries, B, -1, C, H, W)
                normalized_img_emb = F.normalize(proj_queries, p=2, dim=-1)
                normalized_text_emb = proj_tokens
                contrastive_logit = (
                        torch.matmul(normalized_img_emb, normalized_text_emb.transpose(-1, -2)) / self.log_scale.exp())
                contrastive_logits.append(contrastive_logit)

            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:##########
                x = dyhead_tower["visual"][l]
                #x_rel = dyhead_tower_rel["visual"][l]

                if self.cfg.MODEL.RPN.RETURN_FUSED_FEATURES:
                    fused_visual_features.append(x)
                B, C, H, W = x.shape

                # add bias (language)
                dot_product_proj_queries = self.dot_product_projection_image(x)#4 256 h w——feature
                dot_product_proj_queries = permute_and_flatten(dot_product_proj_queries, B, -1, C, H, W)#4 h*w 256
                if self.cfg.MODEL.DYHEAD.RELATION_CONSISTENCY_ON:####################
                    fused_visual_features.append(dot_product_proj_queries)

                A = dot_product_proj_queries.shape[1]#h*w
                bias = dot_product_proj_tokens_bias.unsqueeze(1).repeat(1, A, 1)#4 310 h*w
                #bias_rel = dot_product_proj_tokens_bias_rel.unsqueeze(1).repeat(1, A, 1)

                dot_product_logit = (torch.matmul(dot_product_proj_queries, dot_product_proj_tokens.transpose(-1, -2)) / self.log_scale.exp()) + bias#视觉feat的token label 4 h*w 310
                # dot_product_logit_rel = (torch.matmul(dot_product_proj_queries, dot_product_proj_tokens_rel.transpose(-1,
                #                                                                                               -2)) / self.log_scale.exp()) + bias_rel
                if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_DOT_PRODUCT:###################
                    dot_product_logit = torch.clamp(dot_product_logit, max=50000)
                    dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
                    #dot_product_logit_rel = torch.clamp(dot_product_logit_rel, max=50000)
                    #dot_product_logit_rel = torch.clamp(dot_product_logit_rel, min=-50000)


                dot_product_logits.append(dot_product_logit)#截断到-50000 50000
                #dot_product_logits_rel.append(dot_product_logit_rel)

            if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS:
                feat = feature
                BF, CF, HF, WF = feat.shape
                shallow_img_emb = permute_and_flatten(feat, BF, -1, CF, HF, WF)
                shallow_img_emb_feats.append(shallow_img_emb)

        # no matter the feature is from backboone or from fpn, we use shallow_img_embs all the time
        if shallow_img_emb_feats is not None and shallow_text_emb is not None:
            # shallow_img_embs = torch.cat(shallow_img_embs, dim=1)
            proj_tokens = shallow_text_emb
        return logits, bbox_reg, centerness, t_logits, proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, fused_visual_features,embedding_rel



class VLDyHeadModule(torch.nn.Module):

    def __init__(self, cfg):
        super(VLDyHeadModule, self).__init__()
        self.cfg = cfg
        self.head = VLDyHead(cfg)
        box_coder = BoxCoder(cfg)
        self.loss_evaluator = make_atss_loss_evaluator(cfg, box_coder)
        self.box_selector_train = make_atss_postprocessor(cfg, box_coder, is_train=True)
        self.box_selector_test = make_atss_postprocessor(cfg, box_coder, is_train=False)
        self.anchor_generator = make_anchor_generator_complex(cfg)

        self.lang_model = cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE
        self.joint_embedding_size = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_SIZE
        self.joint_embedding_dropout = cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_DROPOUT


        if self.lang_model in ["bert-base-uncased", "roberta-base", "clip"]:
            self.lang_dim = cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM
        else:
            self.lang_dim = 1024
        #####add module################
        self.box_feat_dim=256

        self.use_object_fuse = False
        self.direct_head_tail = False
        self.useprompt = True
        self.usecausal = False
        self.pseudo_label = False


        self.with_bias = True
        self.with_scale = True

        self.use_rel_token_loss = True
        self.rel_tokens = True

        self.test_with_rel_token_logit = True
        self.iter_c = 0
        self.addbg = False
        self.addfg = False
        self.relation_per_img = 64
        #self.relationfg_per_img=512*0.25
        self.relationfg_per_img = 64

        ###############generate_text####################
        self.generate_text = True
        self.generate_text_causal = False
        self.generate_text_object = False
        self.useCLIP_text = False

        self.rel_caption_base=['kick', 'hits', 'hang', 'kiss', 'under', 'dance', 'inside_of', 'interacts_with', 'plays', 'ride', 'contain', 'holds', 'on', 'wears', 'at']
        self.rel_caption_all=['at', 'holds', 'wears', 'surf', 'hang', 'drink', 'holding_hands', 'on', 'ride', 'dance', 'skateboard', 'catch',
         'highfive', 'inside_of', 'eat', 'cut', 'contain', 'handshake', 'kiss', 'talk_on_phone', 'interacts_with',
         'under', 'hug', 'throw', 'hits', 'snowboard', 'kick', 'ski', 'plays', 'read']

        self.rel_idx=[self.rel_caption_all.index(base_rels) for base_rels in self.rel_caption_base]

        self.use_CLIPtext_object = False
        self.generate_text_logit = False
        self.generate_text_sgdet = True
        self.use_sigmoid = True
        self.use_softmax = False
        self.onlyzR = False

        if self.onlyzR:
            torch.manual_seed(0)
            self.rand_relation_idx = torch.randperm(51 - 1)  # 44 39 7 6 17 29
            self.base_rel_idx_ori = self.rand_relation_idx[:int((51 - 1) * 0.5)].tolist()
            self.ov_relation_idx = list(set(self.rand_relation_idx.tolist()) - set(self.base_rel_idx_ori)) + list(set(self.base_rel_idx_ori) - set(self.rand_relation_idx.tolist()))
            self.base_rel_idx = [i + 1 for i in self.base_rel_idx_ori]
        self.use_topk = True

        if self.use_topk:
            self.topk_relation = 100
            self.topk_for_mapping = 1

        if self.use_CLIPtext_object or self.useCLIP_text:

            self.clip_model, _ = clip.load("ViT-L/14@336px")
            self.topk_for_object = 1
            for p in self.clip_model.parameters():
                p.requires_grad_(False)




        if self.generate_text:
            if self.generate_text_causal:
                self.causal_weight_gen = 100.0
                # self.load_prior = "/data/myk/myk/openSGG/VS3/logits/OI_gen02/qhat_nosqrt_1.csv"
                if self.training == True:
                    self.qhat = self.initial_qhat(class_num=35)
                self.generate_text_causal_test = False
                if self.generate_text_causal_test:
                    self.causal_weight_test_gen = 0.07
                    #if not self.training:
                    self.iter_test=0



            self.gen_weight = 0.2
            torch.manual_seed(0)
            self.rand_relation_idx = torch.randperm(51 - 1)  # 44 39 7 6 17 29
            self.base_rel_idx_ori = self.rand_relation_idx[:int((51 - 1) * 0.5)].tolist()
            self.base_rel_idx = [i + 1 for i in self.base_rel_idx_ori]

            self.seen_rels = []
            for i in self.base_rel_idx:
                self.seen_rels.append(VG150_REL_CATEGORIES[i])


            self.class_generate = Generate_with_T5()
            self.use_topk_text = True
            self.rel_nms = False
            self.topk_text = 5###########1 R 12.64  10 R 12.11   top1  300ranked R 15.16  top3 15.73 top5  R 19.98 zR 0 300ranked 18.40 zR 10.28  top10  R 14.40  zR 10.36 top20 12.20 9.08
            self.num_of_relations = 100
            self.num_beams = 1###########100 no onlyZR R 17.46 zR 16.56   onlyZR R 11.55 zR 19.44
            ####1 beam R 17.27  ZR 14.04 sgdet onlyZR 0
            ####2 beam R 18.36  ZR 15.87 sgdet
            ####3 beam R 17.46  ZR 16.56 sgdet   R 17.75  ZR 16.56 maxlen 3  onlyZR zR 2.54 R 1.61                   top5 R 12.61 zR 3.69  top1 R12.94 zR 0
            ####4 beam R 16.41  ZR 14.55 sgdet
            ####5 beam R 15.50  ZR 16.22 sgdet   R 18.13  ZR 17.16 maxlen 3   onlyZR zR R150 4.11 R 2.42 zR R100 4.33 R 2.46    top10_R100  zR 11.85 R 8.89  top1 R3.63 zR 1.88  maxlen3 4.07 1.96   maxlen10 3.63 1.88（和30一样）   top5 R4.03 zR 1.78
            ####7 beam R 14.58  ZR 14.98 sgdet
            ####10 beam R 14.65  ZR 13.73 sgdet  R 15.76  ZR 13.47 maxlen 3     结果大致符合 onlyZR R100 zR 3.73 R 2.30   R150 zR 3.83 R 2.37
            # self.clip_model, _ = clip.load("ViT-L/14@336px")
            # self.clip_model.to(self.device)
        else:
            if self.generate_text_object:
                self.class_generate = Generate_with_T5()
                self.num_beams = 1
                self.topk_text_object = 2


        if self.pseudo_label:###30.02 42.02
            self.gen_weight= 0.0
            self.pseudo_label_topk = 1
            self.pseudo_gate = 0.3
            self.pseudo_loss_weight = 10.0
            self.pseudo_onlybase = True


        dyhead_tower_L = []
        if cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "bert-base-uncased":
            lang_cfg = BertConfig.from_pretrained(cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE)
        if cfg.MODEL.DYHEAD.CONV_FUNC:
            conv_func = lambda i, o, s: eval(cfg.MODEL.DYHEAD.CONV_FUNC)(i, o, s, bn_type=bn_type)
        else:
            conv_func = lambda i, o, s: Conv3x3Norm(i, o, s, deformable=use_deform, bn_type=bn_type)

        if cfg.MODEL.DYHEAD.USE_GN:
            bn_type = ['gn', cfg.MODEL.GROUP_NORM.NUM_GROUPS]
        elif cfg.MODEL.DYHEAD.USE_NSYNCBN:
            bn_type = 'nsbn'
        elif cfg.MODEL.DYHEAD.USE_SYNCBN:
            bn_type = 'sbn'
        else:
            bn_type = None
        use_deform = cfg.MODEL.DYHEAD.USE_DFCONV
        use_dyrelu = cfg.MODEL.DYHEAD.USE_DYRELU
        use_dyfuse = cfg.MODEL.DYHEAD.USE_DYFUSE
        in_channels = cfg.MODEL.BACKBONE.OUT_CHANNELS
        channels = cfg.MODEL.DYHEAD.CHANNELS
        if self.use_rel_token_loss:
            self.bias_lang = nn.Parameter(torch.zeros(self.cfg.MODEL.LANGUAGE_BACKBONE.LANG_DIM), requires_grad=True)
            prior_prob = cfg.MODEL.DYHEAD.PRIOR_PROB
            log_scale = self.cfg.MODEL.DYHEAD.LOG_SCALE
            bias_value = -math.log((1 - prior_prob) / prior_prob)
            self.bias0 = nn.Parameter(torch.Tensor([bias_value]), requires_grad=True)
            self.log_scale = nn.Parameter(torch.Tensor([log_scale]), requires_grad=True)
            self.token_loss = TokenSigmoidFocalLoss(cfg.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_ALPHA,
                                                    cfg.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_GAMMA)  # 0.25 2.0
            if self.direct_head_tail:
                self.projection_text = nn.Linear(self.lang_dim,self.box_feat_dim*2, bias=True)
            else:
                self.depth=1
                self.projection_text_rel = nn.Linear(self.lang_dim, self.box_feat_dim, bias=True)
                self.fuse_VL=VLFuse(cfg)
                # self.fuse_V=DyConv(
                #     in_channels,
                #     channels,
                #     conv_func=conv_func,
                #     use_dyrelu=(use_dyrelu and in_channels == channels),
                #     use_dyfuse=(use_dyfuse and in_channels == channels),
                #     use_deform=(use_deform and in_channels == channels),
                # )
                for i in range(self.depth):
                    dyhead_tower_L.append(BertEncoderLayer(
                    lang_cfg,
                    clamp_min_for_underflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW,
                    clamp_max_for_overflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW))

                # self.fuse_L=BertEncoderLayer(
                #     lang_cfg,
                #     clamp_min_for_underflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW,
                #     clamp_max_for_overflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW)

            # for i in range(1):  # 6
            #     if cfg.MODEL.DYHEAD.FUSE_CONFIG.EARLY_FUSE_ON:  #################
            #         # cross-modality fusion
            #         dyhead_tower.append(
            #             VLFuse(cfg)
            #         )
                    # self language path
                    # if i < cfg.MODEL.DYHEAD.NUM_CONVS - 1 or cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_FUSED_FEATURES_DOT_PRODUCT:
                    #     if cfg.MODEL.LANGUAGE_BACKBONE.MODEL_TYPE == "bert-base-uncased":
                    #         dyhead_tower.append(
                    #             BertEncoderLayer(
                    #                 lang_cfg,
                    #                 clamp_min_for_underflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MIN_FOR_UNDERFLOW,
                    #                 clamp_max_for_overflow=cfg.MODEL.DYHEAD.FUSE_CONFIG.CLAMP_BERTATTN_MAX_FOR_OVERFLOW)
                    #             ###true
                    #         )
                    # else:  ###identify
                    #     dyhead_tower.append(
                    #         DummyLayer()
                    #     )
                # dyhead_tower.append(
                #     DyConv(
                #         in_channels if i == 0 else channels,
                #         channels,
                #         conv_func=conv_func,
                #         use_dyrelu=(use_dyrelu and in_channels == channels) if i == 0 else use_dyrelu,
                #         use_dyfuse=(use_dyfuse and in_channels == channels) if i == 0 else use_dyfuse,
                #         use_deform=(use_deform and in_channels == channels) if i == 0 else use_deform,
                #     )
                # )
            self.add_module('dyhead_tower_L', nn.Sequential(*dyhead_tower_L))



            ##################################

        if self.usecausal:
            if self.useprompt:
                #self.qhat = self.initial_qhat(class_num=9)
                self.qhat = self.initial_qhat(class_num=25)
            else:
                self.qhat = self.initial_qhat(class_num=51)

            self.causal_weight = 7.0
            self.usestatic = False
            if self.usestatic:
                self.prior=torch.tensor(self.sample_rate())



        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            self.resizer = FeatureResizer(
                input_feat_size=self.lang_dim,
                output_feat_size=self.joint_embedding_size,
                dropout=self.joint_embedding_dropout
            )
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER:
            self.tunable_linear = torch.nn.Linear(self.lang_dim, 1000, bias=False)
            self.tunable_linear.weight.data.fill_(0.0)

        if self.cfg.MODEL.DYHEAD.RELATION_CONSISTENCY_ON:
            self.relation_feat_extractor = RelationFeatureExtractor(cfg)

            # relation feature refinement
            if self.cfg.MODEL.DYHEAD.RELATION_REP_REFINER:
                relation_decoder_layer = nn.TransformerDecoderLayer(d_model=cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_SIZE, nhead=8)
                self.relation_rep_refiner = nn.TransformerDecoder(relation_decoder_layer, num_layers=2)
                self.pos_encoding = PositionalEncoding2D(cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_SIZE)

            # use freq_bias
            if self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS:
                statistics = get_dataset_statistics(cfg)
                self.relation_freq_bias = FrequencyBias(cfg, statistics)

            self.relation_structure_embed = nn.Linear(cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_SIZE, 1)
            self.relation_semantic_embed = nn.Linear(cfg.MODEL.DYHEAD.FUSE_CONFIG.JOINT_EMB_SIZE, cfg.MODEL.ROI_RELATION_HEAD.NUM_CLASSES)

    def sample_rate(self,class_stat=None):

        predicate_new_order_count = [3024465, 109355, 67144, 47326, 31347, 21748, 15300, 10011, 11059, 10764, 6712,
                                         5086, 4810, 3757, 4260, 3167, 2273, 1829, 1603, 1413, 1225, 793, 809, 676, 352,
                                         663, 752, 565, 504, 644, 601, 551, 460, 394, 379, 397, 429, 364, 333, 299, 270,
                                         234, 171, 208, 163, 157, 151, 71, 114, 44, 4]
        assert len(predicate_new_order_count) == 51

        outp = []

        num_list = predicate_new_order_count[0:len(predicate_new_order_count)]
        #num_list[0] = num_list[0] * 10.0
        sum_triplets=sum(num_list)
        for j in range(len(num_list)):
            outp.append(num_list[j]/sum_triplets)
        return outp

    def initial_qhat(self,class_num=1000):
        # initialize qhat of predictions (probability)
        #qhat = (torch.ones([1, class_num], dtype=torch.float) / class_num).cuda()
        qhat = torch.zeros([1, class_num], dtype=torch.float) .cuda()
        print("qhat size: ".format(qhat.size()))
        return qhat
    def update_qhat(self,probs, qhat, momentum, qhat_mask=None):
        if qhat_mask is not None:
            mean_prob = probs.detach() * qhat_mask.detach().unsqueeze(dim=-1)
        else:
            mean_prob = probs.detach().mean(dim=0)
        qhat = momentum * qhat + (1 - momentum) * mean_prob
        return qhat

    def forward(self, images, features, targets=None,
                language_dict_features=None,
                positive_map=None,
                captions=None,
                swint_feature_c4=None,
                language_dict_features_rel=None,
                captions_rel=None,
                positive_map_label_to_token_rel=None,
                train_ov_relation=False,
                eot_indices=None,
                all_embeddings=None,
                all_mask=None,
                eot_indices_all=None,
                all_embeddings_all=None,
                all_mask_all=None,
                lang_net=None,
                tokenize=None,
                prefix=None,
                use_CLIPtext=False,
                rel_caption_all=None,
                now_iter=0):
        self.iter_c+=1
        if self.iter_c==1:
            print("###################################")
            print("generate_text:", self.generate_text)
            print("bias:",self.with_bias)
            print("scacle:", self.with_scale)

            if self.generate_text:
                print("generate_weight:",self.gen_weight)
                print("generate_text_causal",self.generate_text_causal)
                if self.generate_text_causal:
                    print("causalgen_weight:", self.causal_weight_gen)
                    print("generate_text_causal_test",self.generate_text_causal_test)
                    if self.generate_text_causal_test:
                        print("gen_weight_test:",self.causal_weight_test_gen)

            print("generate_text_object:", self.generate_text_object)
            print("useCLIP:", self.useCLIP_text or self.use_CLIPtext_object)

            if self.generate_text_object:
                print("topk_object:", self.topk_text_object)


            if self.generate_text or self.generate_text_object:
                print("num_beams:", self.num_beams)

            print("generate_text_logit:", self.generate_text_logit)


            print("generate_text_sgdet:", self.generate_text_sgdet)
            if self.generate_text_sgdet:
                print("pseudo_label:", self.pseudo_label)
                if self.pseudo_label:
                    print("pseudo_label_topk:", self.pseudo_label_topk)
                    print("pseudo_loss_weight:", self.pseudo_loss_weight)
                    print("pseudo_gate:", self.pseudo_gate)
                    print("pseudo_onlybase:", self.pseudo_onlybase)


                print("num_of_relations:",self.num_of_relations)
                print("use_topk_text", self.use_topk_text)
                print("rel_nms", self.rel_nms)
                #if self.use_topk_text:
                    #print("topk_text", self.topk_text)

            print("use_rel_token_loss:", self.use_rel_token_loss and self.rel_tokens)
            print("test_with_rel_token_logit:", self.test_with_rel_token_logit)
            print("use_object_fuse:", self.use_object_fuse)
            print("direct_head_tail", self.direct_head_tail)
            print("addbg", self.addbg)
            print("addfg", self.addfg)
            print("prompt", self.useprompt)
            print("loss_sigmoid", self.use_sigmoid)
            print("loss_softmax", self.use_softmax)
            if self.use_topk:
                print("topk", self.topk_for_mapping)
                print("toprelation", self.topk_relation)
            print("onlyzR", self.onlyzR)



            if self.usecausal:
                if self.usestatic:
                    print("usestatic", self.usestatic)
                print("causal_weight", self.causal_weight)
            print("###################################")



        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            # resizer needed
            embedding = language_dict_features['embedded']
            embedding = self.resizer(embedding)
        elif self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:##############
            # no resizer needed
            embedding = language_dict_features['embedded']
            if train_ov_relation:
                embedding_rel = language_dict_features_rel['embedded']
                #embedding_rel = None
        else:
            embedding = None

        if "masks" in language_dict_features:
            text_masks = language_dict_features["masks"]
            if train_ov_relation:
                text_masks_rel = language_dict_features_rel["masks"]
            else:
                text_masks_rel = None
        else:
            text_masks = None
        
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.ADD_LINEAR_LAYER:#false
            embedding = self.tunable_linear.weight[:embedding.size(1), :].unsqueeze(0) + embedding
            language_dict_features['embedded'] = embedding
            language_dict_features['hidden'] = self.tunable_linear.weight[:embedding.size(1), :].unsqueeze(0) + language_dict_features['hidden']
        if train_ov_relation:
            box_cls, box_regression, centerness, token_logits, \
            proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, fused_visual_features,embeddings_rel = self.head(features,
                                                                            language_dict_features,
                                                                            embedding,
                                                                            swint_feature_c4,
                                                                            language_dict_features_rel,
                                                                            embedding_rel,
                                                                            train_ov_relation=train_ov_relation,
                                                                            use_object_fuse=self.use_object_fuse)
        else:
            box_cls, box_regression, centerness, token_logits, \
            proj_tokens, contrastive_logits, dot_product_logits, mlm_logits, shallow_img_emb_feats, fused_visual_features,embeddings_rel = self.head(features,
                                                                            language_dict_features,
                                                                            embedding,
                                                                            swint_feature_c4,
                                                                            train_ov_relation=train_ov_relation)
        anchors = self.anchor_generator(images, features)##########按像素分配的框

        if not self.training or self.cfg.MODEL.RELATION_ON:
            return self._forward_test(box_regression, centerness, anchors,
                                      box_cls,
                                      token_logits,
                                      dot_product_logits,
                                      positive_map,
                                      fused_visual_features=fused_visual_features,
                                      img_backbone_features=features,
                                      targets=targets,
                                      positive_map_label_to_token_rel=positive_map_label_to_token_rel,
                                      embeddings_rel=embeddings_rel,
                                      train_ov_relation=train_ov_relation,
                                      text_masks_rel=text_masks_rel,
                                      eot_indices=eot_indices,
                                      all_embeddings=all_embeddings,
                                      all_mask=all_mask,
                                      lang_net=lang_net,
                                      tokenize=tokenize,
                                      prefix=prefix,
                                      use_CLIPtext=use_CLIPtext,
                                      captions=captions
                                      )
        else:###################
            return self._forward_train(box_cls, box_regression, centerness, targets, anchors,
                                       captions,
                                       positive_map,
                                       token_logits,
                                       proj_tokens,
                                       contrastive_logits,
                                       dot_product_logits,
                                       text_masks,
                                       mlm_logits = mlm_logits,
                                       mlm_labels = language_dict_features["mlm_labels"],
                                       shallow_img_emb_feats=shallow_img_emb_feats,
                                       fused_visual_features=fused_visual_features,
                                       img_backbone_features=features,
                                       embeddings_rel=embeddings_rel,
                                       captions_rel=captions_rel,
                                       positive_map_label_to_token_rel=positive_map_label_to_token_rel,
                                       text_masks_rel=text_masks_rel,
                                       train_ov_relation=train_ov_relation,
                                       eot_indices=eot_indices,
                                       all_embeddings=all_embeddings,##########rel_texemb
                                       all_mask=all_mask,
                                       eot_indices_all=eot_indices_all,
                                       all_embeddings_all=all_embeddings_all,
                                       all_mask_all=all_mask_all,
                                       use_CLIPtext=use_CLIPtext,
                                       rel_caption_all=rel_caption_all,
                                       now_iter=now_iter,
                                       prefix=prefix,
                                       lang_net=lang_net,
                                       tokenize=tokenize)

    def create_positive_map(self, tokenized, tokens_positive, max_len=256,unmatched_labels=None):
        """construct a map such that positive_map[i,j] = True iff box i is associated to token j"""
        positive_map = torch.zeros((len(tokens_positive), max_len), dtype=torch.float)

        for j, tok_list in enumerate(tokens_positive):
            if tok_list==0:
                positive_map[j]=unmatched_labels
            else:
                for (beg, end) in tok_list:
                    beg_pos = tokenized.char_to_token(beg)
                    end_pos = tokenized.char_to_token(end - 1)
                    if beg_pos is None:
                        try:
                            beg_pos = tokenized.char_to_token(beg + 1)
                            if beg_pos is None:
                                beg_pos = tokenized.char_to_token(beg + 2)
                        except:
                            beg_pos = None
                    if end_pos is None:
                        try:
                            end_pos = tokenized.char_to_token(end - 2)
                            if end_pos is None:
                                end_pos = tokenized.char_to_token(end - 3)
                        except:
                            end_pos = None
                    if beg_pos is None or end_pos is None:
                        continue

                    assert beg_pos is not None and end_pos is not None
                    positive_map[j, beg_pos: end_pos + 1].fill_(1)  # token编码idx
        return positive_map / (positive_map.sum(-1)[:, None] + 1e-6)



    def _forward_train(self, box_cls, box_regression, centerness, targets, anchors,
                       captions=None,
                       positive_map=None,
                       token_logits=None,
                       proj_tokens=None,
                       contrastive_logits=None,
                       dot_product_logits=None,
                       text_masks=None,
                       mlm_logits=None,
                       mlm_labels=None,
                       shallow_img_emb_feats=None,
                       fused_visual_features=None,
                       img_backbone_features=None,
                       embeddings_rel=None,
                       captions_rel=None,
                       positive_map_label_to_token_rel=None,
                       text_masks_rel=None,
                       train_ov_relation=False,
                       eot_indices=None,
                       all_embeddings=None,
                       all_mask=None,
                       eot_indices_all=None,
                       all_embeddings_all=None,
                       all_mask_all=None,
                       use_CLIPtext=False,
                       rel_caption_all=None,
                       now_iter=0,
                       prefix=None,
                       lang_net=None,
                       tokenize=None):

        proposal_labels,anchor2token_match, loss_box_cls, loss_box_reg, loss_centerness, loss_token, loss_contrastive_align, loss_dot_product_token, loss_shallow_contrastive,tokens_rel = self.loss_evaluator(
            box_cls, box_regression, centerness, targets, anchors,
            captions,
            positive_map,
            token_logits,
            proj_tokens,
            contrastive_logits,
            dot_product_logits,
            text_masks,
            shallow_img_emb_feats,
            captions_rel=captions_rel,
            train_ov_relation=train_ov_relation
        )

        losses = {
            # "loss_cls": loss_box_cls,
            "loss_reg": loss_box_reg,
            "loss_centerness": loss_centerness
        }

        if mlm_labels is not None and mlm_logits is not None:
            losses["mlm_loss"] = nn.CrossEntropyLoss(ignore_index = -100)(mlm_logits.view(-1, mlm_logits.size(-1)), mlm_labels.view(-1)) * self.cfg.MODEL.DYHEAD.FUSE_CONFIG.MLM_LOSS_COEF

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CLASSIFICATION_LOSS:
            losses["loss_cls"] = loss_box_cls
        else:
            #losses["loss_cls"] = 0.0 * loss_box_cls
            aaaa=0

        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_TOKEN_LOSS:
            losses["loss_token"] = loss_token * self.cfg.MODEL.DYHEAD.FUSE_CONFIG.TOKEN_LOSS_WEIGHT
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_CONTRASTIVE_ALIGN_LOSS:
            losses["loss_contrastive_align"] = loss_contrastive_align * \
                                               self.cfg.MODEL.DYHEAD.FUSE_CONFIG.CONTRASTIVE_ALIGN_LOSS_WEIGHT
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_DOT_PRODUCT_TOKEN_LOSS:###################
            losses["loss_dot_product_token"] = loss_dot_product_token * \
                                               self.cfg.MODEL.DYHEAD.FUSE_CONFIG.DOT_PRODUCT_TOKEN_LOSS_WEIGHT
        if self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_SHALLOW_CONTRASTIVE_LOSS or \
                self.cfg.MODEL.DYHEAD.FUSE_CONFIG.USE_BACKBONE_SHALLOW_CONTRASTIVE_LOSS:
            losses["loss_shallow_contrastive"] = loss_shallow_contrastive * \
                                                 self.cfg.MODEL.DYHEAD.FUSE_CONFIG.SHALLOW_CONTRASTIVE_LOSS_WEIGHT





        if self.cfg.MODEL.DYHEAD.RELATION_CONSISTENCY_ON:################relation predictor###########
            if self.cfg.MODEL.DYHEAD.RELATION_REP_REFINER:
                memory_inputs = []
                for l_feat in img_backbone_features:
                    B, C, H, W = l_feat.shape
                    pos = self.pos_encoding(l_feat.permute(0,2,3,1)).permute(0,3,1,2)
                    memory_inputs.append((l_feat + 0.1 * pos).view(B, C, -1).permute(0, 2, 1)) # todo: Turn off pos ablation
                memory_inputs = torch.cat(memory_inputs, dim=1)

            all_pair_reps, all_pair_reps_refine, all_rel_tgt_labels, all_pair_obj_labels = [], [], [], []
            if self.pseudo_label:
                if self.pseudo_label_topk>1:
                    fin_rel_pseudo={}
                    for i in range(self.pseudo_label_topk):
                        fin_rel_pseudo[f'rel_pseudo_top_{i}'] = []
                        fin_rel_pseudo[f'rel_pseudo_top_score_{i}'] = []

                else:
                    rel_pseudo_top1 = []
                    rel_pseudo_top1_score = []
                # rel_pseudo_top2 = []
                # rel_pseudo_top2_score = []
                # rel_pseudo_top3 = []
                # rel_pseudo_top3_score = []
                # rel_pseudo_top4 = []
                # rel_pseudo_top4_score = []
                # rel_pseudo_top5 = []
                # rel_pseudo_top5_score = []
            ###############train token predict logit##################################
            if train_ov_relation:
                # embedding_text = self.projection_text(embeddings_rel / 2.0)
                # embedding_text = self.projection_text_rel(embeddings_rel / 2.0)
                # embedding_text_bias = torch.matmul(embeddings_rel, self.bias_lang) + self.bias0
                dot_product_token_loss_all=torch.zeros(1).to(loss_dot_product_token.device)
            #####################################################################
            if self.use_CLIPtext_object:
                losses['t5_loss_object']=0



            for img_id, target in enumerate(targets):
                # sample relation pairs
                relation_sample_map = target.get_field('relation').clone()############所有relation
                relation_sample_map.fill_diagonal_(-1)
                cand_inds = (relation_sample_map >= 0)
                relation_boxid_pairs = cand_inds.nonzero()##########n*(n-1)
                relation_labels = relation_sample_map[cand_inds]

                # convert to anchor_id pairs
                token2anchor_match_list = torch.zeros_like(text_masks[0])#text_mask——前215
                anchor2token_match_tuples = (anchor2token_match[img_id] * text_masks[img_id:img_id+1]).nonzero()#anchor的token标签——12790 308

                if self.addfg:
                # 检查每一行是否全为0
                # 使用all函数结合dim=1参数
                    is_row_all_zero = torch.all((anchor2token_match[img_id] * text_masks[img_id:img_id+1]) == 0, dim=1)
                    fg_idx=(is_row_all_zero==0).nonzero().squeeze(1)
                    anchor_fg=anchor2token_match[img_id][fg_idx]

                    relation_pairs_fg = (relation_sample_map >0).nonzero()
                    anchors_per_im = cat_boxlist(anchors[img_id])  # anchor:每个feat中每一dim（h*w）对应的原图框坐标
                    ious = boxlist_iou(anchors_per_im, targets[img_id])  # 18159 9
                    proposal_ious=ious[fg_idx]

                    prp_lab=torch.zeros_like(fg_idx)
                    prp_lab_all = {}
                    all_token=anchor_fg.nonzero()
                    for idx, tokenlabel in all_token:
                        prp_lab[idx]+= tokenlabel
                        if int(idx) not in prp_lab_all:
                            prp_lab_all[int(idx)]=[]
                        prp_lab_all[int(idx)].append(tokenlabel)
                    prp_lab = prp_lab.long()


                    tgt_lab=torch.zeros_like(proposal_ious[0])
                    tgt_lab_all = {}
                    tgtall_token=target.get_field('positive_map').nonzero()
                    for idx, tokenlabel in tgtall_token:
                        tgt_lab[idx] += tokenlabel
                        if int(idx) not in tgt_lab_all:
                            tgt_lab_all[int(idx)]=[]
                        tgt_lab_all[int(idx)].append(tokenlabel)
                    tgt_lab = tgt_lab.long()

                    is_match = (prp_lab[:, None] == tgt_lab[None]).T # & (proposal_ious > 0.5)).T   ###########anchor已经筛过
                    tgt_head_idxs = relation_pairs_fg[:, 0].contiguous().view(-1)
                    tgt_tail_idxs = relation_pairs_fg[:, 1].contiguous().view(-1)
                    tgt_rel_labs = relation_sample_map[tgt_head_idxs, tgt_tail_idxs].contiguous().view(-1)


                    fg_rel_triplets = []
                    num_prp = prp_lab.shape[0]  # 80
                    rel_possibility = torch.ones((num_prp, num_prp), device=relation_labels.device).long() - torch.eye(num_prp,
                                                                                                   device=relation_labels.device).long()
                    # only select relations between fg proposals
                    rel_possibility[prp_lab == 0] = 0
                    rel_possibility[:, prp_lab == 0] = 0  # 80 80，去除背景proposal

                    for i in range(len(tgt_rel_labs)):
                        tgt_head_idx = int(tgt_head_idxs[i])
                        tgt_tail_idx = int(tgt_tail_idxs[i])
                        tgt_rel_lab = int(tgt_rel_labs[i])
                        # find matching pair in proposals (might be more than one)
                        prp_head_idxs = torch.nonzero(is_match[tgt_head_idx]).squeeze(1)
                        prp_tail_idxs = torch.nonzero(is_match[tgt_tail_idx]).squeeze(1)
                        num_match_head = prp_head_idxs.shape[0]#6
                        num_match_tail = prp_tail_idxs.shape[0]#9
                        if num_match_head <= 0 or num_match_tail <= 0:
                            continue  ##########无配对返回
                        prp_head_idxs = prp_head_idxs.view(-1, 1).expand(num_match_head,
                                                                         num_match_tail).contiguous().view(-1)
                        prp_tail_idxs = prp_tail_idxs.view(1, -1).expand(num_match_head,
                                                                         num_match_tail).contiguous().view(-1)
                        valid_pair = prp_head_idxs != prp_tail_idxs  ############非同一个proposal
                        if valid_pair.sum().item() <= 0:
                            continue
                        prp_head_idxs = prp_head_idxs[valid_pair]
                        prp_tail_idxs = prp_tail_idxs[valid_pair]
                        rel_possibility[prp_head_idxs, prp_tail_idxs] = 0
                        fg_labels = torch.tensor([tgt_rel_lab] * prp_tail_idxs.shape[0], dtype=torch.int64,
                                                 device=relation_labels.device).view(-1, 1)#6*9 1
                        fg_rel_i = cat((prp_head_idxs.view(-1, 1), prp_tail_idxs.view(-1, 1), fg_labels), dim=-1).to(
                            torch.int64)  #######主，宾，谓 6*9 3
                        if fg_rel_i.shape[0] > 4:  #############每个关系只取 num_sample_per_gt_rel 对=4
                            judge_ious=proposal_ious.T
                            ious_score = (judge_ious[tgt_head_idx, prp_head_idxs] * judge_ious[tgt_tail_idx, prp_tail_idxs]).view(
                                -1).detach().cpu().numpy()
                            ious_score = (ious_score+1e-6/fg_rel_i.shape[0]) / (ious_score.sum()+1e-6)
                            perm = npr.choice(ious_score.shape[0], p=ious_score, size=4,
                                              replace=False)  ###多了按IOU-score随机取
                            fg_rel_i = fg_rel_i[perm]
                        if fg_rel_i.shape[0] > 0:
                            fg_rel_triplets.append(fg_rel_i)

                    # select fg relations
                    if len(fg_rel_triplets) == 0:
                        fg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=relation_labels.device)
                    else:
                        fg_rel_triplets = cat(fg_rel_triplets, dim=0).to(torch.int64)
                        if fg_rel_triplets.shape[0] > self.relationfg_per_img:
                            perm = torch.randperm(fg_rel_triplets.shape[0], device=relation_labels.device)[
                                   :self.relationfg_per_img]  #########每幅图128正例
                            fg_rel_triplets = fg_rel_triplets[perm]





                    #########################################bg############################################


                    bg_rel_inds = torch.nonzero(rel_possibility > 0).view(-1, 2)
                    bg_rel_labs = torch.zeros(bg_rel_inds.shape[0], dtype=torch.int64, device=relation_labels.device)
                    bg_rel_triplets = cat((bg_rel_inds, bg_rel_labs.view(-1, 1)), dim=-1).to(torch.int64)

                    num_neg_per_img = min(self.relation_per_img - fg_rel_triplets.shape[0], bg_rel_triplets.shape[0])
                    if bg_rel_triplets.shape[0] > 0:
                        perm = torch.randperm(bg_rel_triplets.shape[0], device=relation_labels.device)[:num_neg_per_img]
                        bg_rel_triplets = bg_rel_triplets[perm]
                    else:
                        bg_rel_triplets = torch.zeros((0, 3), dtype=torch.int64, device=relation_labels.device)

                    ########################################################################################################


                    # if both fg and bg is none
                    if fg_rel_triplets.shape[0] == 0 and bg_rel_triplets.shape[0] == 0:
                        bg_rel_triplets = torch.zeros((1, 3), dtype=torch.int64, device=relation_labels.device)


                    all_triplets=cat((fg_rel_triplets, bg_rel_triplets), dim=0)
                    sampled_rel_pairs=fg_idx[all_triplets[:,:2]]
                    sampled_rel_labels=all_triplets[:,2]
                else:
                    perm = torch.randperm(len(anchor2token_match_tuples))
                    for aid, tid in (anchor2token_match_tuples[perm]):
                        token2anchor_match_list[tid] = aid#token-anchor一一对应（随机取一个anchor）——14



                    id_box2anchor_all=[]

                    if len(target.get_field('positive_map'))>1:
                        for token_labels in target.get_field('positive_map'):
                            token_id=token_labels[:len(token2anchor_match_list)].nonzero()[0].to(torch.long)
                            id_box2anchor_idx=token2anchor_match_list[token_id]
                            id_box2anchor_all.append(id_box2anchor_idx)
                        # if len(id_box2anchor_all)>1:
                        #     id_box2anchor = torch.zeros(len(id_box2anchor_all),dtype=torch.int64).to(token2anchor_match_list.device)
                        #     id_box2anchor =
                        relation_anchorid_pairs = torch.cat(id_box2anchor_all, dim=0)[relation_boxid_pairs]  # box转为token标签对应的anchor idx
                    else:
                        #############index不对################
                        id_box2anchor = token2anchor_match_list[target.get_field('positive_map').nonzero()[:, 1]] # todo: fix for batchsize > 1#######token标签——有误，可能存在token长于1
                        relation_anchorid_pairs = id_box2anchor[relation_boxid_pairs]  # box转为token标签对应的anchor idx
                    #
                    # aaaa=0
                    # if len(relation_boxid_pairs)>0:
                    #
                    #     aaa=0
                    #relation_anchorid_pairs = torch.cat(id_box2anchor_all,dim=0)[relation_boxid_pairs]#box转为token标签对应的anchor idx

                    # sample pos:neg#########3*pos+1
                    pos_inds = (relation_labels > 0).nonzero()
                    pos_inds_x=pos_inds//(relation_sample_map.shape[0]-1)
                    pos_inds_y =pos_inds%(relation_sample_map.shape[0]-1)
                    for idx_pos,yidx in enumerate(pos_inds%(relation_sample_map.shape[0]-1)):
                        if (yidx>=pos_inds_x[idx_pos]).item():
                            pos_inds_y[idx_pos]=pos_inds_y[idx_pos]+1



                    # boxlist=target.bbox.tolist()
                    # if len (pos_inds)>0:
                    #     bbox_x,bbox_y=boxlist[pos_inds_x],boxlist[pos_inds_y]

                    neg_inds = (relation_labels == 0).nonzero()

                    if self.addbg:
                        num_neg_per_img = min(self.relation_per_img - len(pos_inds), len(neg_inds))
                        if len(neg_inds) > 0:
                            perm = torch.randperm(len(neg_inds), device=relation_labels.device)[:num_neg_per_img]
                            sample_neg_inds = neg_inds[perm]
                        else:
                            sample_neg_inds = neg_inds[torch.randperm(len(neg_inds))][:(len(pos_inds)*3+1)]
                    else:
                        sample_neg_inds = neg_inds[torch.randperm(len(neg_inds))][:(len(pos_inds)*3+1)]############此处少取样
                    neg_inds_x, neg_inds_y = sample_neg_inds // (relation_sample_map.shape[0]-1), sample_neg_inds%(relation_sample_map.shape[0]-1)

                    sampled_inds_x,sampled_inds_y=torch.cat((pos_inds_x, neg_inds_x), dim=0)[:, 0],torch.cat((pos_inds_y, neg_inds_y), dim=0)[:, 0]

                    sampled_inds = torch.cat((pos_inds, sample_neg_inds), dim=0)[:, 0]

                    sampled_rel_pairs = relation_anchorid_pairs[sampled_inds]
                    sampled_rel_labels = relation_labels[sampled_inds]

                # construct pair representations——为对应的anchor feature
                head_tail_reps = torch.cat([r[img_id] for r in fused_visual_features], dim=0)[sampled_rel_pairs]# 关系特征

                ###################
                if self.generate_text_object:
                    obj_visual_feat=torch.cat([r[img_id] for r in fused_visual_features], dim=0)[id_box2anchor]
                    all_captions_text=captions[img_id].split('. ')
                    decriptions=[]
                    for idxs in proposal_labels[img_id][id_box2anchor]:
                        decriptions.append(all_captions_text[idxs-1])

                    obj_visual_outputs = obj_visual_feat.unsqueeze(1)
                    object_features_att_mask = torch.ones(obj_visual_outputs.size()[:-1], dtype=torch.long).to(
                        obj_visual_outputs.device)
                    text_decoder_loss_object = self.class_generate(obj_visual_outputs, decriptions,
                                                            object_features_att_mask)
                    text_decoder_loss_object['t5_loss'] = 1 * text_decoder_loss_object['t5_loss']
                    losses['t5_loss_object'] = losses['t5_loss_object']+text_decoder_loss_object['t5_loss']
                ##################



                if self.addfg:
                    pair_boxes = anchors_per_im.copy_with_fields([]).resize((1, 1)).bbox[sampled_rel_pairs]
                else:
                    pair_boxes = target.copy_with_fields([]).resize((1,1)).bbox[relation_boxid_pairs[sampled_inds]] # nomalized xyxy boxes （gt）
                pair_reps = self.relation_feat_extractor(head_tail_reps, pair_boxes)################relation predictor###########

                ###############train token predict logit##################################
                if train_ov_relation and len(pair_reps)>0 and self.use_rel_token_loss and self.rel_tokens:
                    if self.direct_head_tail:
                        head_reps = head_tail_reps[:, 0]
                        tail_reps = head_tail_reps[:, 1]
                        rel_reps= torch.cat((head_reps,tail_reps),dim=-1)
                        embedding_fuse = {"visual": rel_reps,
                                       "lang": embeddings_rel[img_id],#############embeddings_rel ,grad=True(useVL)
                                       "mask": text_masks_rel[img_id]}

                    else:######################
                        if self.useprompt:########################
                            feat_inputs = {"visual": pair_reps,
                                           "lang": all_embeddings,
                                           "masks": all_mask,
                                           "use_prompt": True,
                                           "single_feat": False,
                                           "eot_indices": eot_indices}
                            if self.pseudo_label:
                                if self.pseudo_onlybase:
                                    feat_inputs_all = feat_inputs
                                else:
                                    feat_inputs_all = {"visual": pair_reps,
                                                   "lang": all_embeddings_all,
                                                   "masks": all_mask_all,
                                                   "use_prompt": True,
                                                   "single_feat": False,
                                                   "eot_indices": eot_indices_all}

                            # def forward(self, inputs, single_feat=False, masks=None, eot_indices=None,
                            #             use_prompt=False):
                            if not use_CLIPtext:#######################
                                embedding_fuse = self.dyhead_tower_L(feat_inputs)
                                if self.pseudo_label:
                                    embedding_fuse_all = self.dyhead_tower_L(feat_inputs_all)
                        else:
                            feat_inputs = {"visual": pair_reps,
                                           "lang": embeddings_rel[img_id],
                                           "masks": text_masks_rel[img_id],
                                           "use_prompt": False,
                                           "single_feat": True,
                                           "eot_indices": None}
                            embedding_fuse = self.fuse_VL(feat_inputs,single_feat=True)
                            embedding_fuse = self.fuse_L(embedding_fuse)
                            #embedding_fuse = self.fuse_V(embedding_fuse, single_feat=True)#############有二维卷积
                    if not use_CLIPtext:
                        if self.useprompt:
                            embedding_text_emb = embedding_fuse["lang"][torch.arange(embedding_fuse["lang"].shape[0]), eot_indices].squeeze(0)
                            if self.pseudo_label:
                                if self.pseudo_onlybase:
                                    embedding_text_emb_all = embedding_fuse["lang"][torch.arange(embedding_fuse["lang"].shape[0]), eot_indices].squeeze(0)
                                else:
                                    embedding_text_emb_all = embedding_fuse_all["lang"][torch.arange(embedding_fuse_all["lang"].shape[0]), eot_indices_all].squeeze(0)
                        else:
                            embedding_text_emb = embedding_fuse["lang"].squeeze(0)
                    else:
                        embedding_text_emb = all_embeddings



                    if not use_CLIPtext:
                        embedding_vis_emb = embedding_fuse["visual"]#.squeeze(0)
                        if self.pseudo_label:
                            embedding_vis_emb_all = embedding_fuse_all["visual"]  # .squeeze(0)
                    else:
                        embedding_vis_emb = feat_inputs["visual"].squeeze(0)

                    embedding_rel = F.normalize(embedding_text_emb, p=2, dim=-1)
                    if self.pseudo_label:
                        embedding_rel_all = F.normalize(embedding_text_emb_all, p=2, dim=-1)

                    if self.direct_head_tail:
                        embedding_text = self.projection_text(embedding_rel / 2.0)
                    else:
                        embedding_text = self.projection_text_rel(embedding_rel / 2.0)
                        if self.pseudo_label:
                            embedding_text_all = self.projection_text_rel(embedding_rel_all / 2.0)

                    if self.pseudo_label:
                        embedding_text_bias_all = torch.matmul(embedding_rel_all, self.bias_lang) + self.bias0####wx+b
                    embedding_text_bias = torch.matmul(embedding_rel, self.bias_lang) + self.bias0  ####wx+b

                    # embedding_text = self.projection_text_rel(embeddings_rel / 2.0)
                    # embedding_text_bias = torch.matmul(embeddings_rel, self.bias_lang) + self.bias0


                    new_target=[]
                    # unmatched_labels = torch.zeros(embeddings_rel.shape[1], device=embeddings_rel.device)
                    # unmatched_labels[-1] = 1
                    for i in range(len(sampled_rel_labels)):
                        label_i=sampled_rel_labels[i]
                        #new_target_i=[]
                        if int(label_i) in target.extra_fields['rel_label_to_positions']:  # NOTE: Only add those that actually appear in the final caption
                            new_target_i = [target.extra_fields['rel_label_to_positions'][int(label_i)]]
                            new_target.append(new_target_i)
                        else:
                            new_target.append(0)


                    if self.useprompt:
                        labels = torch.zeros([len(sampled_rel_labels),len(embedding_rel)], device=sampled_rel_labels.device)
                        rel_id=0
                        for rel_label in sampled_rel_labels:
                            if rel_label>0:
                                labels[rel_id, rel_label-1] = 1
                            rel_id+=1
                        text_masks_token=torch.ones(len(embedding_rel), device=sampled_rel_labels.device)
                    else:
                        rel_positive_map=self.create_positive_map(tokens_rel, new_target, max_len=130,unmatched_labels=unmatched_labels)












                    #=bias = embedding_text_bias[img_id].unsqueeze(0).repeat(A, 1)
                    #dot_product_logit = (torch.matmul(pair_reps, embedding_text[img_id].transpose(-1,-2)) / self.log_scale.exp()) + bias

                    if self.with_bias:
                        A = pair_reps.shape[0]
                        bias = embedding_text_bias.unsqueeze(0).repeat(A, 1)
                        dot_product_logit = (torch.matmul(embedding_vis_emb, embedding_text.transpose(-1,-2)) / self.log_scale.exp()) + bias

                        if self.pseudo_label:
                            bias_all = embedding_text_bias_all.unsqueeze(0).repeat(A, 1)
                            dot_product_logit_all = (torch.matmul(embedding_vis_emb_all, embedding_text_all.transpose(-1,-2)) / self.log_scale.exp()) + bias_all
                    else:
                        if self.with_scale:
                            dot_product_logit = (torch.matmul(embedding_vis_emb,embedding_text.transpose(-1,-2)) / self.log_scale.exp())
                        else:
                            dot_product_logit = (torch.matmul(embedding_vis_emb,embedding_text.transpose(-1, -2)))


                    dot_product_logit = torch.clamp(dot_product_logit, max=50000)
                    dot_product_logit = torch.clamp(dot_product_logit, min=-50000)


                    if self.pseudo_label:

                        dot_product_logit_all = torch.clamp(dot_product_logit_all, max=50000)
                        dot_product_logit_all = torch.clamp(dot_product_logit_all, min=-50000)
                        sim_values, sim_indices = torch.sigmoid(dot_product_logit_all).topk(self.pseudo_label_topk)
                        if self.pseudo_label_topk>1:
                            relscores_all={}
                            relclass_all = {}
                            all_rel_pseudo={}
                            for i in range(self.pseudo_label_topk):
                                relscores_all[f'rel_scores_text_top_{i}']= sim_values[:, i]
                                relclass_all[f'rel_class_text_top_{i}'] = sim_indices[:, i]
                                all_rel_pseudo[f'rel_pseudo_top_{i}']=[]
                                all_rel_pseudo[f'rel_pseudo_score_top_{i}'] = []

                        else:
                            rel_scores_text_topk = sim_values.view(-1)
                            rel_class_text_topk = sim_indices.view(-1)




                    if self.usecausal:

                        if self.usestatic:
                            delta_logits = torch.log(self.prior)
                            # delta_logits[0] = torch.max(delta_logits[1:])
                            delta_logits = delta_logits.to(dot_product_logit.device)
                        else:#################################
                            self.qhat = self.update_qhat(torch.sigmoid(dot_product_logit.detach()),self.qhat,momentum=0.99)
                            delta_logits = self.qhat
                            # self.qhat = self.update_qhat(torch.softmax(dot_product_logit.detach(),dim=-1), self.qhat,
                            #                              momentum=0.99)
                            # delta_logits = torch.log(self.qhat)

                            #delta_logits[0][0] = torch.max(delta_logits[0][1:])

                        dot_product_logit = dot_product_logit + self.causal_weight * delta_logits
                        dot_product_logit = torch.clamp(dot_product_logit, max=50000)
                        dot_product_logit = torch.clamp(dot_product_logit, min=-50000)


                    if self.generate_text_causal:
                        self.qhat = self.update_qhat(torch.sigmoid(dot_product_logit.detach()), self.qhat,momentum=0.99)
                        # delta_logits = torch.log(self.qhat)

                    if self.useprompt:###########################
                        dot_product_token_loss = self.token_loss(dot_product_logit,labels, text_masks=text_masks_token,
                                                                 version="binary")
                    else:
                        rel_positive_map=rel_positive_map.to(dot_product_logit.device)
                        dot_product_token_loss = self.token_loss(dot_product_logit,
                                                                      rel_positive_map, text_masks=text_masks_rel[img_id],
                                                                      version="binary")



                    dot_product_token_loss_all+=dot_product_token_loss#1.6269e-5

                #################################################


                # refine pair_representation
                pair_reps_refine = pair_reps
                if self.cfg.MODEL.DYHEAD.RELATION_REP_REFINER:
                    pair_reps_refine = self.relation_rep_refiner(
                        pair_reps.unsqueeze(1),
                        memory_inputs[img_id:img_id+1].permute(1, 0, 2)
                    )[:, 0]

                # collect reps & tgts
                all_pair_reps.append(pair_reps)
                if self.pseudo_label:
                    if self.pseudo_label_topk>1:
                        for i in range(self.pseudo_label_topk):
                            fin_rel_pseudo[f'rel_pseudo_top_score_{i}'].append(relscores_all[f'rel_scores_text_top_{i}'])
                            fin_rel_pseudo[f'rel_pseudo_top_{i}'].append(relclass_all[f'rel_class_text_top_{i}'])
                    else:
                        rel_pseudo_top1.append(rel_class_text_topk)
                        rel_pseudo_top1_score.append(rel_scores_text_topk)
                all_pair_reps_refine.append(pair_reps_refine)
                all_rel_tgt_labels.append(sampled_rel_labels)
                if self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS:###############Full-supervised
                    pair_obj_labels = target.get_field('labels')
                    all_pair_obj_labels.append(pair_obj_labels[relation_boxid_pairs[sampled_inds]])

            # compute structural consistency loss and semantic consistency loss######################

            if self.pseudo_label:
                if self.pseudo_label_topk>1:
                    fin_pseudo={}
                    for i in range(self.pseudo_label_topk):
                        fin_pseudo[f'pseudo_top{i}'] = torch.cat(fin_rel_pseudo[f'rel_pseudo_top_{i}'], dim=0)
                else:
                    pseudo_top1 = torch.cat(rel_pseudo_top1, dim=0)
                    pseudo_top1_score = torch.cat(rel_pseudo_top1_score, dim=0)

            all_pair_reps = torch.cat(all_pair_reps, dim=0)#主宾feat
            all_pair_reps_refine = torch.cat(all_pair_reps_refine, dim=0)#同上
            all_rel_tgt_labels = torch.cat(all_rel_tgt_labels, dim=0)

            ################generate text###################
            if self.generate_text_object:
                if losses['t5_loss_object']>0:
                    losses['t5_loss_object']=losses['t5_loss_object']/len(targets)


            if self.generate_text:

                rel_text = captions_rel[0].split('. ')
                rel_descriptions = []
                rel_weight=[]
                if self.pseudo_label:
                    visual_embbedings_pseudo = []
                    if self.pseudo_label_topk > 1:
                        all_rel_descriptions_pseudo={}
                        for i in range(self.pseudo_label_topk):
                            all_rel_descriptions_pseudo[f'rel_descriptions_pseudo_top_{i}']=[]
                    else:
                        rel_descriptions_pseudo_top1 = []

                visual_embbedings = []

                visual_emb = all_pair_reps

                rel_id = 0
                # num_rels = len(all_rel_tgt_labels.nonzero())
                # if num_rels > 0:

                pseudo_ids = []
                tgt_ids = []
                rel_weight_pseudo = []
                for tgt_label in all_rel_tgt_labels:
                    if self.pseudo_label:
                        if self.pseudo_gate > 0:
                            if pseudo_top1_score[rel_id] > self.pseudo_gate:############！！！需要每个都判断
                                if self.pseudo_label_topk>1:
                                    for i in range(self.pseudo_label_topk):
                                        all_rel_descriptions_pseudo[f'rel_descriptions_pseudo_top_{i}'].append(rel_caption_all[fin_pseudo[f'pseudo_top{i}'][rel_id]])
                                else:
                                    if self.pseudo_onlybase:
                                        pseudo_ids.append(rel_id)
                                        rel_descriptions_pseudo_top1.append(rel_text[pseudo_top1[rel_id]])
                                        if self.generate_text_causal:
                                            rel_weight_pseudo.append((1 - self.causal_weight_gen * self.qhat).squeeze(0)[pseudo_top1[rel_id]])

                                    else:
                                        rel_descriptions_pseudo_top1.append(rel_caption_all[pseudo_top1[rel_id]])
                                visual_embbedings_pseudo.append(visual_emb[rel_id])

                    if tgt_label > 0:
                        if self.generate_text_causal:
                            rel_weight.append((1-self.causal_weight_gen*self.qhat).squeeze(0)[tgt_label-1])
                        rel_descriptions.append(rel_text[tgt_label-1])
                        visual_embbedings.append(visual_emb[rel_id])
                        tgt_ids.append(rel_id)
                        if self.pseudo_label:
                            if self.pseudo_gate==0:
                                if self.pseudo_label_topk>1:
                                    for i in range(self.pseudo_label_topk):
                                        all_rel_descriptions_pseudo[f'rel_descriptions_pseudo_top_{i}'].append(rel_caption_all[fin_pseudo[f'pseudo_top{i}'][rel_id]])
                                else:
                                    rel_descriptions_pseudo_top1.append(rel_caption_all[pseudo_top1[rel_id]])
                                visual_embbedings_pseudo.append(visual_emb[rel_id])
                    rel_id += 1

                if len(rel_descriptions) > 0:
                    visual_outputs = torch.stack(visual_embbedings, dim=0).unsqueeze(1)
                    object_features_att_mask = torch.ones(visual_outputs.size()[:-1], dtype=torch.long).to(
                        visual_outputs.device)
                    if self.generate_text_causal:
                        text_decoder_loss = self.class_generate(visual_outputs, rel_descriptions,
                                                            object_features_att_mask,rel_weight=rel_weight)
                    else:
                        text_decoder_loss = self.class_generate(visual_outputs, rel_descriptions,
                                                            object_features_att_mask)

                    text_decoder_loss['t5_loss'] = self.gen_weight * text_decoder_loss['t5_loss']


                    #if self.generate_text_causal:
                        # input_visual_feat = visual_outputs
                        # atts_t5 = torch.ones(input_visual_feat.size()[:-1], dtype=torch.long).to( input_visual_feat.device)
                        # text_decoder_output = self.class_generate.text_decoder(
                        #      {'object_features': input_visual_feat, 'atts_t5': atts_t5}, num_beams=self.num_beams)
                        #
                        #
                        # generate_logit = self.encode_text(text_decoder_output, tokenize=tokenize, lang_net=lang_net,
                        #                                   prefix=prefix, visual_input=input_visual_feat,
                        #                                   target_emb=embedding_text, beams=self.num_beams,
                        #                                   only_zR=self.onlyzR, use_CLIPtext_object=use_CLIPtext,
                        #                                   generate_text=self.generate_text)
                        #
                        #
                        # generate_logit = generate_logit.sigmoid()
                        # generate_logit =torch.sqrt(generate_logit)
                        # self.qhat = self.update_qhat(generate_logit.detach(), self.qhat,
                        #                              momentum=0.99)
                        #
                        #
                        # delta_logits = self.qhat
                        # if now_iter % 1000 == 0 or now_iter==5:
                        #     print("###################################################")
                        #     print("qhat:", delta_logits)
                        #     # ckpt_dir=('/qhat_%d.csv' % self.iter_2)
                        #     np.savetxt(self.load_prior, delta_logits.detach().cpu().numpy(), fmt='%f', delimiter=',',
                        #                footer='\n')




                    if self.pseudo_label:
                        if self.pseudo_label_topk>1:
                            all_pseudo_loss={}
                            for i in range(self.pseudo_label_topk):
                                all_pseudo_loss[f'rel_descriptions_pseudo_top_{i}'] = self.class_generate(visual_outputs,all_rel_descriptions_pseudo[f'rel_descriptions_pseudo_top_{i}'],
                                                                       object_features_att_mask)
                                losses[f't5_loss_pseudo_top{i}'] = self.pseudo_loss_weight * all_pseudo_loss[f'rel_descriptions_pseudo_top_{i}']['t5_loss']

                        else:
                            if len(visual_embbedings_pseudo)>0:
                                visual_outputs_pseudo = torch.stack(visual_embbedings_pseudo, dim=0).unsqueeze(1)
                                object_features_att_mask_pseudo = torch.ones(visual_outputs_pseudo.size()[:-1], dtype=torch.long).to(
                                    visual_outputs_pseudo.device)
                                if self.generate_text_causal:
                                    pseudo_loss_top1 = self.class_generate(visual_outputs_pseudo, rel_descriptions_pseudo_top1,
                                                                       object_features_att_mask_pseudo,rel_weight=rel_weight_pseudo)
                                else:
                                    pseudo_loss_top1 = self.class_generate(visual_outputs_pseudo, rel_descriptions_pseudo_top1,
                                                                       object_features_att_mask_pseudo)

                                losses['t5_loss_pseudo_top1'] = self.pseudo_loss_weight * pseudo_loss_top1['t5_loss']
                            else:
                                losses['t5_loss_pseudo_top1'] = 0.0
                    losses.update(text_decoder_loss)
                ################generate text###################
                if len(rel_descriptions) > 0:
                    input_visual_feat = visual_outputs
                    atts_t5 = torch.ones(input_visual_feat.size()[:-1], dtype=torch.long).to(input_visual_feat.device)
                    text_decoder_output = self.class_generate.text_decoder(
                        {'object_features': input_visual_feat, 'atts_t5': atts_t5}, num_beams=self.num_beams)
                    # extra_description={}########self.num_beams = 5 description比较多  text_len 3
                    # for texts in text_decoder_output['pred_object_descriptions']:
                    #     if texts not in VG150_REL_CATEGORIES:
                    #         if texts not in extra_description:
                    #             extra_description[texts]=0
                    #         else:
                    #             extra_description[texts]=extra_description[texts]+1





            ###############train token predict logit##################################
            if train_ov_relation:
                if self.use_rel_token_loss and self.rel_tokens:
                    num_rels=len(all_rel_tgt_labels.nonzero())
                    if num_rels>0:
                        dot_product_token_loss_all=dot_product_token_loss_all/(num_rels+ 1e-6)
                        losses['loss_dot_product_token_rel'] = dot_product_token_loss_all
                        #############pseudo_label###########################
                        #losses['loss_dot_product_token_rel']= 0.0
                        #################################

            ########################################################################
            if train_ov_relation and self.use_rel_token_loss and self.rel_tokens:
                # losses['rel_structural_cons_loss'] = self._relation_structure_consistency_loss(
                #     self.relation_structure_embed(all_pair_reps).squeeze(), (all_rel_tgt_labels > 0).float()
                # )#binary loss
                aaaa=0
            else:
                losses['rel_structural_cons_loss'] = self._relation_structure_consistency_loss(
                    self.relation_structure_embed(all_pair_reps).squeeze(), (all_rel_tgt_labels > 0).float()
                )#binary loss


            if train_ov_relation and self.use_rel_token_loss and self.rel_tokens:
                # rel_cls_logits = self.relation_semantic_embed(all_pair_reps_refine)
                # losses['rel_semantic_cons_loss'] = F.cross_entropy(rel_cls_logits,
                #                                                    all_rel_tgt_labels)  # relation ce loss
                aaaa = 0
            else:
                rel_cls_logits = self.relation_semantic_embed(all_pair_reps_refine)

                if self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS:
                    all_pair_obj_labels = torch.cat(all_pair_obj_labels, dim=0)
                    rel_cls_logits = rel_cls_logits + self.relation_freq_bias.index_with_labels(all_pair_obj_labels)


                if self.usecausal:

                    if self.usestatic:
                        delta_logits = torch.log(self.prior)
                        #delta_logits[0] = torch.max(delta_logits[1:])
                        delta_logits=delta_logits.to(rel_cls_logits.device)
                    else:
                        self.qhat = self.update_qhat(torch.softmax(rel_cls_logits.detach(), dim=-1), self.qhat,
                                                     momentum=0.99)
                        delta_logits = torch.log(self.qhat)
                        delta_logits[0][0] = torch.max(delta_logits[0][1:])


                    rel_cls_logits = rel_cls_logits + self.causal_weight * delta_logits


                losses['rel_semantic_cons_loss'] = F.cross_entropy(rel_cls_logits, all_rel_tgt_labels)#relation ce loss


        if self.cfg.MODEL.RPN_ONLY:#################
            train_dict = {}
            if len(tgt_ids)>0:
                train_dict['sampled_inds_x'] = sampled_inds_x
                train_dict['sampled_inds_y'] = sampled_inds_y
                train_dict['tgt_ids'] = tgt_ids
                train_dict['tgt_text'] = rel_descriptions
                if self.pseudo_label:  #######################
                    if len(pseudo_ids)>0:
                        train_dict['pseudo_ids']=pseudo_ids
                        train_dict['pseudo_text']=rel_descriptions_pseudo_top1

            return None, losses, None,train_dict

        else:
            # Let's just use one image per batch
            assert (box_regression[0].shape[0]) == 1
            positive_map_label_to_token = create_positive_map_label_to_token_from_positive_map(positive_map, plus=1)
            boxes = self.box_selector_train(box_regression, centerness, anchors,
                                        box_cls,
                                        token_logits,
                                        dot_product_logits,
                                        positive_map=positive_map_label_to_token
                                        )
            train_boxes = []
            for b, t in zip(boxes, targets):
                tb = t.copy_with_fields(["labels"])
                tb.add_field("scores", torch.ones(tb.bbox.shape[0], dtype=torch.bool, device=tb.bbox.device))
                train_boxes.append(cat_boxlist([b, tb]))
            return train_boxes, losses, fused_visual_features

    def convert_token_to_rel_logits(self,logits,positive_map_label_to_token_rel, score_agg=None):
        scores = torch.zeros(logits.shape[0], len(positive_map_label_to_token_rel)).to(logits.device)
        # 256 -> OBJ_NUM_CLASSES, average for each class
        if positive_map_label_to_token_rel is not None:
            # score aggregation method
            if score_agg == "MEAN":  ######################
                for label_j in positive_map_label_to_token_rel:
                    # corner case
                    if label_j > scores.shape[-1] or len(positive_map_label_to_token_rel[label_j]) == 0:
                        print("error:", positive_map_label_to_token_rel)
                        break
                    scores[ :, label_j - 1] = logits[ :, torch.LongTensor(positive_map_label_to_token_rel[label_j])].mean(-1)
            else:
                raise NotImplementedError
        return scores


    def encode_text(self,text_decoder_output,tokenize,lang_net,prefix,visual_input,target_emb,beams=1,only_zR=False,use_CLIPtext_object=False,generate_text=False):
        rel_caption = text_decoder_output['pred_object_descriptions']
        log_prob = text_decoder_output['logprobs']
        pred_object_des = []
        eot_indices = []
        all_token_embeddings = []
        all_masks = []
        beam_size = beams
        for ii in range(0, len(rel_caption), beam_size):
            pred_object_des.append(', '.join(rel_caption[ii:ii + beam_size]))


        for pred_object in pred_object_des:
        #for rel in rel_caption:
            if len(pred_object) > 50:
                pred_object = pred_object[:50]


            if use_CLIPtext_object:
                text = clip.tokenize([pred_object]).to(log_prob.device)
                text_features = self.clip_model.encode_text(text)
                text_features = text_features / text_features.norm(dim=1, keepdim=True)  ###text_feature norm
                all_token_embeddings.append(text_features.to(torch.float32))
            else:
                relidx = torch.Tensor(tokenize.encode(pred_object)).to(log_prob.device)  # 101 start 102 end

                remain_length = 77 - 8 - len(relidx)
                rel_mask = torch.cat([torch.ones(len(relidx)), torch.zeros(remain_length)]).to(log_prob.device)

                eot_indices.append(77 - remain_length - 1)
                padding_zeros = torch.zeros(remain_length, dtype=torch.long).to(log_prob.device)
                token = torch.cat([relidx, padding_zeros])
                text_input = {"input_ids": token.unsqueeze(0).type(torch.long),
                              "attention_mask": rel_mask.unsqueeze(0).type(torch.long)}

                language_rel = lang_net(text_input)
                token_embedding = language_rel["hidden"].squeeze(0)
                full_token_embedding = torch.cat([
                    prefix, token_embedding], dim=0)
                all_token_embeddings.append(full_token_embedding)
                full_mask = torch.cat([torch.ones(8).to(log_prob.device), rel_mask]).to(log_prob.device)
                all_masks.append(full_mask.type(relidx.dtype))


        if not use_CLIPtext_object:
            eot_indices = torch.as_tensor(eot_indices)
            all_mask = torch.stack(all_masks, dim=0)
        all_embeddings = torch.stack(all_token_embeddings, dim=0)###tensor 36 77 768
        if not use_CLIPtext_object:
            feat_inputs = {"visual": visual_input,
                           "lang": all_embeddings,
                           "masks": all_mask,
                           "use_prompt": True,
                           "single_feat": False,
                           "eot_indices": eot_indices}
            embedding_fuse = self.dyhead_tower_L(feat_inputs)


        if not use_CLIPtext_object:
            embedding_text_emb = embedding_fuse["lang"][torch.arange(embedding_fuse["lang"].shape[0]), eot_indices].squeeze(0)
            embedding_rel = F.normalize(embedding_text_emb, p=2, dim=-1)
            embedding_text = self.projection_text_rel(embedding_rel / 2.0)
            ###36 768
        else:
            embedding_text = all_embeddings.squeeze(1)
            if generate_text:
                embedding_text = self.projection_text_rel(embedding_text / 2.0)



        if only_zR:
            target_emb=target_emb[self.ov_relation_idx]

        dot_product_logit = torch.matmul(embedding_text,target_emb.transpose(-1, -2)) / self.log_scale.exp()
        if use_CLIPtext_object and not generate_text:
            sim_values, sim_indices = dot_product_logit.topk(self.topk_for_object)
            rel_scores_text_topk = sim_values.view(-1)
            rel_class_text_topk = sim_indices.view(-1)+1
            return rel_scores_text_topk,rel_class_text_topk,dot_product_logit
        else:
            return dot_product_logit

    def topk_select(self,logit=None,topk=1,feats=None,boxes=None):

        sim_values, sim_indices = logit.topk(topk)
        num_feat = feats.shape[0]
        feats_new = feats.unsqueeze(1).repeat(1, topk, 1).view(topk * num_feat, -1)
        boxes_new = boxes.unsqueeze(1).repeat(1, topk, 1).view(topk * num_feat, -1)
        rel_scores_text_topk = sim_values.view(-1)
        rel_class_text_topk = sim_indices.view(-1)+1
        return rel_scores_text_topk,rel_class_text_topk,feats_new,boxes_new

    def area(self,bbox=None):

        TO_REMOVE = 1
        box = bbox
        area = (box[:, 2] - box[:, 0] + TO_REMOVE) * (box[:, 3] - box[:, 1] + TO_REMOVE)


        return area

    def boxlist_iou(self,boxlist1,boxlist2):
        """Compute the intersection over union of two set of boxes.
        The box order must be (xmin, ymin, xmax, ymax).

        Arguments:
          box1: (BoxList) bounding boxes, sized [N,4].
          box2: (BoxList) bounding boxes, sized [M,4].

        Returns:
          (tensor) iou, sized [N,M].

        Reference:
          https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
        """
        if boxlist1.size != boxlist2.size:
            raise RuntimeError(
                "boxlists should have same image size, got {}, {}".format(boxlist1, boxlist2))

        N = len(boxlist1)
        M = len(boxlist2)

        area1 = self.area(boxlist1)
        area2 = self.area(boxlist2)

        box1, box2 = boxlist1, boxlist2

        lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

        TO_REMOVE = 1

        wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
        inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

        iou = inter / (area1[:, None] + area2 - inter)
        return iou

    def resize(self,box,size,size_img):
        """
        Returns a resized copy of this bounding box

        :param size: The requested size in pixels, as a 2-tuple:
            (width, height).
        """

        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, size_img))

        ratio_width, ratio_height = ratios
        xmin, ymin, xmax, ymax = box.split(1, dim=-1)
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        scaled_box = torch.cat(
            (scaled_xmin, scaled_ymin, scaled_xmax, scaled_ymax), dim=-1
        )


        return scaled_box


    def _forward_test(self, box_regression, centerness, anchors,
                      box_cls=None,
                      token_logits=None,
                      dot_product_logits=None,
                      positive_map=None,
                      fused_visual_features=None,
                      img_backbone_features=None,
                      targets=None,
                      positive_map_label_to_token_rel=None,
                      embeddings_rel=None,
                      train_ov_relation=False,
                      text_masks_rel=None,
                      eot_indices=None,
                      all_embeddings=None,
                      all_mask=None,
                      lang_net=None,
                      tokenize=None,
                      prefix=None,
                      use_CLIPtext=False,
                      captions=None
                      ):

        boxes = self.box_selector_test(box_regression, centerness, anchors,
                                       box_cls,
                                       token_logits,
                                       dot_product_logits,
                                       positive_map,
                                       fused_visual_features,
                                       sgg_mode=self.cfg.MODEL.DYHEAD.SGG_MODE,
                                       targets=targets,
                                       positive_map_label_to_token_rel=positive_map_label_to_token_rel)

        if self.cfg.MODEL.DYHEAD.RELATION_CONSISTENCY_ON:##########rel predictor##################
            if self.cfg.MODEL.DYHEAD.RELATION_REP_REFINER:
                memory_inputs = []
                for l_feat in img_backbone_features:
                    B, C, H, W = l_feat.shape
                    pos = self.pos_encoding(l_feat.permute(0,2,3,1)).permute(0,3,1,2)
                    memory_inputs.append((l_feat + 0.1 * pos).view(B, C, -1).permute(0, 2, 1))
                memory_inputs = torch.cat(memory_inputs, dim=1)

            device = boxes[0].bbox.device

            ###############use token predict logit##################################
            if train_ov_relation:
                aaaa=0
                #embedding_text = self.projection_text(embedding_rel / 2.0)
                #embedding_text = self.projection_text_rel(embedding_rel / 2.0)
                #embedding_text_bias = torch.matmul(embedding_rel, self.bias_lang) + self.bias0
            #################################################



            for img_id, boxes_per_img in enumerate(boxes):
                # prepare test object pairs
                if self.generate_text_object:
                    input_visual_feat = boxes_per_img.get_field('box_features').unsqueeze(1)
                    atts_t5 = torch.ones(input_visual_feat.size()[:-1], dtype=torch.long).to(
                        input_visual_feat.device)
                    text_decoder_output = self.class_generate.text_decoder(
                        {'object_features': input_visual_feat, 'atts_t5': atts_t5}, num_beams=self.num_beams)

                    if self.use_CLIPtext_object:
                        all_object_captions_text = captions[img_id].split('. ')
                        text = [clip.tokenize([pred_object]) for pred_object in all_object_captions_text]  # tokenize
                        text = torch.stack(text, dim=0).squeeze(1).to(device)  # num_boxes
                        text_features = self.clip_model.encode_text(text)  # pretrain encode
                        text_features = text_features / text_features.norm(dim=1, keepdim=True)  ###text_feature norm
                        all_embeddings_objects = text_features.to(torch.float32)
                        #all_embeddings_objects = self.projection_text_rel(all_embeddings_objects / 2.0)


                    obj_scores,obj_labels,obj_logit = self.encode_text(text_decoder_output, tokenize=tokenize,
                                                         lang_net=lang_net,
                                                         prefix=prefix,
                                                         visual_input=boxes_per_img.get_field('box_features'),
                                                         target_emb=all_embeddings_objects,
                                                         only_zR=self.onlyzR,
                                                         beams=self.num_beams,
                                                         use_CLIPtext_object=self.use_CLIPtext_object)

                    bbox_feat=boxes_per_img.get_field('box_features')
                    bbox_boxes = boxes_per_img.bbox

                    obj_scores,obj_labels,bbox_feat_topk,bbox_boxes_topk = self.topk_select(topk=self.topk_text_object,logit=obj_logit,feats=bbox_feat,boxes=bbox_boxes)
                else:
                    obj_scores, obj_labels = boxes_per_img.get_field('scores'), boxes_per_img.get_field('labels')


                if self.generate_text_object and self.topk_text_object>1:
                    box_num = len(obj_scores)
                else:
                    box_num = len(boxes_per_img)

                cand_matrix = torch.ones((box_num, box_num), device=device) - torch.eye(box_num, device=device)
                if self.cfg.MODEL.ROI_RELATION_HEAD.REQUIRE_BOX_OVERLAP:################
                    if self.generate_text_object and self.topk_text_object>1:
                        cand_matrix = cand_matrix.byte() & self.boxlist_iou(bbox_boxes_topk, bbox_boxes_topk).gt(0).byte()
                    else:
                        cand_matrix = cand_matrix.byte() & boxlist_iou(boxes_per_img, boxes_per_img).gt(0).byte()#######仅挑选有并集的框 gt(0):>0


                    # box_uids = boxes_per_img.get_field('per_box_loc')
                    # cand_matrix = cand_matrix & (box_uids[:, None] != box_uids[None, :]).byte() # remove same box pairs


                pair_ids = torch.nonzero(cand_matrix).view(-1, 2)

                # construct pair representations
                if self.generate_text_object and self.topk_text_object>1:
                    head_tail_reps = bbox_feat_topk[pair_ids]
                    pair_boxes = self.resize(size=(1,1),size_img=boxes_per_img.size,box=bbox_boxes_topk)[pair_ids]
                else:
                    head_tail_reps = boxes_per_img.get_field('box_features')[pair_ids]
                    pair_boxes = boxes_per_img.copy_with_fields([]).resize((1,1)).bbox[pair_ids]
                pair_reps = self.relation_feat_extractor(head_tail_reps, pair_boxes)###48 2 4
                #relateness = self.relation_structure_embed(pair_reps).squeeze()


                ###############use token predict logit##################################
                if train_ov_relation and len(pair_reps)>0:
                    if self.test_with_rel_token_logit:
                        if self.direct_head_tail:
                            head_reps = head_tail_reps[:, 0]
                            tail_reps = head_tail_reps[:, 1]
                            rel_reps= torch.cat((head_reps,tail_reps),dim=-1)
                            embedding_fuse = {"visual": rel_reps,
                                           "lang": embeddings_rel[img_id],
                                           "mask": text_masks_rel[img_id]}
                        else:
                            if self.useprompt:###################
                                feat_inputs = {"visual": pair_reps,
                                               "lang": all_embeddings,
                                               "masks": all_mask,
                                               "use_prompt": True,
                                               "single_feat": False,
                                               "eot_indices": eot_indices}
                                if not use_CLIPtext:
                                    embedding_fuse = self.dyhead_tower_L(feat_inputs)

                            else:
                                feat_inputs = {"visual": pair_reps,
                                               "lang": embeddings_rel[img_id],
                                               "masks": text_masks_rel[img_id],
                                                "use_prompt": False,
                                                "single_feat": True,
                                                "eot_indices": None}


                                embedding_fuse = self.fuse_VL(feat_inputs, single_feat=True)
                                embedding_fuse = self.fuse_L(embedding_fuse)
                        if not use_CLIPtext:
                            if self.useprompt:#######################
                                embedding_text_emb = embedding_fuse["lang"][
                                    torch.arange(embedding_fuse["lang"].shape[0]), eot_indices].squeeze(0)
                            else:
                                embedding_text_emb = embedding_fuse["lang"].squeeze(0)
                        else:
                            embedding_text_emb = all_embeddings

                        if not use_CLIPtext:
                            embedding_vis_emb = embedding_fuse["visual"].squeeze(0)
                        else:
                            embedding_vis_emb = feat_inputs["visual"].squeeze(0)


                        embedding_rel = F.normalize(embedding_text_emb, p=2, dim=-1)
                        if self.direct_head_tail:
                            embedding_text = self.projection_text(embedding_rel / 2.0)
                        else:
                            embedding_text = self.projection_text_rel(embedding_rel / 2.0)

                        if self.generate_text_logit:
                            input_visual_feat = embedding_vis_emb.unsqueeze(1)
                            atts_t5 = torch.ones(input_visual_feat.size()[:-1], dtype=torch.long).to(
                                input_visual_feat.device)
                            text_decoder_output = self.class_generate.text_decoder(
                                {'object_features': input_visual_feat, 'atts_t5': atts_t5}, num_beams=self.num_beams)

                            dot_product_logit = self.encode_text(text_decoder_output, tokenize=tokenize,
                                                                 lang_net=lang_net,
                                                                 prefix=prefix,
                                                                 visual_input=embedding_vis_emb,
                                                                 target_emb=embedding_text,
                                                                 only_zR=self.onlyzR,
                                                                 beams=self.num_beams)###0.7 和后面一样？
                        else:##### onlyzR predcls myeval 0.03
                            if self.onlyzR:
                                embedding_rel_ZR = embedding_rel[self.ov_relation_idx]
                                embedding_text_bias = torch.matmul(embedding_rel_ZR, self.bias_lang) + self.bias0
                            else:
                                embedding_text_bias = torch.matmul(embedding_rel, self.bias_lang) + self.bias0

                            A = pair_reps.shape[0]

                            # bias = embedding_text_bias[img_id].unsqueeze(0).repeat(A, 1)
                            # dot_product_logit = (torch.matmul(pair_reps, embedding_text[img_id].transpose(-1,-2)) / self.log_scale.exp()) + bias
                            bias = embedding_text_bias.unsqueeze(0).repeat(A, 1)
                            if self.onlyzR:
                                embedding_text_ZR=embedding_text[self.ov_relation_idx]
                                dot_product_logit = (torch.matmul(embedding_vis_emb, embedding_text_ZR.transpose(-1,
                                                                                                              -2)) / self.log_scale.exp()) + bias
                            else:########################
                                dot_product_logit = (torch.matmul(embedding_vis_emb, embedding_text.transpose(-1,-2)) / self.log_scale.exp()) + bias


                        dot_product_logit = torch.clamp(dot_product_logit, max=50000)
                        dot_product_logit = torch.clamp(dot_product_logit, min=-50000)
                        if self.use_sigmoid:
                            if self.use_softmax:
                                dot_product_logit = F.softmax(dot_product_logit, -1)
                            else:
                                dot_product_logit = dot_product_logit.sigmoid()

                        if self.useprompt:
                            rel_logit = dot_product_logit
                        else:
                            rel_logit = self.convert_token_to_rel_logits(dot_product_logit,positive_map_label_to_token_rel,score_agg="MEAN")
                        #rel_class_logit = F.softmax(rel_logit, -1)
                        rel_logit=torch.sqrt(rel_logit)


                        #####top-k search###########
                        if self.use_topk and not self.generate_text_sgdet:
                            sim_values, sim_indices = rel_logit.topk(self.topk_for_mapping)
                            num_pairs=pair_ids.shape[0]
                            pair_ids_new=pair_ids.unsqueeze(1).repeat(1,self.topk_for_mapping,1).view(self.topk_for_mapping*num_pairs,-1)
                            rel_scores_text_topk = sim_values.view(-1)
                            rel_class_text_topk = sim_indices.view(-1)
                            if self.onlyzR:
                                for i,rel_class_idx in enumerate(rel_class_text_topk):
                                    rel_class_text_topk[i]= self.ov_relation_idx[rel_class_idx]
                        ###############################
                        rel_scores_text, _ = rel_logit[:, :].max(dim=1)
                        # rel_class_text = rel_class_text + 2
                        background_logit=torch.zeros(len(rel_logit),1).to(rel_scores_text.device)
                        rel_scores_text_fin = torch.cat((background_logit, rel_logit), dim=-1)
                else:
                    print("###########################no_rels#############################")
                    rel_scores_text_fin = torch.zeros((1, 51)).to(cand_matrix.device)
                    if self.use_topk:
                        sim_values, sim_indices = rel_scores_text_fin.topk(1)
                        pair_ids_new = pair_ids
                        rel_scores_text_topk = sim_values.view(-1)
                        rel_class_text_topk = sim_indices.view(-1)
                #################################################










                boxes_per_img.add_field('pred_labels', obj_labels.clone()) # for eval
                boxes_per_img.add_field('pred_scores', obj_scores.clone())


                # if self.cfg.MODEL.DYHEAD.RELATION_REP_REFINER:
                #     _, resample_inds = (obj_scores[pair_ids].prod(-1) * relateness.sigmoid()).topk(min(100, len(relateness)))
                #     pair_ids = pair_ids[resample_inds]
                #     pair_reps = pair_reps[resample_inds]
                #     relateness = relateness[resample_inds]
                #     pair_reps_refine = self.relation_rep_refiner(pair_reps.unsqueeze(1), memory_inputs[img_id:img_id+1].permute(1,0,2))[:, 0]
                #

                #




                ###############use token predict logit##################################
                if train_ov_relation and self.test_with_rel_token_logit:
                    #rel_class_scores_union=1*rel_scores_text_fin+0.0*rel_class_prob
                    ###########################################
                    if self.use_topk and not self.generate_text_sgdet or not len(pair_reps)>0:
                        rel_scores_union_topk=rel_scores_text_topk
                        rel_class_union_topk=rel_class_text_topk+1

#########################################################
                    rel_class_scores_union = 1 * rel_scores_text_fin
                    rel_scores_union, rel_class_union = rel_class_scores_union[:, 1:].max(dim=1)
                    rel_class_union=rel_class_union+1
                else:
                    # relation post-process
                    # resample and pair rep refinement
                    pair_reps_refine = pair_reps
                    relation_logits = self.relation_semantic_embed(pair_reps_refine)
                    if self.cfg.MODEL.ROI_RELATION_HEAD.PREDICT_USE_BIAS:
                        relation_logits = relation_logits + self.relation_freq_bias.index_with_labels(obj_labels[pair_ids])
                    rel_class_prob = F.softmax(relation_logits, -1)
                    rel_scores, rel_class = rel_class_prob[:, 1:].max(dim=1)
                    rel_class = rel_class + 1
                #####################################################################

                if train_ov_relation and self.test_with_rel_token_logit:
                    if self.use_topk and not self.generate_text_sgdet or not len(pair_reps)>0:
                        triple_scores_topk = obj_scores[pair_ids_new].prod(-1) * rel_scores_union_topk
                    triple_scores = obj_scores[pair_ids].prod(-1) * rel_scores_union # * relateness.sigmoid()
                else:
                    triple_scores = obj_scores[pair_ids].prod(-1) * rel_scores  # * relateness.sigmoid()
                _, sorting_idx = torch.sort(triple_scores.view(-1), dim=0, descending=True)


                ################################################################################################
                if self.generate_text_sgdet and len(pair_reps)>0:#####myeval 14.29
                    input_visual_feat = embedding_vis_emb[sorting_idx[:self.num_of_relations]].unsqueeze(1)
                    atts_t5 = torch.ones(input_visual_feat.size()[:-1], dtype=torch.long).to(input_visual_feat.device)
                    text_decoder_output = self.class_generate.text_decoder({'object_features': input_visual_feat, 'atts_t5': atts_t5}, num_beams=self.num_beams)


                    extra_description={}########self.num_beams = 5 description比较多  text_len 3
                    for texts in text_decoder_output['pred_object_descriptions']:
                        if texts not in VG150_REL_CATEGORIES:
                            if texts not in extra_description:
                                extra_description[texts]=0
                            else:
                                extra_description[texts]=extra_description[texts]+1

                    generate_logit = self.encode_text(text_decoder_output, tokenize=tokenize, lang_net=lang_net,
                                                         prefix=prefix, visual_input=embedding_vis_emb[sorting_idx[:self.num_of_relations]],
                                                         target_emb=embedding_text,beams=self.num_beams,only_zR=self.onlyzR,use_CLIPtext_object=use_CLIPtext,generate_text=self.generate_text)
                    pair_ids_text = pair_ids[sorting_idx[:self.num_of_relations]]

                    generate_logit = torch.clamp(generate_logit, max=50000)
                    generate_logit = torch.clamp(generate_logit, min=-50000)
                    generate_logit = generate_logit.sigmoid()
                    generate_logit = torch.sqrt(generate_logit)

                    #all_boxes=boxes_per_img.bbox[pair_ids_text]
                    if self.generate_text_causal:
                        if self.generate_text_causal_test:
                            delta_logits = np.loadtxt(self.load_prior, delimiter=',')
                            delta_logits = torch.Tensor(delta_logits).cuda()
                            all_logits = torch.zeros(30).cuda()
                            if self.iter_test == 0:
                                print("############################loadprior##############################################")
                                print("load prior:", self.load_prior)
                                print("####################################################################################")
                            for idx,all_id in enumerate(self.rel_idx):
                                all_logits[all_id] = delta_logits[idx]
                            generate_logit = generate_logit - self.causal_weight_test_gen * all_logits
                            self.iter_test+=1







                    if self.use_topk_text:###############
                        text_score_values, text_score_indices = generate_logit.topk(self.topk_text)
                        num_text_pairs = pair_ids_text.shape[0]
                        text_pair_ids_new = pair_ids_text.unsqueeze(1).repeat(1, self.topk_text, 1).view(
                            self.topk_text * num_text_pairs, -1)

                        decoder_out=text_decoder_output['pred_object_descriptions']
                        all_decode=[]
                        for beams in range(self.num_of_relations):
                            all_decode.append([])

                        for rel_idx in range(self.num_of_relations):
                            for beam_idx in range(self.num_beams):
                                gen_index = rel_idx * self.num_beams + beam_idx
                                #print("gen_index",gen_index)
                                #print("rel_idx", rel_idx)
                                #print("beam_idx", beam_idx)
                                #print("all_decode", all_decode)
                                # print("decoder_out", decoder_out)
                                all_decode[rel_idx].append(decoder_out[gen_index])

                        #decoder_out = [out for out in decoder_out for top_times in range(self.topk_text)]


                        rel_scores_union_topk = text_score_values.view(-1)
                        rel_class_union_topk = text_score_indices.view(-1)
                        if self.onlyzR:
                            for i, rel_class_idx in enumerate(rel_class_union_topk):
                                rel_class_union_topk[i] = self.ov_relation_idx[rel_class_idx]
                        rel_class_union_topk=rel_class_union_topk+1
                        triple_scores_topk = obj_scores[text_pair_ids_new].prod(-1) * rel_scores_union_topk
                    else:
                        if self.rel_nms:
                            #################################################
                            def rel_nms(pred_boxes, pred_classes, pred_rel_inds, rel_scores, nms_thresh=0.5):
                                ###########pred_boxes-N*4;pred_classes-N;pred_rel_inds-N*(N-1)*2;rel_scores-logits;
                                ious = boxlist_iou(pred_boxes, pred_boxes)  # rel_scores N*N-N 所有可能的pair的logit
                                sub_ious = ious[pred_rel_inds[:, 0]][:,
                                           pred_rel_inds[:, 0]]  # pred_rel_inds N*N-N 所有可能的pair
                                obj_ious = ious[pred_rel_inds[:, 1]][:, pred_rel_inds[:, 1]]
                                rel_ious = torch.min(sub_ious, obj_ious)  # 取关系中相对较小的IOU
                                sub_labels = pred_classes[pred_rel_inds[:, 0]]
                                obj_labels = pred_classes[pred_rel_inds[:, 1]]
                                # N*N-N 1 C 和 1 N*N-N C;isoverlap:SO相同，IOU高，但predlogits不一样
                                is_overlap = (rel_ious >= nms_thresh) & (sub_labels[:, None] == sub_labels[None, :]) & (
                                        obj_labels[:, None] == obj_labels[None, :])
                                is_overlap = is_overlap[:, :, None].repeat(1, 1, rel_scores.shape[1])

                                rel_scores_cp = rel_scores.clone()
                                pred_rels = torch.zeros(rel_scores_cp.shape[0],dtype=torch.long)

                                for i in range(rel_scores_cp.shape[0]):
                                    box_ind = rel_scores_cp.topk(1)[0].argmax()
                                    cls_ind = rel_scores_cp.topk(1)[1][box_ind]
                                    if float(pred_rels[int(box_ind)]) > 0:  # 玄学情况——一般不会有
                                        pass
                                    else:
                                        pred_rels[int(box_ind)] = int(cls_ind)  # 取最大logit对应为pred
                                    rel_scores_cp[is_overlap[box_ind, :, cls_ind].squeeze(), cls_ind] = 0.0
                                    rel_scores_cp[box_ind] = -1.

                                return pred_rels, rel_scores[torch.arange(pred_rels.shape[0]), pred_rels]

                            pred_rels, pred_scores = rel_nms(boxes_per_img, boxes_per_img.extra_fields['pred_labels'],
                                                             pair_ids_text, generate_logit, 0.5)

                            rel_scores_union_topk = pred_scores
                            rel_class_union_topk = pred_rels + 1
                            triple_scores_topk = obj_scores[pair_ids_text].prod(-1) * rel_scores_union_topk
                            #####################################################
                        else:
                            rel_scores_union_topk, rel_class_union_topk = generate_logit.max(dim=1)####0.7
                            rel_class_union_topk=rel_class_union_topk+1
                            triple_scores_topk = obj_scores[pair_ids_text].prod(-1) * rel_scores_union_topk####0.3


                    sorted_score, sorting_idx_topk = torch.sort(triple_scores_topk.view(-1), dim=0, descending=True)

                    if self.use_topk_text:
                        rel_pair_idx_topk = text_pair_ids_new[sorting_idx_topk]
                        # decoder_out_fin = [all_decode[sort_id] for sort_id in sorting_idx_topk]
                    else:
                        rel_pair_idx_topk = pair_ids_text[sorting_idx_topk]
                #########################################################################################






                if self.use_topk and not self.generate_text_sgdet or not len(pair_reps)>0:
                    _, sorting_idx_topk = torch.sort(triple_scores_topk.view(-1), dim=0, descending=True)
                    rel_pair_idx_topk = pair_ids_new[sorting_idx_topk]
                rel_pair_idx = pair_ids[sorting_idx]



                ###############use token predict logit##################################
                if train_ov_relation and self.test_with_rel_token_logit:
                    if self.use_topk:
                        rel_class_scores_union_topk = rel_scores_union_topk[sorting_idx_topk]
                        rel_class_topk = rel_class_union_topk[sorting_idx_topk]
                    rel_class_scores_union = rel_class_scores_union[sorting_idx]
                    rel_class_union = rel_class_union[sorting_idx]
                else:
                    rel_class_prob = rel_class_prob[sorting_idx]
                    rel_labels = rel_class[sorting_idx]

                #########################################################################



                boxes_per_img.add_field('rel_pair_idxs', rel_pair_idx) # (#rel, 2)
                if train_ov_relation and self.test_with_rel_token_logit:
                    if self.generate_text_object and self.topk_text_object>1:
                        boxes_per_img.add_field('topk_boxes', bbox_boxes_topk)

                    if self.use_topk:
                        #if self.use_topk_text:
                            #boxes_per_img.add_field('decoder_output_topk', decoder_out_fin[:self.topk_relation])
                        boxes_per_img.add_field('rel_pair_idxs_topk', rel_pair_idx_topk[:self.topk_relation])
                        boxes_per_img.add_field('pred_rel_cls_topk', rel_class_topk[:self.topk_relation])
                        boxes_per_img.add_field('pred_rel_scores_topk', rel_class_scores_union_topk[:self.topk_relation])
                    boxes_per_img.add_field('pred_rel_scores', rel_class_scores_union) # (#rel, #rel_class)
                    boxes_per_img.add_field('pred_rel_labels', rel_class_union)
                else:
                    boxes_per_img.add_field('pred_rel_scores', rel_class_prob) # (#rel, #rel_class)
                    boxes_per_img.add_field('pred_rel_labels', rel_labels)

        return boxes, {}, fused_visual_features

    def _relation_structure_consistency_loss(self, inputs, targets, gamma=2):
        probs = inputs.sigmoid()
        pos_inds = targets.eq(1).float()
        neg_inds = targets.lt(1).float()
        pos_loss = torch.log(probs) * torch.pow(1 - probs, gamma) * pos_inds
        neg_loss = torch.log(1 - probs) * torch.pow(probs, gamma) * neg_inds
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        # normalize
        num_pos = pos_inds.float().sum()
        if num_pos == 0:
            loss = -neg_loss
        else:
            loss = -(pos_loss + neg_loss) / num_pos
        return loss

class RelationFeatureExtractor(nn.Module):
    def __init__(self, cfg, dim=256):
        super(RelationFeatureExtractor, self).__init__()
        self.cfg = cfg

        # head-tail fusion
        self.diff_fusion = nn.Sequential(
            make_fc(dim, dim), nn.ReLU(),
            make_fc(dim, dim)
        )
        self.sum_fusion = nn.Sequential(
            make_fc(dim, dim), nn.ReLU(),
            make_fc(dim, dim)
        )

        # spatial feature
        spatial_in_dim = 16
        self.spatial_proj = make_fc(spatial_in_dim, dim)

        # fusion
        self.fusion_fc = nn.Sequential(
            make_fc(dim*2, dim), nn.ReLU(),
            make_fc(dim, dim)
        )

    def forward(self, head_tail_reps, pair_boxes_xyxy):
        head_reps = head_tail_reps[:, 0]
        tail_reps = head_tail_reps[:, 1]
        rel_embed_reps = self.diff_fusion(head_reps - tail_reps) + self.sum_fusion(head_reps + tail_reps)#s-o +  s+o

        # spatial features
        head_boxes = pair_boxes_xyxy[:, 0]
        tail_boxes = pair_boxes_xyxy[:, 1]
        rel_spatial_feats = self.spatial_proj(
            torch.cat([head_boxes, tail_boxes, self.extract_spatial_layout_feats(head_boxes, tail_boxes)], dim=-1)
        )

        rel_reps = self.fusion_fc(torch.cat([rel_embed_reps, rel_spatial_feats], dim=-1))
        return rel_reps

    def extract_spatial_layout_feats(self, head_boxes, tail_boxes):
        head_center = torch.stack([(head_boxes[:, 0] + head_boxes[:, 2]) / 2, (head_boxes[:, 1] + head_boxes[:, 3]) / 2], dim=1)
        tail_center = torch.stack([(tail_boxes[:, 0] + tail_boxes[:, 2]) / 2, (tail_boxes[:, 1] + tail_boxes[:, 3]) / 2], dim=1)
        dxdy = head_center - tail_center # distances
        theta = (torch.atan2(dxdy[..., 1], dxdy[..., 0]) / np.pi).unsqueeze(-1)
        dis = dxdy.norm(dim=-1, keepdim=True)

        # overlap and union
        intersec_lt = torch.max(head_boxes[...,:2], tail_boxes[...,:2])
        intersec_rb = torch.min(head_boxes[...,2:], tail_boxes[...,2:])
        overlap = (intersec_rb - intersec_lt).clamp(min=0).prod(dim=-1, keepdim=True)

        union_lt = torch.min(head_boxes[...,:2], tail_boxes[...,:2])
        union_rb = torch.max(head_boxes[...,2:], tail_boxes[...,2:])
        union = (union_rb - union_lt).clamp(min=0).prod(dim=-1, keepdim=True)

        # areas
        head_area = (head_boxes[:, 2:] - head_boxes[:, :2]).prod(dim=1, keepdim=True)
        tail_area = (tail_boxes[:, 2:] - tail_boxes[:, :2]).prod(dim=1, keepdim=True)

        spatial_interaction_feats = torch.cat([
            dxdy, dis, theta, # dx, dy, distance, theta
            overlap, union, head_area, tail_area # overlap, union, subj, obj
        ], dim=-1)
        return spatial_interaction_feats

class FrequencyBias(nn.Module):
    """
    The goal of this is to provide a simplified way of computing
    P(predicate | obj1, obj2, img).
    """

    def __init__(self, cfg, statistics, eps=1e-3):
        super(FrequencyBias, self).__init__()
        pred_dist = statistics['pred_dist'].float()
        assert pred_dist.size(0) == pred_dist.size(1)

        self.num_objs = pred_dist.size(0)
        self.num_rels = pred_dist.size(2)
        pred_dist = pred_dist.view(-1, self.num_rels)

        self.obj_baseline = nn.Embedding(self.num_objs*self.num_objs, self.num_rels)
        with torch.no_grad():
            self.obj_baseline.weight.copy_(pred_dist, non_blocking=True)

    def index_with_labels(self, labels):
        """
        :param labels: [batch_size, 2]
        :return:
        """
        return self.obj_baseline(labels[:, 0] * self.num_objs + labels[:, 1])

    def index_with_probability(self, pair_prob):
        """
        :param labels: [batch_size, num_obj, 2]
        :return:
        """
        batch_size, num_obj, _ = pair_prob.shape

        joint_prob = pair_prob[:,:,0].contiguous().view(batch_size, num_obj, 1) * pair_prob[:,:,1].contiguous().view(batch_size, 1, num_obj)

        return joint_prob.view(batch_size, num_obj*num_obj)  @ self.obj_baseline.weight

    def forward(self, labels):
        # implement through index_with_labels
        return self.index_with_labels(labels)

class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise "The input tensor has to be 4d!"

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = self._get_emb(sin_inp_x).unsqueeze(1)
        emb_y = self._get_emb(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc

    def _get_emb(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
        return torch.flatten(emb, -2, -1)
