import math
import copy
import logging
from typing import Optional, List
from collections import OrderedDict
from functools import partial

import numpy
import torch
from torch import nn, Tensor
import torch.nn.functional as F
import torch.nn as nn
import numpy as np

from .registry import register_model
from .resnet import resnet26d, resnet50d, resnet18d, resnet26, resnet50, resnet101d, resnet34d
from .layers import StdConv2dSame, StdConv2d, to_2tuple
from .pointpillar import PointPillarNet, ConvBackbone

_logger = logging.getLogger(__name__)


class HybridEmbed(nn.Module):
    def __init__(
        self,
        backbone,
        img_size=224,
        patch_size=1,
        feature_size=None,
        in_chans=3,
        embed_dim=768,
    ):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                backbone.cuda()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]).cuda())
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, "feature_info"):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features

        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x)
        global_x = torch.mean(x, [2, 3], keepdim=False)[:, :, None]
        return x, global_x


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """

    def __init__(
        self, num_pos_feats=64, temperature=10000, normalize=False, scale=None
    ):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor):
        x = tensor
        bs, _, h, w = x.shape
        not_mask = torch.ones((bs, h, w), device=x.device)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        output = src

        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )

        if self.norm is not None:
            output = self.norm(output)

        return output


class SpatialSoftmax(nn.Module):
    def __init__(self, height, width, channel, temperature=None, data_format="NCHW"):
        super().__init__()

        self.data_format = data_format
        self.height = height
        self.width = width
        self.channel = channel

        if temperature:
            self.temperature = Parameter(torch.ones(1) * temperature)
        else:
            self.temperature = 1.0

        pos_x, pos_y = np.meshgrid(
            np.linspace(-1.0, 1.0, self.height), np.linspace(-1.0, 1.0, self.width)
        )
        pos_x = torch.from_numpy(pos_x.reshape(self.height * self.width)).float()
        pos_y = torch.from_numpy(pos_y.reshape(self.height * self.width)).float()
        self.register_buffer("pos_x", pos_x)
        self.register_buffer("pos_y", pos_y)

    def forward(self, feature):
        # Output:
        #   (N, C*2) x_0 y_0 ...

        if self.data_format == "NHWC":
            feature = (
                feature.transpose(1, 3)
                .tranpose(2, 3)
                .view(-1, self.height * self.width)
            )
        else:
            feature = feature.view(-1, self.height * self.width)

        weight = F.softmax(feature / self.temperature, dim=-1)
        expected_x = torch.sum(
            torch.autograd.Variable(self.pos_x) * weight, dim=1, keepdim=True
        )
        expected_y = torch.sum(
            torch.autograd.Variable(self.pos_y) * weight, dim=1, keepdim=True
        )
        expected_xy = torch.cat([expected_x, expected_y], 1)
        feature_keypoints = expected_xy.view(-1, self.channel, 2)
        feature_keypoints[:, :, 1] = (feature_keypoints[:, :, 1] - 1) * 12
        feature_keypoints[:, :, 0] = feature_keypoints[:, :, 0] * 12
        return feature_keypoints

class GRUWaypointsPredictor(nn.Module):
    def __init__(self, input_dim, waypoints=5):
        super().__init__()
        self.gru = torch.nn.GRU(input_size=input_dim, hidden_size=64, batch_first=True)
        self.encoder = nn.Linear(2, 64)
        self.decoder = nn.Linear(64, 2)
        self.waypoints = waypoints

    def forward(self, x, target_point):
        bs = x.shape[0]
        z = self.encoder(target_point).unsqueeze(0)
        output, _ = self.gru(x, z)
        output = output.reshape(bs * self.waypoints, -1)
        output = self.decoder(output).reshape(bs, self.waypoints, 2)
        output = torch.cumsum(output, 1)
        return output


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_layer, num_layers, norm=None, return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        output = tgt

        intermediate = []

        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            return torch.stack(intermediate)

        return output.unsqueeze(0)


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU(),
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    def forward(
        self,
        src,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        activation=nn.ReLU(),
        normalize_before=False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation()
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward_pre(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2, attn_mask=tgt_mask, key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt

    def forward(
        self,
        tgt,
        memory,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ):
        if self.normalize_before:
            return self.forward_pre(
                tgt,
                memory,
                tgt_mask,
                memory_mask,
                tgt_key_padding_mask,
                memory_key_padding_mask,
                pos,
                query_pos,
            )
        return self.forward_post(
            tgt,
            memory,
            tgt_mask,
            memory_mask,
            tgt_key_padding_mask,
            memory_key_padding_mask,
            pos,
            query_pos,
        )


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


class LidarModel(nn.Module):
    def __init__(self, num_input=9, num_features=[32,32],
        backbone='conv',
        min_x=-20, max_x=30,
        min_y=-25, max_y=25,
        pixels_per_meter=4,
        output_features=256):

        super().__init__()

        self.point_pillar_net = PointPillarNet(
            num_input, num_features,
            min_x=min_x, max_x=max_x,
            min_y=min_y, max_y=max_y,
            pixels_per_meter=pixels_per_meter,
        )

        num_feature = num_features[-1]
        self.backbone = ConvBackbone(num_feature=num_feature)
        self.reduce_size = nn.Conv2d(6*num_feature, output_features, 2, 2)

    def forward(self, lidars, num_points):
        features = self.point_pillar_net(lidars, num_points)
        features = self.backbone(features)
        features = self.reduce_size(features)
        features = features[:, :, 5:55, 5:55]
        return features


class Memfuser(nn.Module):
    def __init__(
        self,
        img_size=224,
        multi_view_img_size=112,
        patch_size=16,
        in_chans=3,
        embed_dim=768,
        enc_depth=6,
        dec_depth=6,
        dim_feedforward=2048,
        normalize_before=False,
        rgb_backbone_name="r26",
        lidar_backbone_name="r26",
        num_heads=8,
        norm_layer=None,
        dropout=0.1,
        separate_view_attention=False,
        separate_all_attention=False,
        act_layer=None,
        weight_init="",
        freeze_num=-1,
        with_lidar=True,
        with_right_left_sensors=True,
        with_rear_sensor=True,
        with_center_sensor=True,
        traffic_pred_head_type="det",
        waypoints_pred_head="heatmap",
        reverse_pos=True,
        use_view_embed=True,
        use_mmad_pretrain=None,
        return_feature=False,
    ):
        super().__init__()
        self.traffic_pred_head_type = traffic_pred_head_type
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.reverse_pos = reverse_pos
        self.waypoints_pred_head = waypoints_pred_head
        self.with_lidar = with_lidar
        self.with_right_left_sensors = with_right_left_sensors
        self.with_rear_sensor = with_rear_sensor
        self.with_center_sensor = with_center_sensor

        self.separate_view_attention = separate_view_attention
        self.separate_all_attention = separate_all_attention
        self.use_view_embed = use_view_embed
        self.return_feature = return_feature

        self.attn_mask = None

        if rgb_backbone_name == "r50":
            self.rgb_backbone = resnet50d(
                pretrained=True,
                in_chans=in_chans,
                features_only=True,
                out_indices=[4],
            )
        elif rgb_backbone_name == "r26":
            self.rgb_backbone = resnet26d(
                pretrained=True,
                in_chans=in_chans,
                features_only=True,
                out_indices=[4],
            )
        elif rgb_backbone_name == "r18":
            self.rgb_backbone = resnet18d(
                pretrained=True,
                in_chans=in_chans,
                features_only=True,
                out_indices=[4],
            )

        self.lidar_backbone = LidarModel(num_input=9, num_features=[32,32],
                                         backbone='conv', min_x=-25, max_x=35,
                                         min_y=-30, max_y=30, pixels_per_meter=4,
                                         output_features=embed_dim)

        rgb_embed_layer = partial(HybridEmbed, backbone=self.rgb_backbone)

        if use_mmad_pretrain:
            params = torch.load(use_mmad_pretrain)["state_dict"]
            updated_params = OrderedDict()
            for key in params:
                if "backbone" in key:
                    updated_params[key.replace("backbone.", "")] = params[key]
            self.rgb_backbone.load_state_dict(updated_params)

        self.rgb_patch_embed = rgb_embed_layer(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )

        self.global_embed = nn.Parameter(torch.zeros(1, embed_dim, 6))
        self.view_embed = nn.Parameter(torch.zeros(1, embed_dim, 5, 1))

        self.query_pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 6))
        self.query_embed = nn.Parameter(torch.zeros(6, 1, embed_dim))

        self.waypoints_generator = GRUWaypointsPredictor(embed_dim)

        self.traffic_light_pred_head = nn.Linear(embed_dim, 2)
        self.stop_sign_head = nn.Linear(embed_dim, 2)

        if self.traffic_pred_head_type == "det":
            self.traffic_pred_head = nn.Sequential(
                *[
                    nn.Linear(embed_dim + 32, 64),
                    nn.ReLU(),
                    nn.Linear(64, 8),
                ]
            )
        elif self.traffic_pred_head_type == "seg":
            self.traffic_pred_head = nn.Sequential(
                *[nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()]
            )

        self.position_encoding = PositionEmbeddingSine(embed_dim // 2, normalize=True)

        encoder_layer = TransformerEncoderLayer(
            embed_dim, num_heads, dim_feedforward, dropout, act_layer, normalize_before
        )
        self.encoder = TransformerEncoder(encoder_layer, enc_depth, None)

        decoder_layer = TransformerDecoderLayer(
            embed_dim, num_heads, dim_feedforward, dropout, act_layer, normalize_before
        )
        decoder_norm = nn.LayerNorm(embed_dim)
        self.decoder = TransformerDecoder(
            decoder_layer, dec_depth, decoder_norm, return_intermediate=False
        )
        self.velocity_fc = nn.Linear(1, embed_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.global_embed)
        nn.init.uniform_(self.view_embed)
        nn.init.uniform_(self.query_embed)
        nn.init.uniform_(self.query_pos_embed)

    def forward_features(
        self,
        front_image,
        left_image,
        right_image,
        rear_image,
        front_center_image,
        lidar,
        num_points,
    ):
        features = []

        # Front view processing
        front_image_token, front_image_token_global = self.rgb_patch_embed(front_image)
        if self.use_view_embed:
            front_image_token = (
                front_image_token
                + self.view_embed[:, :, 0:1, :]
                + self.position_encoding(front_image_token)
            )
        else:
            front_image_token = front_image_token + self.position_encoding(
                front_image_token
            )
        front_image_token = front_image_token.flatten(2).permute(2, 0, 1)
        front_image_token_global = (
            front_image_token_global
            + self.view_embed[:, :, 0, :]
            + self.global_embed[:, :, 0:1]
        )
        front_image_token_global = front_image_token_global.permute(2, 0, 1)
        features.extend([front_image_token, front_image_token_global])

        if self.with_right_left_sensors:
            # Left view processing
            left_image_token, left_image_token_global = self.rgb_patch_embed(left_image)
            if self.use_view_embed:
                left_image_token = (
                    left_image_token
                    + self.view_embed[:, :, 1:2, :]
                    + self.position_encoding(left_image_token)
                )
            else:
                left_image_token = left_image_token + self.position_encoding(
                    left_image_token
                )
            left_image_token = left_image_token.flatten(2).permute(2, 0, 1)
            left_image_token_global = (
                left_image_token_global
                + self.view_embed[:, :, 1, :]
                + self.global_embed[:, :, 1:2]
            )
            left_image_token_global = left_image_token_global.permute(2, 0, 1)

            # Right view processing
            right_image_token, right_image_token_global = self.rgb_patch_embed(
                right_image
            )
            if self.use_view_embed:
                right_image_token = (
                    right_image_token
                    + self.view_embed[:, :, 2:3, :]
                    + self.position_encoding(right_image_token)
                )
            else:
                right_image_token = right_image_token + self.position_encoding(
                    right_image_token
                )
            right_image_token = right_image_token.flatten(2).permute(2, 0, 1)
            right_image_token_global = (
                right_image_token_global
                + self.view_embed[:, :, 2, :]
                + self.global_embed[:, :, 2:3]
            )
            right_image_token_global = right_image_token_global.permute(2, 0, 1)

            features.extend(
                [
                    left_image_token,
                    left_image_token_global,
                    right_image_token,
                    right_image_token_global,
                ]
            )

        if self.with_center_sensor:
            # Front center view processing
            (
                front_center_image_token,
                front_center_image_token_global,
            ) = self.rgb_patch_embed(front_center_image)
            if self.use_view_embed:
                front_center_image_token = (
                    front_center_image_token
                    + self.view_embed[:, :, 3:4, :]
                    + self.position_encoding(front_center_image_token)
                )
            else:
                front_center_image_token = (
                    front_center_image_token
                    + self.position_encoding(front_center_image_token)
                )

            front_center_image_token = front_center_image_token.flatten(2).permute(
                2, 0, 1
            )
            front_center_image_token_global = (
                front_center_image_token_global
                + self.view_embed[:, :, 3, :]
                + self.global_embed[:, :, 3:4]
            )
            front_center_image_token_global = front_center_image_token_global.permute(
                2, 0, 1
            )
            features.extend([front_center_image_token, front_center_image_token_global])


        if self.with_rear_sensor:
            # Rear view processing
            (
                rear_image_token,
                rear_image_token_global,
            ) = self.rgb_patch_embed(rear_image)
            if self.use_view_embed:
                rear_image_token = (
                    rear_image_token
                    + self.view_embed[:, :, 4:5, :]
                    + self.position_encoding(rear_image_token)
                )
            else:
                rear_image_token = (
                    rear_image_token
                    + self.position_encoding(rear_image_token)
                )

            rear_image_token = rear_image_token.flatten(2).permute(
                2, 0, 1
            )
            rear_image_token_global = (
                rear_image_token_global
                + self.view_embed[:, :, 4, :]
                + self.global_embed[:, :, 5:6]
            )
            rear_image_token_global = rear_image_token_global.permute(
                2, 0, 1
            )
            features.extend([rear_image_token, rear_image_token_global])

        lidar_token = self.lidar_backbone(lidar, num_points) # Batchsize * embed_dim * 50 * 50
        lidar_token = (
            lidar_token
            + self.position_encoding(lidar_token)
        )
        lidar_token = lidar_token.flatten(2).permute(2, 0, 1)

        features = torch.cat(features, 0)
        return features, lidar_token

    def forward(self, x):
        front_image = x["rgb_front"]
        left_image = x["rgb_left"]
        right_image = x["rgb_right"]
        rear_image = x["rgb_rear"]
        front_center_image = x["rgb_center"]
        lidar = x["lidar"]
        num_points = x['num_points']
        if not self.return_feature:
            velocity = x['velocity'].view(1, -1, 1)
            target_point = x["target_point"]
            velocity_feature = self.velocity_fc(velocity)
            velocity_feature = velocity_feature.repeat(6, 1, 1)
        else:
            velocity = x['velocity']
            velocity = velocity.view(1, -1, 1)
            velocity_feature = self.velocity_fc(velocity)
            velocity_feature = velocity_feature.repeat(6, 1, 1)

        features, lidar_token = self.forward_features(
            front_image,
            left_image,
            right_image,
            rear_image,
            front_center_image,
            lidar,
            num_points,
        )

        bs = front_image.shape[0]

        tgt = self.position_encoding(
            torch.ones((bs, 1, 50, 50), device=x["rgb_front"].device)
        )
        tgt = tgt.flatten(2)
        tgt = torch.cat([tgt, self.query_pos_embed.repeat(bs, 1, 1)], 2)
        tgt = tgt.permute(2, 0, 1)

        memory = self.encoder(features, mask=self.attn_mask)

        query_embed = self.query_embed.repeat(1, bs, 1)
        query_embed = query_embed + velocity_feature

        query = torch.cat([lidar_token, query_embed], 0)
        hs = self.decoder(query, memory, query_pos=tgt)[0]

        hs = hs.permute(1, 0, 2)  # Batchsize ,  N, C

        traffic_feature = hs[:, :2500]
        traffic_light_state_feature = hs[:, 2500]
        stop_sign_feature = hs[:, 2500]
        waypoints_feature = hs[:, 2501:2506]
        if self.return_feature:
            traffic_feature = traffic_feature.reshape(bs, 50, 50, -1).permute(0, 3, 1, 2)
            traffic_feature = F.adaptive_avg_pool2d(traffic_feature, (10, 10)).view(bs, -1, 100).permute(0, 2, 1)
            return torch.cat([traffic_feature, traffic_light_state_feature.view(bs, 1, -1), waypoints_feature], 1)
            return waypoints_feature[:, :5]

        if self.waypoints_pred_head == "gru":
            waypoints = self.waypoints_generator(waypoints_feature, target_point)
        elif self.waypoints_pred_head == "gru-command":
            waypoints = self.waypoints_generator(waypoints_feature, target_point, measurements)

        traffic_light_state = self.traffic_light_pred_head(traffic_light_state_feature)
        stop_sign = self.stop_sign_head(stop_sign_feature)

        velocity = velocity.view(-1, 1, 1)
        velocity = velocity.repeat(1, 2500, 32)
        traffic_feature_with_vel = torch.cat([traffic_feature, velocity], dim=2)
        traffic = self.traffic_pred_head(traffic_feature_with_vel)
        return traffic, waypoints, traffic_light_state, stop_sign, traffic_feature


@register_model
def memfuser_baseline(**kwargs):
    model = Memfuser(
        enc_depth=2,
        dec_depth=4,
        embed_dim=256,
        rgb_backbone_name="r50",
        lidar_backbone_name="conv",
        waypoints_pred_head="gru",
    )
    return model

@register_model
def memfuser_baseline_return_feature(**kwargs):
    model = Memfuser(
        enc_depth=2,
        dec_depth=4,
        embed_dim=256,
        rgb_backbone_name="r50",
        lidar_backbone_name="conv",
        waypoints_pred_head="gru",
        return_feature=True,
    )
    return model

@register_model
def memfuser_baseline_e3d3(**kwargs):
    model = Memfuser(
        enc_depth=3,
        dec_depth=3,
        embed_dim=256,
        rgb_backbone_name="r50",
        lidar_backbone_name="conv",
        waypoints_pred_head="gru",
    )
    return model

@register_model
def memfuser_baseline_e1d3(**kwargs):
    model = Memfuser(
        enc_depth=1,
        dec_depth=3,
        embed_dim=256,
        rgb_backbone_name="r50",
        lidar_backbone_name="conv",
        waypoints_pred_head="gru",
    )
    return model

@register_model
def memfuser_baseline_e1d3(**kwargs):
    model = Memfuser(
        enc_depth=1,
        dec_depth=3,
        embed_dim=256,
        rgb_backbone_name="r50",
        lidar_backbone_name="conv",
        waypoints_pred_head="gru",
    )
    return model

@register_model
def memfuser_baseline_e1d3_return_feature(**kwargs):
    '''
    _return_feature means the model only output the intermediate features,
        and it does not need to take as inputs the target waypoint.
    '''
    model = Memfuser(
        enc_depth=1,
        dec_depth=3,
        embed_dim=256,
        rgb_backbone_name="r50",
        lidar_backbone_name="conv",
        waypoints_pred_head="gru",
        return_feature=True,
    )
    return model

@register_model
def memfuser_baseline_e1d3_r26(**kwargs):
    model = Memfuser(
        enc_depth=1,
        dec_depth=3,
        embed_dim=256,
        rgb_backbone_name="r26",
        lidar_backbone_name="conv",
        waypoints_pred_head="gru",
    )
    return model

@register_model
def memfuser_baseline_e1d3_r26_return_feature(**kwargs):
    '''
    _return_feature means the model only output the intermediate features,
        and it does not need to take as inputs the target waypoint.
    '''
    model = Memfuser(
        enc_depth=1,
        dec_depth=3,
        embed_dim=256,
        rgb_backbone_name="r26",
        lidar_backbone_name="conv",
        waypoints_pred_head="gru",
        return_feature=True,
    )
    return model

@register_model
def memfuser_baseline_e2d2(**kwargs):
    model = Memfuser(
        enc_depth=2,
        dec_depth=2,
        embed_dim=256,
        rgb_backbone_name="r50",
        lidar_backbone_name="conv",
        waypoints_pred_head="gru",
    )
    return model

