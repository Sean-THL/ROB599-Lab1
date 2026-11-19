"""Compact MinkowskiNet-style sparse U-Net implemented with spconv.

This is a student architecture, not the published MinkUNet34 or a wrapper
around a pretrained segmentation network. Convolutions come from spconv;
the residual blocks, encoder, decoder and classification head are defined here.
"""
from __future__ import annotations

import torch
from torch import nn
import spconv.pytorch as spconv


class SparseResidualBlock(nn.Module):
    """Two submanifold convolutions preserve voxel locations and row order."""

    def __init__(self, in_channels: int, out_channels: int, key: str):
        super().__init__()
        self.convs = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, padding=1,
                             bias=False, indice_key=key),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            spconv.SubMConv3d(out_channels, out_channels, 3, padding=1,
                             bias=False, indice_key=key),
            nn.BatchNorm1d(out_channels),
        )
        self.shortcut = (
            nn.Identity() if in_channels == out_channels else
            nn.Linear(in_channels, out_channels, bias=False)
        )

    def forward(self, x):
        residual = self.shortcut(x.features)
        out = self.convs(x)
        return out.replace_feature(torch.relu(out.features + residual))


class SparseUNet(nn.Module):
    """Residual encoder-decoder with a configurable number of resolutions.

    forward(features, coordinates, batch_size=None) -> logits (M, num_classes)

    features: float32 (M, in_channels), e.g. mean normalized RGB per voxel.
    coordinates: int32 (M, 4), columns [batch, x, y, z], on the same device.
        Coordinates must be unique, nonnegative voxel indices. Shift each
        scene's quantized coordinates to start at zero before batching.
    Output rows correspond to input coordinate rows. No softmax is applied.

    Voxelization, feature aggregation and point-to-voxel inverse mapping belong
    in the data pipeline. For point-level evaluation use logits[inverse].
    Use model.cuda() and CUDA inputs for the installed spconv CUDA backend.
    """

    def __init__(self, num_classes: int, in_channels: int = 3,
                 channels=(32, 64, 128, 256)):
        super().__init__()
        if num_classes < 1 or in_channels < 1:
            raise ValueError("num_classes and in_channels must be positive")
        channels = tuple(channels)
        if len(channels) < 2 or any(
            not isinstance(c, int) or isinstance(c, bool) or c < 1
            for c in channels
        ):
            raise ValueError("channels must contain at least two positive integers")
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.channels = channels
        self.from_pretrained = False
        self.stem = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, channels[0], 3, padding=1,
                             bias=False, indice_key="stem"),
            nn.BatchNorm1d(channels[0]),
            nn.ReLU(),
        )
        self.encoders = nn.ModuleList([
            SparseResidualBlock(c, c, f"enc_{i}")
            for i, c in enumerate(channels)
        ])
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i, (fine, coarse) in enumerate(zip(channels[:-1], channels[1:])):
            # Save each downsampling map for its matching inverse convolution.
            self.down.append(spconv.SparseSequential(
                spconv.SparseConv3d(fine, coarse, 2, stride=2,
                                   bias=False, indice_key=f"down_{i}"),
                nn.BatchNorm1d(coarse),
                nn.ReLU(),
            ))
            self.up.append(spconv.SparseSequential(
                spconv.SparseInverseConv3d(coarse, fine, 2, bias=False,
                                          indice_key=f"down_{i}"),
                nn.BatchNorm1d(fine),
                nn.ReLU(),
            ))
            self.decoders.append(
                SparseResidualBlock(2 * fine, fine, f"dec_{i}")
            )
        self.classifier = nn.Linear(channels[0], num_classes)

    def forward(self, features, coordinates, batch_size=None):
        if features.ndim != 2 or features.shape[1] != self.in_channels:
            raise ValueError(f"features must have shape (M, {self.in_channels})")
        if coordinates.ndim != 2 or coordinates.shape != (len(features), 4):
            raise ValueError("coordinates must have shape (M, 4)")
        if not len(features):
            raise ValueError("an input batch must contain occupied voxels")
        if coordinates.dtype != torch.int32:
            raise TypeError("coordinates must be int32 voxel indices")
        if not features.is_floating_point():
            raise TypeError("features must be floating point")
        if coordinates.device != features.device:
            raise ValueError("features and coordinates must share a device")
        if torch.any(coordinates < 0).item():
            raise ValueError("shift voxel coordinates to nonnegative indices")
        required_batch_size = int(coordinates[:, 0].max().item()) + 1
        if batch_size is None:
            batch_size = required_batch_size
        if batch_size < required_batch_size:
            raise ValueError("batch_size does not cover all batch indices")
        # A padded extent keeps every axis valid through all stride-2 stages,
        # including scenes that are thin along one axis.
        multiple = 2 ** (len(self.channels) - 1)
        extent = coordinates[:, 1:].max(dim=0).values.to(torch.int64) + 1
        spatial_shape = (
            ((extent + multiple - 1) // multiple) * multiple
        ).tolist()
        x = spconv.SparseConvTensor(
            features.contiguous(), coordinates.contiguous(),
            spatial_shape, batch_size,
        )
        x = self.stem(x)
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            if i < len(self.down):
                skips.append(x)
                x = self.down[i](x)
        for i in reversed(range(len(self.up))):
            x = self.up[i](x)
            # Inverse convolutions restore their saved input indices/order;
            # submanifold convolutions between stages preserve those indices.
            x = x.replace_feature(torch.cat((x.features, skips[i].features), dim=1))
            x = self.decoders[i](x)
        return self.classifier(x.features)
