# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import logging
from argparse import Namespace

import torch
from dlnr_lite import DLNR, InputPadder

from .config import get_weights_path_for_model


class DLNRModel:
    """
    A wrapper for the DLNR (Decouple LSTM and Normalization Refinement) model for optical flow and
    disparity estimation. Used for evaluation only.

    DLNR is a high-frequency stereo matching network that computes optical flow and disparity maps
    between two images. It is designed to work with stereo pairs and can handle various image sizes
    by padding them to a size divisible by 32.

    The original DLNR paper can be found here:
        https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_High-Frequency_Stereo_Matching_Network_CVPR_2023_paper.pdf
    """

    def __init__(self, backbone="middleburry", device: torch.device | str = "cuda"):
        """
        Initialize a DLNR model for evaluation.

        Args:
            backbone (str): Backbone to use for the DLNR model. Options are "middleburry" or "sceneflow".
            device (torch.device | str): Device to load the model on (default is "cuda").
        """
        middleburry_weights_url = (
            "https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/DLNR_Middlebury.pth"
        )
        sceneflow_weights_url = "https://fvdb-data.s3.us-east-2.amazonaws.com/fvdb-reality-capture/DLNR_SceneFlow.pth"

        self._logger = logging.getLogger(f"{self.__class__.__module__}.{self.__class__.__name__}")
        if backbone == "middleburry":
            path_to_weights = get_weights_path_for_model(
                "middlebury_dlnr.pth", middleburry_weights_url, model_name="Middleburry"
            )
        elif backbone == "things":
            path_to_weights = get_weights_path_for_model(
                "sceneflow_dlnr.pth", sceneflow_weights_url, model_name="SceneFlow"
            )
        else:
            raise ValueError(f"Unknown backbone: {backbone}")

        self._DLNR_args = Namespace(
            corr_implementation="reg",
            corr_levels=4,
            corr_radius=4,
            dataset="things",
            hidden_dims=[128, 128, 128],
            mixed_precision=True,
            n_downsample=2,
            n_gru_layers=3,
            restore_ckpt=path_to_weights,
            shared_backbone=False,
            slow_fast_gru=False,
        )

        self._logger.debug(f"Loading DLNR model with args: {self._DLNR_args}")
        self._logger.info(f"Loading DLNR pretrained weights from {path_to_weights}")
        dlnr_load = torch.nn.DataParallel(DLNR(self._DLNR_args), device_ids=[0])
        dlnr_load.load_state_dict(torch.load(path_to_weights, weights_only=False))

        self._DLNR_model = dlnr_load.module.to(device)
        self._DLNR_model.eval()
        # self._DLNR_model.freeze_bn()
        self._logger.info("DLNR model loaded successfully.")

    def predict_flow(
        self,
        images1: torch.Tensor,
        images2: torch.Tensor,
        iters=10,
        flow_init=None,
        return_unpadded=False,
        disparity_init: torch.Tensor | None = None,
    ):
        """
        Compute optical flow and disparity between two batches of images using DLNR.

        Args:
            images1 (torch.Tensor): First batch of images, shape [B, H, W, C] (channels last) normalized in the range [0, 1].
            images2 (torch.Tensor): Second batch of images, shape [B, H, W, C] (channels last) normalized in the range [0, 1].
            iters (int): Number of iterations for the DLNR model to run. Defaults to 10.
            flow_init (torch.Tensor, optional): Initial flow estimate, at the model's internal
                resolution (1/2**n_downsample) and in that scale's units. Defaults to None.
            disparity_init (torch.Tensor, optional): Full-resolution positive disparity, shape
                [B, H, W], to warm-start the refinement from. Converted internally to ``flow_init``
                (padded, downsampled and rescaled). Takes precedence over ``flow_init``. Use this to
                seed the matcher where the images carry no texture to match on.
            return_unpadded (bool):
                If True, returns the unpadded flow and disparity. Defaults to False. _i.e._ When False, the input
                is padded to a shape compatible with the DLNR model, and the output is unpadded to match the original input shape.

        Returns:
            flow (torch.Tensor): Optical flow, shape [B, 2, H, W].
            disparity (torch.Tensor): Disparity map, shape [B, H, W].
        """
        if not isinstance(images1, torch.Tensor) or not isinstance(images2, torch.Tensor):
            raise TypeError("images1 and images2 must be torch.Tensor objects.")
        if images1.shape != images2.shape:
            raise ValueError("images1 and images2 must have the same shape.")

        if images1.shape[-1] != 3 or images2.shape[-1] != 3:
            raise ValueError("images1 and images2 must have 3 channels (RGB).")

        if images1.dim() != 4 or images2.dim() != 4:
            raise ValueError("images1 and images2 must be 4D tensors with shape [B, H, W, C].")

        images1 = images1.permute(0, 3, 1, 2) * 255.0  # [B, C, H, W]
        images2 = images2.permute(0, 3, 1, 2) * 255.0  # [B, C, H, W]
        images1 = images1.contiguous()
        images2 = images2.contiguous()
        padder = InputPadder(images1.shape, divis_by=32)
        image1_padded, image2_padded = padder.pad(images1, images2)

        # Warm-start the iterative refinement from a known disparity.
        #
        # DLNR refines from an initial guess, and with no guess it starts at zero. Where the image
        # has texture the correlation volume pulls it to the right match within a few iterations;
        # where it does not -- a blank wall -- there is nothing to pull it, so it stays near zero
        # and the depth derived from it (fx*baseline/disparity) comes out far too large. Measured on
        # pilastro, that biases flat regions +18.7% in depth against +8.3% on textured edges, which
        # is what makes walls look sunken relative to the object's edges.
        #
        # Seeding the iteration with the disparity implied by the splat's own rendered depth fixes
        # the failure at its source: textured regions still refine away from the seed as before,
        # while untextured ones keep the splat's geometry instead of collapsing toward zero.
        #
        # The model adds flow_init to `coords1`, which lives at 1/2**n_downsample resolution (4 here)
        # and in that scale's pixel units, and it is applied after padding -- so the seed is padded
        # with the same padder, downsampled, and its magnitude divided by the same factor. The sign
        # is negative because callers negate the returned disparity.
        if disparity_init is not None:
            if disparity_init.dim() != 3:
                raise ValueError("disparity_init must have shape [B, H, W].")
            factor = 2**self._DLNR_model.args.n_downsample
            seed = -disparity_init.unsqueeze(1).to(images1.dtype)  # [B, 1, H, W]
            seed = padder.pad(seed, seed)[0]
            seed = (
                torch.nn.functional.interpolate(
                    seed, scale_factor=1.0 / factor, mode="bilinear", align_corners=False
                )
                / factor
            )
            flow_init = torch.cat([seed, torch.zeros_like(seed)], dim=1)  # [B, 2, h/4, w/4]

        flow, disparity = self._DLNR_model(
            image1_padded, image2_padded, iters=iters, flow_init=flow_init, test_mode=True
        )

        if return_unpadded:
            return flow, disparity.squeeze(1)  # [B, 2, H, W], [B, H, W]
        else:
            return padder.unpad(flow), padder.unpad(disparity).squeeze(1)  # [B, 2, H, W[, [B, H, W]
