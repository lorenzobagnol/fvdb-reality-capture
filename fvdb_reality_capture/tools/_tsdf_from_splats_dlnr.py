# Copyright Contributors to the OpenVDB Project
# SPDX-License-Identifier: Apache-2.0
#
import pathlib
import tempfile

import numpy as np
import os
import torch
import tqdm
from PIL import Image
from fvdb import Grid
from fvdb.types import NumericMaxRank2, NumericMaxRank3

from fvdb_reality_capture.radiance_fields.gaussian_splatting import GaussianSplat3d

from fvdb_reality_capture.foundation_models.dlnr import DLNRModel
from fvdb_reality_capture.sfm_scene import SfmCache

from ..enums import CameraModel
from ._common import validate_camera_matrices_and_image_sizes, validate_pinhole_camera_models


def debug_plot(
    disparity_l2r: torch.Tensor,
    disparity_r2l: torch.Tensor,
    depth: torch.Tensor,
    image_l: torch.Tensor,
    image_r: torch.Tensor,
    occlusion_mask: torch.Tensor,
    out_filename: str,
) -> None:
    """
    Debug plotting. Plots the disparity maps, depth map, left and right images,
    and the occlusion mask.

    Args:
        disparity_l2r (torch.Tensor): Left-to-right disparity map.
        disparity_r2l (torch.Tensor): Right-to-left disparity map.
        depth (torch.Tensor): Depth map.
        image_l (torch.Tensor): Left image.
        image_r (torch.Tensor): Right image.
        occlusion_mask (torch.Tensor): Occlusion mask.
        out_filename (str): Output filename for the plot.
    """
    import cv2
    import matplotlib.pyplot as plt

    depth_np = depth.squeeze().cpu().numpy()
    occlusion_mask_np = occlusion_mask.cpu().numpy()

    # Shade the depth map by 1 / norm(gradient(depth_np) + shading_eps)
    shading_eps = 1e-6
    g_x = cv2.Sobel(depth_np, cv2.CV_64F, 1, 0)
    g_y = cv2.Sobel(depth_np, cv2.CV_64F, 0, 1)
    shading = 1 / (np.sqrt((g_x**2) + (g_y**2) + shading_eps))
    shading[~occlusion_mask_np] = shading.max()  # Highlight occluded areas

    depth_np[~occlusion_mask_np] = depth_np.max()  # Highlight occluded areas in depth map

    plt.figure(figsize=(10, 20))
    plt.subplot(4, 2, 1)
    plt.title("Depth Map")
    plt.imshow(depth_np, cmap="turbo")
    plt.colorbar()

    plt.subplot(4, 2, 2)
    plt.title("Shaded Depth Map")
    plt.imshow(shading, cmap="turbo")

    plt.subplot(4, 2, 3)
    plt.title("Disparity L2R")
    plt.imshow(disparity_l2r.squeeze().cpu().numpy(), cmap="jet")
    plt.colorbar()

    plt.subplot(4, 2, 4)
    plt.title("Disparity R2L")
    plt.imshow(disparity_r2l.squeeze().cpu().numpy(), cmap="jet")
    plt.colorbar()

    plt.subplot(4, 2, 5)
    plt.title("Left Image")
    plt.imshow(image_l.squeeze().cpu().numpy())

    plt.subplot(4, 2, 6)
    plt.title("Right Image")
    plt.imshow(image_r.squeeze().cpu().numpy())

    plt.savefig(out_filename, bbox_inches="tight")
    plt.close()


class TSDFInputDataset(torch.utils.data.Dataset):
    """
    A torch Dataset that computes RGB images, depths, and weights for TSDF fusion using a Gaussian splat model for images
    and DLNR for depth, and occlusion masking. The dataset caches the results to disk to avoid recomputing them
    when running TSDF fusion.
    """

    def __init__(
        self,
        cache_path: pathlib.Path,
        model: GaussianSplat3d,
        camera_to_world_matrices: torch.Tensor,
        projection_matrices: torch.Tensor,
        image_sizes: torch.Tensor,
        camera_models: torch.Tensor | None,
        distortion_coeffs: torch.Tensor | None,
        baseline: float,
        near: float,
        far: float,
        reprojection_threshold: float,
        alpha_threshold: float,
        dlnr_model: DLNRModel | None,
        use_absolute_baseline: bool,
        show_progress: bool,
        fusion_foreground_mask_paths: list[str | None] | None = None,
        fusion_reference_depths=None,
        rendered_depth_fallback: float = 0.0,
        warm_start_from_render: bool = False,
        dlnr_iters: int = 10,
        vpp_density: float = 0.0,
        vpp_alpha: float = 0.4,
        vpp_texture_quantile: float = 0.5,
    ):
        """
        Create a TSDFInputDataset by precomputing and caching the RGB images, depths, and weights for TSDF fusion.

        Args:
            cache_path (pathlib.Path): Path to the directory to use for caching the results.
            model (GaussianSplat3d): The Gaussian splat model to render from.
            camera_to_world_matrices (torch.Tensor): A (C, 4, 4)-shaped Tensor containing the camera to world
                matrices to render depth images from for mesh extraction where C is the number of camera views.
            projection_matrices (torch.Tensor): A (C, 3, 3)-shaped Tensor containing the perspective projection matrices
                used to render images for mesh extraction where C is the number of camera views.
            image_sizes (torch.Tensor): A (C, 2)-shaped Tensor containing the width and height of each image to extract
                from the Gaussian splat where C is the number of camera views.
            baseline (float): The distance between the two camera positions along the camera -x axis.
                If use_absolute_baseline is False, this is interpreted as a fraction of the mean depth of each image.
            near (float): Near plane distance below which to ignore depth samples, as a multiple of the baseline.
            far (float): Far plane distance above which to ignore depth samples, as a multiple of the baseline.
            reprojection_threshold (float): Reprojection error threshold for occlusion masking in pixels.
            alpha_threshold (float): Alpha threshold to mask pixels where the Gaussian splat model is transparent
                (usually indicating the background).
            dlnr_model (DLNRModel | None): The DLNR model to compute optical flow and disparity. May be
                None only when the cache already holds rgb/depth/weight for every view, in which case
                no inference is needed.
            use_absolute_baseline (bool): If True, use the provided baseline as an absolute distance in world units.
            show_progress (bool): Whether to show a progress bar (default is True).
            fusion_foreground_mask_paths (list[str | None] | None): Optional per-view mask paths used for
                streaming foreground gating. Each entry is either a mask image path or None.
        """
        if not cache_path.exists():
            cache_path.mkdir(parents=True, exist_ok=True)

        self.cache = SfmCache.get_cache(cache_path, "TSDFInputs", "Cache for TSDF inputs")
        self.num_images = camera_to_world_matrices.shape[0]
        self.model = model
        self.baseline_fraction_of_depth_or_absolute = baseline
        self.use_absolute_baseline = use_absolute_baseline
        self.near = near
        self.far = far
        self.reprojection_threshold = reprojection_threshold
        self.alpha_threshold = alpha_threshold
        self.dlnr_model = dlnr_model
        self.rendered_depth_fallback = rendered_depth_fallback
        self.warm_start_from_render = warm_start_from_render
        self.dlnr_iters = dlnr_iters
        self.vpp_density = vpp_density
        self.vpp_alpha = vpp_alpha
        self.vpp_texture_quantile = vpp_texture_quantile
        self.fusion_foreground_mask_paths = fusion_foreground_mask_paths
        # Per-view reference depth for the baseline (None -> mean rendered depth, the upstream rule)
        self.fusion_reference_depths = None if fusion_reference_depths is None else [float(v) for v in fusion_reference_depths]
        self._current_ref_depth = None
        self.camera_models = (
            camera_models
            if camera_models is not None
            else torch.full((self.num_images,), int(CameraModel.PINHOLE), dtype=torch.int32)
        )
        self.distortion_coeffs = (
            distortion_coeffs
            if distortion_coeffs is not None
            else torch.zeros((self.num_images, 12), dtype=torch.float32)
        )

        if self.fusion_foreground_mask_paths is not None and len(self.fusion_foreground_mask_paths) != self.num_images:
            raise ValueError(
                "fusion_foreground_mask_paths length mismatch: "
                f"expected {self.num_images}, got {len(self.fusion_foreground_mask_paths)}"
            )

        device = model.device

        enumerator = (
            tqdm.tqdm(range(self.num_images), unit="imgs", desc="Generating DLNR Depths")
            if show_progress
            else range(self.num_images)
        )

        def _cache_has_all_required_for_view(view_idx: int) -> bool:
            required = [f"rgb_{view_idx}", f"depth_{view_idx}", f"weight_{view_idx}"]
            for key in required:
                try:
                    self.cache.read_file(key)
                except (FileNotFoundError, ValueError):
                    return False
            return True

        reused_count = 0
        generated_count = 0

        for i in enumerator:
            # Skip views already present in the cache so a re-run (e.g. re-meshing at different TSDF
            # settings) does not redo DLNR inference, which dominates wall-clock time.
            if _cache_has_all_required_for_view(i):
                reused_count += 1
                continue

            if self.dlnr_model is None:
                raise RuntimeError(
                    "Missing cached TSDF inputs and DLNR model is not available. "
                    "Provide dlnr_model or ensure cache contains rgb/depth/weight for all views."
                )

            cam_to_world_matrix = camera_to_world_matrices[i].to(dtype=torch.float32, device=device)
            world_to_cam_matrix = (
                torch.linalg.inv(cam_to_world_matrix).contiguous().to(dtype=torch.float32, device=device)
            )
            projection_matrix = projection_matrices[i].to(dtype=torch.float32, device=device)
            camera_model = CameraModel(int(self.camera_models[i].item()))
            distortion_coeffs_i = self.distortion_coeffs[i].to(dtype=torch.float32, device=device)
            image_height, image_width = int(image_sizes[i][0].item()), int(image_sizes[i][1].item())
            self._current_ref_depth = self.fusion_reference_depths[i] if self.fusion_reference_depths is not None else None

            rgb_image, depth_image, weight_image = self.extract_single_tsdf_input(
                world_to_cam_matrix=world_to_cam_matrix,
                projection_matrix=projection_matrix,
                camera_model=camera_model,
                distortion_coeffs=distortion_coeffs_i,
                image_width=image_width,
                image_height=image_height,
                fusion_foreground_mask=None,
                save_debug_images_to=None,  # Set to a path if you want to save debug images
            )

            self.cache.write_file(f"rgb_{i}", rgb_image.cpu().numpy(), data_type="npy")
            self.cache.write_file(f"depth_{i}", depth_image.cpu().numpy(), data_type="npy")
            self.cache.write_file(f"weight_{i}", weight_image.cpu().numpy(), data_type="npy")

            generated_count += 1

        if reused_count > 0:
            print(
                "Reused cached TSDF inputs: "
                f"{reused_count}/{self.num_images} views "
                f"(generated {generated_count} missing views)."
            )

    def extract_single_tsdf_input(
        self,
        world_to_cam_matrix: torch.Tensor,
        projection_matrix: torch.Tensor,
        camera_model: CameraModel,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
        fusion_foreground_mask: torch.Tensor | None = None,
        save_debug_images_to: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute an RGB image, depth map, and weight image to be used as input to TSDF fusion for a given camera pose.
        This function uses the Gaussian splat model for images and the DLNR model for depth, and occlusion masking.
        This algorithm is roughly based on the GS2Mesh algorithm described in https://arxiv.org/abs/2404.01810.

        The algorithm renders a stereo pair of images from the Gaussian splat model for each camera position, computes disparities
        using DLNR, computes an occlusion mask based on the disparities, and then computes a
        near/far mask based on the depth. The final weights are a combination of the near/far mask and the occlusion mask.

        Args:
            world_to_cam_matrix (torch.Tensor): The camera_to_world transformation matrix for the first
                image in the stereo pair. The second image has the same transformation but with the x position
                shifted by the negative baseline.
            projection_matrix (torch.Tensor): The projection matrix for the camera.
            image_width (int): The width of the rendered images in pixels.
            image_height (int): The height of the rendered images in pixels.
            alpha_threshold (float): Alpha threshold to mask pixels where the Gaussian splat model is not confident.
            save_debug_images_to (str | None): If provided, saves debug images to this path.

        Returns:
            image_l (torch.Tensor): The first rendered image whose camera to world matrix is the same as the input.
            depth (torch.Tensor): The rendered depth image for the first camera.
            weights (torch.Tensor): The computed weight image for the first camera.
        """

        # The splat's own depth is needed to size the baseline, and again below if the fallback is
        # on, so render it once here rather than twice.
        need_rendered = (
            (not self.use_absolute_baseline)
            or self.rendered_depth_fallback > 0.0
            or self.warm_start_from_render
            or self.vpp_density > 0.0
            or float(os.environ.get("TEST_TSDF_AGREE_RENDER", "0") or 0) > 0
        )
        rendered_depth, rendered_alpha = (
            self.render_reference_depth(
                world_to_camera_matrix=world_to_cam_matrix,
                projection_matrix=projection_matrix,
                camera_model=camera_model,
                distortion_coeffs=distortion_coeffs,
                image_width=image_width,
                image_height=image_height,
            )
            if need_rendered
            else (None, None)
        )
        self._vpp_alpha_map = rendered_alpha

        if not self.use_absolute_baseline and self._current_ref_depth is not None and self._current_ref_depth > 0:
            # reference depth from the SfM points this view sees: independent of how the splat fills
            # the sky (transparent -> tiny mean -> far band cuts the object's far parts; sphere -> huge)
            baseline = self.baseline_fraction_of_depth_or_absolute * float(self._current_ref_depth)
        elif not self.use_absolute_baseline:
            baseline = self.baseline_fraction_of_depth_or_absolute * rendered_depth.mean().item()
        else:
            baseline = self.baseline_fraction_of_depth_or_absolute

        near = self.near * baseline
        far = self.far * baseline

        # Render the stereo pair of images and clip to [0, 1]
        image_l, image_r, alpha_mask = self.render_stereo_pair(
            baseline,
            world_to_cam_matrix,
            projection_matrix,
            camera_model,
            distortion_coeffs,
            image_width,
            image_height,
        )
        image_l.clip_(min=0.0, max=1.0)
        image_r.clip_(min=0.0, max=1.0)

        # Compute left-to-right and right-to-left disparities and depth using DLNR
        disparity_l2r, disparity_r2l, depth = self.compute_disparities_and_depth(
            image_l=image_l,
            image_r=image_r,
            projection_matrix=projection_matrix,
            baseline=baseline,
            init_depth=(
                rendered_depth.reshape(image_l.shape[:2])
                if (self.warm_start_from_render or self.vpp_density > 0.0)
                else None
            ),
        )

        # Compute an occlusion mask based on the reprojection error of the disparities
        occlusion_mask = self.compute_occlusion_mask(
            disparity_l2r,
            disparity_r2l,
        )

        # Create masks using the near and far values
        near_far_mask = (depth > near) & (depth < far)
        # Grazing-angle gate (TEST_TSDF_GRAZING = minimum |cos| between view ray and surface normal).
        _graz = float(os.environ.get("TEST_TSDF_GRAZING", "0") or 0)
        if _graz > 0:
            _d = depth.reshape(image_height, image_width).float()
            _fx = float(projection_matrix[0, 0]); _fy = float(projection_matrix[1, 1]); _cx = float(projection_matrix[0, 2]); _cy = float(projection_matrix[1, 2])
            _v, _u = torch.meshgrid(torch.arange(image_height, device=_d.device, dtype=torch.float32), torch.arange(image_width, device=_d.device, dtype=torch.float32), indexing="ij")
            _X = torch.stack([(_u - _cx) / _fx * _d, (_v - _cy) / _fy * _d, _d], dim=-1)   # camera-space points
            _dx = torch.zeros_like(_X); _dy = torch.zeros_like(_X)
            _dx[:, 1:-1] = (_X[:, 2:] - _X[:, :-2]) / 2; _dy[1:-1, :] = (_X[2:, :] - _X[:-2, :]) / 2
            _n = torch.cross(_dx, _dy, dim=-1); _n = _n / _n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            _ray = _X / _X.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            _cos = (_n * _ray).sum(-1).abs()
            _ok = (_cos >= _graz) & torch.isfinite(_cos)
            near_far_mask = near_far_mask & _ok.reshape(near_far_mask.shape)

        # The final weights are a combination of the near/far mask and the occlusion mask
        if alpha_mask is not None:
            weights = near_far_mask & occlusion_mask & alpha_mask
        else:
            weights = near_far_mask & occlusion_mask

        # Stereo matching needs texture. On a blank wall DLNR has nothing to lock onto, returns a
        # near-zero disparity, and the depth it derives from it explodes -- measured on one scene,
        # a median relative error of 5.7 (570%) on the flattest texture decile against 0.004 on the
        # most textured. Those depths land outside the near/far band, the pixel is weighted out, no
        # voxel is ever deposited, and the mesh ends up with a hole exactly where the wall is.
        #
        # The splat itself does not have that problem: its depth on the same flat regions agrees to
        # within 0.6% when rendered from one view and reprojected into another, at alpha 0.9999. It
        # reconstructs the wall as a surface whether or not the wall has texture. So where the two
        # disagree, prefer the splat and keep the pixel instead of discarding it.
        #
        # Relaxing `disparity_reprojection_threshold` is the wrong lever here and was measured to
        # make things worse: it admits more pixels of lower quality (3px accepts 0.304 of which 8.2%
        # are within 10% of correct; 20px accepts 0.764 of which 4.3% are).
        if self.rendered_depth_fallback > 0.0:
            rendered = rendered_depth.reshape(depth.shape)
            rendered_a = rendered_alpha.reshape(depth.shape)

            agrees = (depth - rendered).abs() / rendered.clamp(min=1e-6) <= self.rendered_depth_fallback
            trust_dlnr = weights & agrees
            # The splat's depth is only usable where the splat actually has a surface and that
            # surface is inside the band we are fusing.
            _fb_alpha = float(os.environ.get("TEST_TSDF_FALLBACK_ALPHA", "0") or 0) or self.alpha_threshold
            splat_has_surface = (rendered_a > _fb_alpha) & (rendered > near) & (rendered < far)

            depth = torch.where(trust_dlnr, depth, rendered)
            weights = trust_dlnr | splat_has_surface

        if fusion_foreground_mask is not None:
            fg_mask = fusion_foreground_mask
            if fg_mask.dim() == 3 and fg_mask.shape[0] == 1:
                fg_mask = fg_mask[0]
            if fg_mask.dim() != 2:
                raise ValueError(
                    f"Expected fusion_foreground_mask to have shape (H, W) or (1, H, W), got {tuple(fg_mask.shape)}"
                )
            fg_mask = fg_mask.to(device=weights.device)
            if fg_mask.dtype != torch.bool:
                fg_mask = fg_mask > 0.5
            if fg_mask.shape != weights.shape:
                raise ValueError(
                    f"fusion_foreground_mask shape {tuple(fg_mask.shape)} does not match weights shape {tuple(weights.shape)}"
                )
            weights = weights & fg_mask

        if save_debug_images_to is not None:
            debug_plot(
                disparity_l2r=disparity_l2r,
                disparity_r2l=disparity_r2l,
                depth=depth,
                image_l=image_l,
                image_r=image_r,
                occlusion_mask=occlusion_mask,
                out_filename=save_debug_images_to,
            )

        # Agreement substitution (TEST_TSDF_AGREE_RENDER = k voxels): DLNR validates, the render supplies the value.
        _agree_k = float(os.environ.get("TEST_TSDF_AGREE_RENDER", "0") or 0)
        if _agree_k > 0 and rendered_depth is not None:
            _vox = float(os.environ.get("TEST_MESH_VOXEL", "0.0125"))
            _r = rendered_depth.reshape(depth.shape); _ra = rendered_alpha.reshape(depth.shape)
            _ok = (weights.to(torch.bool)) & (_ra > 0.5) & ((depth - _r).abs() <= _agree_k * _vox)
            if os.environ.get("TEST_TSDF_AGREE_MODE", "sub") == "avg":
                depth = torch.where(_ok, 0.5 * (_r + depth), depth)
            else:
                depth = torch.where(_ok, _r, depth)
            self._agree_stats = (int(_ok.sum()), int(weights.to(torch.bool).sum()))
        # Precision weighting (TEST_TSDF_PRECISION_WEIGHT = power): w = (Zref/Z)^power in [0,1] with Zref the
        # view's reference depth (SfM median when TEST_TSDF_DEPTH_REF=sfm, else the rendered mean).
        _pw = float(os.environ.get("TEST_TSDF_PRECISION_WEIGHT", "0") or 0)
        if _pw > 0:
            _zref = float(self._current_ref_depth) if getattr(self, "_current_ref_depth", None) else float(baseline / self.baseline_fraction_of_depth_or_absolute)
            _w = (_zref / depth.clamp(min=1e-6)).clamp(max=1.0) ** _pw
            weights = weights.to(torch.float32) * _w.reshape(weights.shape).to(torch.float32)
        return image_l, depth, weights

    def estimate_baseline_from_depth(
        self,
        world_to_camera_matrix: torch.Tensor,
        projection_matrix: torch.Tensor,
        camera_model: CameraModel,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> float:
        """
        Estimate a baseline distance as a percentage of the mean depth of an image rendered from a Gaussian Splat model.

        We want to choose a baseline that is wide enough to get good depth estimates from stereo matching,
        but not so wide that the two images have little overlap (and thus have high error in the disparity estimate).
        A common heuristic is to set the baseline to be a small percentage of the mean depth of the scene.

        Args:
            world_to_camera_matrix (torch.Tensor): The camera_to_world transformation matrix for the image.
            projection_matrix (torch.Tensor): The projection matrix for the camera.
            image_width (int): The width of the rendered image.
            image_height (int): The height of the rendered image.

        Returns:
            float: The estimated baseline distance in world units.
        """

        depth_0, _ = self.render_reference_depth(
            world_to_camera_matrix=world_to_camera_matrix,
            projection_matrix=projection_matrix,
            camera_model=camera_model,
            distortion_coeffs=distortion_coeffs,
            image_width=image_width,
            image_height=image_height,
        )
        baseline = self.baseline_fraction_of_depth_or_absolute * depth_0.mean().item()
        return baseline

    def render_reference_depth(
        self,
        world_to_camera_matrix: torch.Tensor,
        projection_matrix: torch.Tensor,
        camera_model: CameraModel,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Render the splat's own depth and alpha for a camera, with the depth divided through by
        alpha so it is a depth rather than an alpha-weighted accumulation.

        This is the depth the splat itself reconstructs, as opposed to the depth DLNR estimates by
        matching a stereo pair. It is used both to size the stereo baseline and, when
        ``rendered_depth_fallback`` is enabled, to stand in for DLNR wherever the two disagree.

        Returns:
            depth (torch.Tensor): The alpha-normalized rendered depth.
            alpha (torch.Tensor): The accumulated alpha, for deciding where the splat has a surface.
        """
        depth, alpha = self.model.render_depths(
            world_to_camera_matrices=world_to_camera_matrix.unsqueeze(0),
            projection_matrices=projection_matrix.unsqueeze(0),
            image_width=image_width,
            image_height=image_height,
            near=0.0,
            far=1e10,
            camera_model=camera_model,
            distortion_coeffs=distortion_coeffs.unsqueeze(0) if camera_model != CameraModel.PINHOLE else None,
        )
        return depth / alpha.clamp(min=1e-10), alpha

    def render_stereo_pair(
        self,
        baseline: float,
        world_to_camera_matrix: torch.Tensor,
        projection_matrix: torch.Tensor,
        camera_model: CameraModel,
        distortion_coeffs: torch.Tensor,
        image_width: int,
        image_height: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """
        Render a pair of stereo images from a Gaussian Splat model.

        The pair of images is rendered by shifting the camera position by a baseline distance along
        the camera's -x axis to simulate stereo vision.

        Args:
            baseline (float): The distance between the two camera positions along the camera -x axis (in world units).
            world_to_camera_matrix (torch.Tensor): The camera_to_world transformation matrix for the first
                image in the stereo pair. The second image has the same transformation but with the x position
                shifted by the negative baseline.
            projection_matrix (torch.Tensor): The projection matrix for the camera.
            image_width (int): The width of the rendered images in pixels.
            image_height (int): The height of the rendered images in pixels.

        Returns:
            image_1 (torch.Tensor): The first rendered image whose camera to world matrix is the same as the input.
            image_2 (torch.Tensor): The second rendered image whose camera to world matrix is the same as the input but
                with the x position shifted by the negative baseline.
            alpha_mask (torch.Tensor): A binary mask indicating pixels where the alpha value exceeds the alpha
                threshold if self._alpha_threshold > 0.0, else None.
        """
        # Compute the left and right camera poses
        world_to_camera_matrix_left = world_to_camera_matrix.clone()
        world_to_camera_matrix_right = world_to_camera_matrix.clone()
        world_to_camera_matrix_right[0, 3] -= baseline

        world_to_camera_matrix = torch.stack([world_to_camera_matrix_left, world_to_camera_matrix_right], dim=0)
        projection_matrix = torch.stack([projection_matrix, projection_matrix], dim=0)

        images, alphas = self.model.render_images(
            world_to_camera_matrices=world_to_camera_matrix,
            projection_matrices=projection_matrix,
            image_width=image_width,
            image_height=image_height,
            near=0.0,
            far=1e10,
            camera_model=camera_model,
            distortion_coeffs=(
                torch.stack([distortion_coeffs, distortion_coeffs], dim=0)
                if camera_model != CameraModel.PINHOLE
                else None
            ),
        )

        alpha_mask = alphas[0].squeeze(-1) > self.alpha_threshold if self.alpha_threshold > 0.0 else None

        return images[0], images[1], alpha_mask

    def compute_occlusion_mask(self, l2r_disparity: torch.Tensor, r2l_disparity: torch.Tensor) -> torch.Tensor:
        """
        Compute an occlusion mask using the disparity maps by filtering pixels where the
        reprojection error exceeds the reprojection threshold.

        Given a point in space, and a stereo pair of images, disparity maps are computed as the
        difference in pixel coordinates between the projection of that point in the left and right images.

        The occlusion mask is computed by using the left-to-right disparity map to project pixels from the left image
        to the right image, and then using the right-to-left disparity map to reproject those pixels back to the left image.
        If the reprojection error exceeds the reprojection threshold, the pixel is considered occluded.

        Args:
            l2r_disparity (torch.Tensor): Left-to-right disparity map.
            r2l_disparity (torch.Tensor): Right-to-left disparity map.

        Returns:
            torch.Tensor: Binary occlusion mask where 0 indicates occluded pixels and 1 indicates visible pixels.
        """

        height, width = l2r_disparity.shape

        x_values = torch.arange(width, device=l2r_disparity.device)
        y_values = torch.arange(height, device=l2r_disparity.device)
        x_grid, y_grid = torch.meshgrid(x_values, y_values, indexing="xy")

        x_projected = (x_grid - l2r_disparity).to(torch.int32)
        x_projected_clipped = torch.clamp(x_projected, 0, width - 1)

        x_reprojected = x_projected_clipped + r2l_disparity[y_grid, x_projected_clipped]
        x_reprojected_clipped = torch.clamp(x_reprojected, 0, width - 1)

        disparity_difference = torch.abs(x_grid - x_reprojected_clipped)

        occlusion_mask = disparity_difference > self.reprojection_threshold

        occlusion_mask[(x_projected < 0) | (x_projected >= width)] = True

        return ~occlusion_mask

    def apply_virtual_pattern(
        self,
        image_l: torch.Tensor,  # [H, W, C]
        image_r: torch.Tensor,  # [H, W, C]
        init_depth: torch.Tensor,  # [H, W]
        fx: float,
        baseline: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Blend a sparse pattern into both rendered images at corresponding positions, so that
        stereo matching has something to lock onto where the surface itself offers nothing.

        For a pixel ``(x, y)`` in the left image at depth ``Z``, the same 3D point lands at
        ``x - fx*baseline/Z`` in the right image. Writing the same pattern value at both places
        gives the correlation volume a peak at the correct disparity rather than a plateau.

        Pixels are chosen from the least textured part of the image -- where matching actually
        fails -- and only a small fraction of them, so the matcher is helped rather than simply
        told the answer.

        Returns copies: the originals are the fused mesh colours and must not carry the pattern.
        """
        h, w, _ = image_l.shape
        device = image_l.device

        disparity = (fx * baseline) / init_depth.clamp(min=1e-6)
        y_grid, x_grid = torch.meshgrid(
            torch.arange(h, device=device), torch.arange(w, device=device), indexing="ij"
        )
        x_right = torch.round(x_grid.float() - disparity).long()

        usable = (
            torch.isfinite(disparity) & (disparity > 0) & (x_right >= 0) & (x_right < w)
        )
        _vpp_alpha = float(os.environ.get("TEST_TSDF_VPP_ALPHA", "0") or 0)
        if _vpp_alpha > 0 and getattr(self, "_vpp_alpha_map", None) is not None:
            usable = usable & (self._vpp_alpha_map.reshape(h, w) > _vpp_alpha)
        _vpp_dg = float(os.environ.get("TEST_TSDF_VPP_DEPTHGRAD", "0") or 0)
        if _vpp_dg > 0:
            d = init_depth
            gx = torch.zeros_like(d); gy = torch.zeros_like(d)
            gx[:, 1:-1] = (d[:, 2:] - d[:, :-2]).abs() / 2; gy[1:-1, :] = (d[2:, :] - d[:-2, :]).abs() / 2
            rel = torch.maximum(gx, gy) / d.clamp(min=1e-6)
            # dilate the edge mask by 3 px so the pattern keeps clear of silhouettes
            edge = (rel > _vpp_dg).float()[None, None]
            edge = torch.nn.functional.max_pool2d(edge, kernel_size=7, stride=1, padding=3)[0, 0] > 0
            usable = usable & ~edge
        if not bool(usable.any()):
            return image_l, image_r

        gray = image_l.mean(dim=-1)
        texture = torch.zeros_like(gray)
        texture[:-1, :-1] = torch.abs(torch.diff(gray, dim=1))[:-1, :] + torch.abs(
            torch.diff(gray, dim=0)
        )[:, :-1]

        # Restrict to the flattest part of the image, then keep `vpp_density` of those. Sampling
        # uniformly instead would spend most of the pattern where matching already works.
        flat_cut = torch.quantile(texture[usable].float(), self.vpp_texture_quantile)
        candidates = usable & (texture <= flat_cut)
        keep = candidates & (torch.rand(h, w, device=device) < self.vpp_density)
        if not bool(keep.any()):
            return image_l, image_r

        ys, xs = torch.nonzero(keep, as_tuple=True)
        xr = x_right[ys, xs]
        # One value per point, written identically to both views: the pattern must be a property of
        # the 3D point, not of the image.
        pattern = torch.rand(ys.shape[0], 1, device=device, dtype=image_l.dtype)

        out_l = image_l.clone()
        out_r = image_r.clone()
        a = self.vpp_alpha
        out_l[ys, xs] = (1.0 - a) * out_l[ys, xs] + a * pattern
        out_r[ys, xr] = (1.0 - a) * out_r[ys, xr] + a * pattern
        return out_l, out_r

    def compute_disparities_and_depth(
        self,
        image_l: torch.Tensor,  # [H, W, C]
        image_r: torch.Tensor,  # [H, W, C]
        projection_matrix: torch.Tensor,  # [3, 3]
        baseline: float,
        init_depth: torch.Tensor | None = None,  # [H, W], the splat's own rendered depth
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute left-to-right and right-to-left disparities and depth using the DLNR model, and use
        the pinhole camera model to convert the left-to-right disparity to depth.

        Args:
            image_l (torch.Tensor): The left image.
            image_r (torch.Tensor): The right image.
            projection_matrix (torch.Tensor): The projection matrix for the camera.
            baseline (float): The distance between the two camera positions along the camera -x axis (in world units).

        Returns:
            disparity_l2r (torch.Tensor): Left-to-right disparity map.
            disparity_r2l (torch.Tensor): Right-to-left disparity map.
            depth (torch.Tensor): Depth map computed from the left-to-right disparity.
        """
        fx_seed = projection_matrix[0, 0].item()

        # Virtual pattern projection.
        #
        # A blank wall gives stereo nothing to match: measured on these renders, the local gradient
        # on flat regions is 0.00013 against 0.00240 on textured ones, eighteen times less. No
        # amount of tuning recovers a signal that is not there, so this puts one in -- blending a
        # pattern into BOTH rendered images at positions that correspond to the same 3D point, using
        # the splat's own depth to place it. The correlation volume then has a sharp peak exactly
        # where it currently has a plateau.
        #
        # Two properties make this legitimate rather than a trick. The pattern is geometrically
        # consistent across the pair, so it does not bias the match -- uncorrelated noise would be
        # anti-correlated between the views and would make matching worse, not better. And it is
        # kept sparse: with a dense pattern the network would simply read the splat's depth back
        # out, inheriting its errors, instead of estimating geometry independently.
        #
        # Only possible because the pair is rendered rather than captured: the published method
        # needs a depth sensor for the hints, while here depth is free and dense.
        img_l_in, img_r_in = image_l, image_r
        if self.vpp_density > 0.0 and init_depth is not None:
            img_l_in, img_r_in = self.apply_virtual_pattern(
                image_l, image_r, init_depth, fx_seed, baseline
            )
        # Seed the matcher with the disparity the splat's own depth implies. Without a seed DLNR
        # starts from zero disparity, and on a surface with no texture nothing in the correlation
        # volume moves it away from there -- so depth = fx*baseline/disparity comes out far too
        # large and the wall is reconstructed behind where it belongs.
        #
        # Both directions are seeded, not just left-to-right. Seeding only one would leave the other
        # collapsed at zero on exactly those untextured pixels, the reprojection check would see the
        # two disagree, and the pixels would be masked out -- trading a sunken wall for a missing
        # one. With both seeded the check still does its job: genuine occlusions make the two
        # directions disagree regardless of where they started.
        seed_l2r = seed_r2l = None
        if init_depth is not None:
            disp_seed = (fx_seed * baseline) / init_depth.clamp(min=1e-6)
            seed_l2r = disp_seed.unsqueeze(0)
            # The right-to-left pass runs on horizontally flipped images, so its seed is flipped too.
            seed_r2l = torch.flip(disp_seed, dims=[1]).unsqueeze(0)

        _, disparity_l2r = self.dlnr_model.predict_flow(
            images1=img_l_in.unsqueeze(0),  # [1, H, W, C]
            images2=img_r_in.unsqueeze(0),  # [1, H, W, C]
            flow_init=None,
            disparity_init=seed_l2r,
            iters=self.dlnr_iters,
        )
        disparity_l2r = -disparity_l2r[0]  # [H, W]

        image_l_flip = torch.flip(img_l_in, dims=[1])
        image_r_flip = torch.flip(img_r_in, dims=[1])
        _, disparity_r2l = self.dlnr_model.predict_flow(
            images1=image_r_flip.unsqueeze(0),  # [1, H, W, C]
            images2=image_l_flip.unsqueeze(0),  # [1, H, W, C]
            flow_init=None,
            disparity_init=seed_r2l,
            iters=self.dlnr_iters,
        )
        disparity_r2l = -torch.flip(disparity_r2l[0], dims=[1])  # [H, W]

        fx = projection_matrix[0, 0].item()
        depth = (fx * baseline) / disparity_l2r

        return disparity_l2r, disparity_r2l, depth

    def __len__(self):
        return self.num_images

    def _load_fusion_foreground_mask_for_idx(self, idx: int, image_height: int, image_width: int) -> torch.Tensor | None:
        """
        Load the foreground mask for a single view, or None when this view has no mask.

        A mask that fails to load is treated as "no gating" rather than an error: losing background
        suppression for one view degrades that view's contribution, whereas aborting would throw away
        an entire (expensive) TSDF run.
        """
        if self.fusion_foreground_mask_paths is None:
            return None

        mask_path = self.fusion_foreground_mask_paths[idx]
        if mask_path is None or len(mask_path) == 0:
            return None

        try:
            with Image.open(mask_path) as mask_img:
                mask_arr = np.asarray(mask_img.convert("L"), dtype=np.uint8)
            if mask_arr.shape != (image_height, image_width):
                raise ValueError(f"expected {(image_height, image_width)} got {tuple(mask_arr.shape)}")
            return torch.from_numpy(mask_arr > 127)
        except Exception as e:
            print(f"WARNING: Failed to load foreground mask {mask_path}: {e}. Using all pixels for this view.")
            return None

    def __getitem__(self, idx):
        _, rgb = self.cache.read_file(f"rgb_{idx}")
        _, depth = self.cache.read_file(f"depth_{idx}")
        _, weight = self.cache.read_file(f"weight_{idx}")

        rgb_t = torch.from_numpy(rgb) if not torch.is_tensor(rgb) else rgb
        depth_t = torch.from_numpy(depth) if not torch.is_tensor(depth) else depth
        weight_t = torch.from_numpy(weight) if not torch.is_tensor(weight) else weight

        # Masks gate the per-pixel fusion weights, so background pixels never allocate voxels.
        # Applied here rather than at generation time so a cached run can be re-fused with
        # different masks without redoing DLNR inference.
        if self.fusion_foreground_mask_paths is not None:
            image_height, image_width = weight_t.shape[-2], weight_t.shape[-1]
            fusion_foreground_mask = self._load_fusion_foreground_mask_for_idx(idx, image_height, image_width)
            if fusion_foreground_mask is not None:
                _m = fusion_foreground_mask.to(device=weight_t.device)
                weight_t = (weight_t & _m) if weight_t.dtype == torch.bool else weight_t * _m.to(weight_t.dtype)

        return rgb_t, depth_t, weight_t


@torch.no_grad()
def tsdf_from_splats_dlnr(
    model: GaussianSplat3d,
    camera_to_world_matrices: NumericMaxRank3,
    projection_matrices: NumericMaxRank3,
    image_sizes: NumericMaxRank2,
    truncation_margin: float,
    camera_models: torch.Tensor | None = None,
    distortion_coeffs: torch.Tensor | None = None,
    grid_shell_thickness: float | int = 3.0,
    baseline: float = 0.07,
    near: float = 4.0,
    far: float = 20.0,
    disparity_reprojection_threshold: float = 3.0,
    alpha_threshold: float = 0.1,
    image_downsample_factor: int = 1,
    dtype: torch.dtype = torch.float16,
    feature_dtype: torch.dtype = torch.uint8,
    dlnr_backbone: str = "middleburry",
    use_absolute_baseline: bool = False,
    show_progress: bool = True,
    num_workers: int = 8,
    dlnr_cache_path: str | pathlib.Path | None = None,
    fusion_foreground_mask_paths: list[str | None] | None = None,
    fusion_reference_depths=None,
    rendered_depth_fallback: float = 0.0,
    warm_start_from_render: bool = False,
    dlnr_iters: int = 10,
    vpp_density: float = 0.0,
    vpp_alpha: float = 0.4,
    vpp_texture_quantile: float = 0.5,
    min_voxel_weight: float = 0.0,
) -> tuple[Grid, torch.Tensor, torch.Tensor]:
    """
    Extract a Truncated Signed Distance Field (TSDF) from a `fvdb_reality_capture.GaussianSplat3d` using TSDF fusion from depth maps
    predicted from the Gaussian splat radiance field and the
    `DLNR foundation model <https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_High-Frequency_Stereo_Matching_Network_CVPR_2023_paper.pdf>`_.
    DLNR is a high-frequency stereo matching network that computes optical flow and disparity maps between two images, which can be used to compute depth.

    This algorithm proceeds in two steps:

    1. First, it renders stereo pairs of images from the Gaussian splat radiance field, and uses
       DLNR to compute depth maps from these stereo pairs in the frame of the first image in the pair.
       The result is a set of depth maps aligned with the rendered images.

    2. Second, it integrates the depths and colors/features into a sparse :class:`fvdb.Grid` in a narrow band
       around the surface using sparse truncated signed distance field (TSDF) fusion.
       The result is a sparse voxel grid representation of the scene where each voxel stores a signed distance
       value and color (or other features).


    .. note::
        You can extract a mesh from the TSDF using the marching cubes algorithm implemented in
        :class:`fvdb.Grid.marching_cubes`. If your goal is to extract a mesh from a Gaussian splat model,
        consider using :func:`fvdb_reality_capture.tools.mesh_from_splats_dlnr` which combines this function
        with marching cubes to directly extract a mesh.

    .. note::

        This algorithm implemented is based on the paper
        `"GS2Mesh: Surface Reconstruction from Gaussian Splatting via Novel Stereo Views" <https://arxiv.org/abs/2404.01810>`_.
        We make key improvements to the method by using a more robust stereo baseline estimation method and by using a much
        more efficient sparse TSDF fusion implementation built on `fVDB <https://fvdb-core.readthedocs.io>`_.

    .. note::

        The TSDF fusion algorithm is a method for integrating multiple depth maps into a single volumetric representation of a scene encoded a
        truncated signed distance field (*i.e.* a signed distance field in a narrow band around the surface).
        TSDF fusion was first described in the paper
        `"KinectFusion: Real-Time Dense Surface Mapping and Tracking" <https://www.microsoft.com/en-us/research/publication/kinectfusion-real-time-3d-reconstruction-and-interaction-using-a-moving-depth-camera/>`_.
        We use a modified version of this algorithm which only allocates voxels in a narrow band around the surface of the model
        to reduce memory usage and speed up computation.

    .. note::

        The DLNR model is a high-frequency stereo matching network that computes optical flow and disparity maps
        between two images. The DLNR model is described in the paper
        `"High-Frequency Stereo Matching Network" <https://openaccess.thecvf.com/content/CVPR2023/papers/Zhao_High-Frequency_Stereo_Matching_Network_CVPR_2023_paper.pdf>`_.

    .. note::

        Meshing currently supports only :class:`fvdb_reality_capture.CameraModel.PINHOLE` cameras. While the
        rendering step can handle additional camera models, the underlying fVDB TSDF integration
        path currently assumes perspective pinhole projection. Passing distorted or orthographic
        cameras will raise :class:`NotImplementedError`.


    Args:
        model (GaussianSplat3d): The Gaussian splat radiance field to extract a mesh from.
        camera_to_world_matrices (NumericMaxRank3): A ``(C, 4, 4)``-shaped Tensor containing the camera to world
            matrices to render depth images from for mesh extraction where ``C`` is the number of camera views.
        projection_matrices (NumericMaxRank3): A ``(C, 3, 3)``-shaped Tensor containing the perspective projection matrices
            used to render images for mesh extraction where ``C`` is the number of camera views.
        image_sizes (NumericMaxRank2): A ``(C, 2)``-shaped Tensor containing the height and width of each image to extract
            from the Gaussian splat where ``C`` is the number of camera views. *i.e.*, ``image_sizes[c] = (height_c, width_c)``.
        truncation_margin (float): Margin for truncating the TSDF, in world units. This defines the half-width of the band around the surface
            where the TSDF is defined in world units.
        grid_shell_thickness (float): The number of voxels along each axis to include in the TSDF volume.
            This defines the resolution of the Grid around narrow band around the surface.
            Default is 3.0.
        baseline (float): Baseline distance for stereo depth estimation.
            If ``use_absolute_baseline`` is ``False``, this is interpreted as a fraction of
            the mean depth of each image. Otherwise, it is interpreted as an absolute distance in world units.
            Default is 0.07.
        near (float): Near plane distance below which to ignore depth samples, as a multiple of the baseline.
        far (float): Far plane distance above which to ignore depth samples, as a multiple of the baseline.
        disparity_reprojection_threshold (float): Reprojection error threshold for occlusion
            masking (in pixels units). Default is 3.0.
        alpha_threshold (float): Alpha threshold to mask pixels where the Gaussian splat model is transparent
            (usually indicating the background). Default is 0.1.
        image_downsample_factor (int): Factor by which to downsample the rendered images for depth estimation.
            Default is 1, *i.e.* no downsampling.
        dtype (torch.dtype): Data type for the TSDF grid values. Default is ``torch.float16``.
        feature_dtype (torch.dtype): Data type for the color features. Default is ``torch.uint8``.
        dlnr_backbone (str): Backbone to use for the DLNR model, either ``"middleburry"`` or ``"sceneflow"``.
            Default is ``"middleburry"``.
        use_absolute_baseline (bool): If ``True``, treat the provided baseline as an absolute distance in world units.
            If ``False``, treat the baseline as a fraction of the mean depth of each image estimated using the
            Gaussian splat radiance field. Default is ``False``.
        show_progress (bool): Whether to show a progress bar during processing. Default is ``True``.
        num_workers (int): Number of workers to use for loading data generated by DLNR. Default is 8.
        dlnr_cache_path (str | pathlib.Path | None): Optional cache directory for intermediate DLNR TSDF
            inputs (depth, RGB, weights). If ``None``, a fresh temporary directory is created for this call
            (not reused across calls). When the cache already holds every view, DLNR inference is skipped
            entirely. The cache is keyed only by this path, so pass a per-dataset directory if you want it
            reused across runs - a shared path silently reuses one scene's depths for another.
        fusion_foreground_mask_paths (list[str | None] | None): Optional per-view mask paths for
            streaming TSDF fusion gating. Each entry is either a mask image path or ``None``.

    Returns:
        accum_grid (Grid): The accumulated :class:`fvdb.Grid` representing the voxels in the TSDF volume.
        tsdf (torch.Tensor): The TSDF values for each voxel in the grid.
        colors (torch.Tensor): The colors/features for each voxel in the grid.
    """

    if model.num_channels != 3:
        raise ValueError(f"Expected model with 3 channels, got {model.num_channels} channels.")

    if grid_shell_thickness <= 1.0:
        raise ValueError("grid_shell_thickness must be greater than 1.0")

    camera_to_world_matrices, projection_matrices, image_sizes = validate_camera_matrices_and_image_sizes(
        camera_to_world_matrices, projection_matrices, image_sizes
    )
    camera_models = validate_pinhole_camera_models(
        camera_models, camera_to_world_matrices.shape[0], operation_name="TSDF fusion"
    )

    if image_downsample_factor > 1:
        image_sizes = image_sizes // image_downsample_factor
        projection_matrices = projection_matrices.clone()
        projection_matrices[:, :2, :] /= image_downsample_factor

    cache_path = (
        pathlib.Path(dlnr_cache_path)
        if dlnr_cache_path is not None
        # No fixed workspace layout to fall back to here (this used to assume RunPod's
        # /workspace convention). A silently-reused fixed path is also the exact footgun the
        # docstring above warns against, so give every uncached call its own directory instead.
        else pathlib.Path(tempfile.mkdtemp(prefix="tsdf_dlnr_cache_"))
    )

    # A populated cache lets a re-run skip DLNR inference entirely, which dominates wall-clock
    # time. Only construct the (expensive) DLNR model when something is actually missing.
    cache = SfmCache.get_cache(cache_path, "TSDFInputs", "Cache for TSDF inputs")

    def _cache_has_all_required_inputs() -> bool:
        num_views = int(camera_to_world_matrices.shape[0])
        for i in range(num_views):
            for suffix in ("rgb", "depth", "weight"):
                try:
                    cache.read_file(f"{suffix}_{i}")
                except (FileNotFoundError, ValueError):
                    return False
        return True

    if _cache_has_all_required_inputs():
        print(f"Reusing complete TSDF DLNR cache from {cache_path}.")
        dlnr_model = None
    else:
        print("Generating TSDF inputs with DLNR...")
        dlnr_model = DLNRModel(backbone=dlnr_backbone, device=model.device)

    dataset = TSDFInputDataset(
        cache_path=cache_path,
        model=model,
        camera_to_world_matrices=camera_to_world_matrices,
        projection_matrices=projection_matrices,
        image_sizes=image_sizes,
        camera_models=camera_models,
        distortion_coeffs=distortion_coeffs,
        baseline=baseline,
        near=near,
        far=far,
        reprojection_threshold=disparity_reprojection_threshold,
        alpha_threshold=alpha_threshold,
        dlnr_model=dlnr_model,
        use_absolute_baseline=use_absolute_baseline,
        show_progress=show_progress,
        fusion_foreground_mask_paths=fusion_foreground_mask_paths,
        fusion_reference_depths=fusion_reference_depths,
        rendered_depth_fallback=rendered_depth_fallback,
        warm_start_from_render=warm_start_from_render,
        dlnr_iters=dlnr_iters,
        vpp_density=vpp_density,
        vpp_alpha=vpp_alpha,
        vpp_texture_quantile=vpp_texture_quantile,
    )
    print("Done preparing TSDF inputs.")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    device = model.device
    # The voxel size is set by dividing the truncation margin by the grid shell thickness.
    # This ensures that the truncation margin spans 'grid_shell_thickness' number of voxels,
    # controlling the grid resolution and mesh quality. Adjusting grid_shell_thickness changes
    # how many voxels fit within the truncation margin, affecting surface detail.
    voxel_size = truncation_margin / grid_shell_thickness
    accum_grid = Grid.from_dense(dense_dims=1, ijk_min=0, voxel_size=voxel_size, origin=0.0, device=device)
    tsdf = torch.zeros(accum_grid.num_voxels, device=device, dtype=dtype)
    weights = torch.zeros(accum_grid.num_voxels, device=device, dtype=dtype)
    colors = torch.zeros((accum_grid.num_voxels, model.num_channels), device=device, dtype=feature_dtype)

    enumerator = tqdm.tqdm(dataloader, unit="imgs", desc="Extracting TSDF") if show_progress else dataloader

    for i, tsdf_input in enumerate(enumerator):
        cam_to_world_matrix = camera_to_world_matrices[i].to(dtype=torch.float32, device=device)
        projection_matrix = projection_matrices[i].to(dtype=torch.float32, device=device)

        rgb_image, depth_image, weight_image = tsdf_input
        if feature_dtype == torch.uint8:
            rgb_image = (rgb_image * 255).to(feature_dtype)
        else:
            rgb_image = rgb_image.to(feature_dtype)
        depth_image = depth_image.to(dtype)
        weight_image = weight_image.to(dtype)

        # Zero the depth wherever the fusion weight is zero. integrate_tsdf_with_features sizes
        # its voxel allocation from the depth image itself, so a pixel that fusion will ignore
        # still expands the grid if it carries a garbage depth. Measured on pilastro: one view
        # held a raw depth of 1.7e7 in a ~10-unit scene (DLNR produces such values in pixels the
        # occlusion/alpha masks then reject), and the integrator requested 27-30 GiB and aborted.
        # It failed at that same view on both fvdb 0.5 and 0.6, with ~19 GiB still free, and was
        # unaffected by truncation_margin and grid_shell_thickness - which is exactly what made it
        # look like a library memory-management bug rather than bad input data.
        depth_image = torch.where(weight_image > 0, depth_image, torch.zeros_like(depth_image))

        # squeeze(0) drops the DataLoader collation dim; unsqueeze(0) then adds the
        # fvdb grid-batch dim (1 for a single Grid). fvdb-core expects batched
        # inputs: projection (B, 3, 3), cam-to-world (B, 4, 4), depth (B, H, W),
        # features (B, H, W, C), weights (B, H, W).
        accum_grid, tsdf, weights, colors = accum_grid.integrate_tsdf_with_features(
            truncation_margin,
            projection_matrix.to(dtype).unsqueeze(0),
            cam_to_world_matrix.to(dtype).unsqueeze(0),
            tsdf,
            colors,
            weights,
            depth_image.squeeze(0).to(device).unsqueeze(0),
            rgb_image.squeeze(0).to(device).unsqueeze(0),
            weight_image.squeeze(0).to(device).unsqueeze(0),
        )

        if show_progress:
            assert isinstance(enumerator, tqdm.tqdm)
            enumerator.set_postfix({"accumulated_voxels": accum_grid.num_voxels})

        # Prune out zero weight voxels to save memory
        new_grid = accum_grid.pruned_grid(weights > 0.0)
        tsdf = new_grid.inject_from(accum_grid, tsdf)
        colors = new_grid.inject_from(accum_grid, colors)
        weights = new_grid.inject_from(accum_grid, weights)
        accum_grid = new_grid

        # TSDF fusion is a bit of a torture case for the PyTorch memory allocator since
        # it progressively allocates bigger tensors which don't fit in the memory pool,
        # causing the pool to grow larger and larger.
        # To avoid this, we synchronize the CUDA device and empty the cache after each image.
        del rgb_image, depth_image, weight_image
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # After integrating all the images, we prune the grid to remove empty voxels which have no weights.
    # This is done to reduce the size of the grid and speed up the marching cubes algorithm
    # which will be used to extract the mesh.
    #
    # The threshold is also the only place that separates surface from debris. A voxel's weight is
    # how much evidence accumulated for it across all views, so keeping everything above zero keeps
    # surfaces that a single view asserted once -- and a single wrong depth sample is exactly what
    # produces a detached fragment floating near the object. Measured on pilastro: 37,902 detached
    # fragments, 37,048 of them under 100 faces. Requiring agreement from more than one view removes
    # them at the source, rather than filtering the mesh afterwards.
    #
    # Note this is the FINAL prune. The per-view prune inside the loop must stay at > 0, since a
    # voxel legitimately starts with evidence from one view and accumulates more later.
    new_grid = accum_grid.pruned_grid(weights > min_voxel_weight)
    filter_tsdf = new_grid.inject_from(accum_grid, tsdf)
    filter_colors = new_grid.inject_from(accum_grid, colors)

    return new_grid, filter_tsdf, filter_colors
