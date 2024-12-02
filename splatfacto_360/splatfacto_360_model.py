# ruff: noqa: E741
# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Gaussian Splatting implementation that combines many recent advancements.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type, Union

import numpy as np
import torch

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
except ImportError:
    print("Please install diff_gaussian_rasterization")

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.models.splatfacto import (
    SplatfactoModel,
    SplatfactoModelConfig,
    get_viewmat,
)


def getWorld2View2(R, t, translate=torch.tensor([0.0, 0.0, 0.0]), scale=1.0):
    Rt = torch.zeros((4, 4))
    Rt[:3, :3] = R.transpose(0, 1)
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = torch.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = torch.linalg.inv(C2W)
    return Rt


def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


@dataclass
class Splatfacto360ModelConfig(SplatfactoModelConfig):
    """Splatfacto-360 Model Config, nerfstudio's implementation of Gaussian Splatting"""

    _target: Type = field(default_factory=lambda: Splatfacto360Model)


class Splatfacto360Model(SplatfactoModel):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto-360 configuration to instantiate model
    """

    config: Splatfacto360ModelConfig

    def after_train(self, step: int):
        assert step == self.step
        if self.step >= self.config.stop_split_at:
            return

        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()

            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(
                    self.num_points, device=self.device, dtype=torch.float32
                )
                self.vis_counts = torch.ones(
                    self.num_points, device=self.device, dtype=torch.float32
                )
            assert self.vis_counts is not None

            grads = self.xys_grad_norm / self.vis_counts
            grads[grads.isnan()] = 0.0
            self.vis_counts[visible_mask] += 1
            self.xys_grad_norm[visible_mask] += grads[visible_mask]

            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    def get_outputs(
        self, camera: Cameras, scaling_modifier: float = 1.0
    ) -> Dict[str, Union[torch.Tensor, List]]:
        """Takes in a Ray Bundle and returns a dictionary of outputs.

        Args:
            ray_bundle: Input bundle of rays. This raybundle should have all the
            needed information to compute the outputs.

        Returns:
            Outputs of model. (ie. rendered colors)
        """
        if not isinstance(camera, Cameras):
            print("Called get_outputs with not a camera")
            return {}

        # cropping
        if self.crop_box is not None and not self.training:
            crop_ids = self.crop_box.within(self.means).squeeze()
            if crop_ids.sum() == 0:
                return self.get_empty_outputs(
                    int(camera.width.item()),
                    int(camera.height.item()),
                    self.background_color,
                )
        else:
            crop_ids = None

        if crop_ids is not None:
            opacities_crop = self.opacities[crop_ids]
            means_crop = self.means[crop_ids]
            features_dc_crop = self.features_dc[crop_ids]
            features_rest_crop = self.features_rest[crop_ids]
            scales_crop = self.scales[crop_ids]
            quats_crop = self.quats[crop_ids]
        else:
            opacities_crop = self.opacities
            means_crop = self.means
            features_dc_crop = self.features_dc
            features_rest_crop = self.features_rest
            scales_crop = self.scales
            quats_crop = self.quats

        colors_crop = torch.cat(
            (features_dc_crop[:, None, :], features_rest_crop), dim=1
        )

        # NeRF 'transform_matrix' is a camera-to-world transform
        c2w = torch.eye(4).to(camera.camera_to_worlds)
        c2w[:3, :] = camera.camera_to_worlds[0]
        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1

        # get the world-to-camera transform and set R, T
        w2c = torch.linalg.inv(c2w)
        R = w2c[:3, :3].T  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)
        K = camera.get_intrinsics_matrices().cuda()
        W, H = int(camera.width.item()), int(camera.height.item())
        self.last_size = (H, W)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        fovx = focal2fov(K[0, 0, 0], W)
        fovy = focal2fov(K[0, 1, 1], H)
        tanfovx = math.tan(fovx * 0.5)
        tanfovy = math.tan(fovy * 0.5)

        world_view_transform = (
            torch.tensor(getWorld2View2(R, T)).transpose(0, 1).unsqueeze(0).cuda()
        )
        projection_matrix = (
            getProjectionMatrix(znear=0.01, zfar=1e10, fovX=fovx, fovY=fovy)
            .transpose(0, 1)
            .cuda()
        )
        full_proj_transform = (
            world_view_transform.bmm(projection_matrix.unsqueeze(0))
        ).squeeze(0)
        camera_center = world_view_transform.squeeze(0).inverse()[3, :3]

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(
                self.step // self.config.sh_degree_interval, self.config.sh_degree
            )
        else:
            colors_crop = torch.sigmoid(colors_crop)
            sh_degree_to_use = None

        raster_settings = GaussianRasterizationSettings(
            image_height=H,
            image_width=W,
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=self._get_background_color(),
            scale_modifier=scaling_modifier,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=sh_degree_to_use,
            campos=camera_center,
            prefiltered=False,
            spherical=False,
            debug=False,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = (
            torch.zeros_like(
                means_crop, dtype=means_crop.dtype, requires_grad=True, device="cuda"
            )
            + 0
        )
        try:
            screenspace_points.retain_grad()
        except:
            pass

        rendered_image, radii = rasterizer(
            means3D=means_crop,
            means2D=screenspace_points,
            shs=colors_crop,
            colors_precomp=None,
            opacities=torch.sigmoid(opacities_crop),
            scales=torch.exp(scales_crop),
            rotations=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            cov3D_precomp=None,
        )

        rendered_image = rendered_image.permute(1, 2, 0).squeeze(0)

        self.xys = screenspace_points.unsqueeze(0)  # [1, N, 2]
        self.radii = radii  # [N]

        fake_depth = torch.zeros((H, W, 1)).cuda()

        return {
            "rgb": rendered_image,  # type: ignore
            "depth": fake_depth,  # type: ignore
            "accumulation": None,  # type: ignore
            "background": None,  # type: ignore
        }  # type: ignore

    def composite_with_background(self, image, background) -> torch.Tensor:
        """Composite the ground truth image with a background color when it has an alpha channel.

        Args:
            image: the image to composite
            background: the background color
        """
        return image[..., :3]
