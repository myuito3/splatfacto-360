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

from dataclasses import dataclass, field
from typing import Dict, List, Type, Union

import torch

from nerfstudio.cameras.cameras import CameraType, Cameras
from nerfstudio.models.splatfacto import SplatfactoModel, SplatfactoModelConfig
from splatfacto_360.gaussian_splatting.cameras import convert_to_colmap_camera
from splatfacto_360.gaussian_splatting.gaussian_renderer import render


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
        # to save some training time, we no longer need to update those stats post refinement
        if self.step >= self.config.stop_split_at:
            return
        with torch.no_grad():
            # keep track of a moving average of grad norms
            visible_mask = (self.radii > 0).flatten()
            grads = torch.norm(self.xys.grad[visible_mask, :2], dim=-1)
            # print(f"grad norm min {grads.min().item()} max {grads.max().item()} mean {grads.mean().item()} size {grads.shape}")
            if self.xys_grad_norm is None:
                self.xys_grad_norm = torch.zeros(
                    self.num_points, device=self.device, dtype=torch.float32
                )
                self.vis_counts = torch.ones(
                    self.num_points, device=self.device, dtype=torch.float32
                )
            assert self.vis_counts is not None
            self.vis_counts[visible_mask] += 1
            self.xys_grad_norm[visible_mask] += grads
            # update the max screen size, as a ratio of number of pixels
            if self.max_2Dsize is None:
                self.max_2Dsize = torch.zeros_like(self.radii, dtype=torch.float32)
            newradii = self.radii.detach()[visible_mask]
            self.max_2Dsize[visible_mask] = torch.maximum(
                self.max_2Dsize[visible_mask],
                newradii / float(max(self.last_size[0], self.last_size[1])),
            )

    def get_outputs(self, camera: Cameras) -> Dict[str, Union[torch.Tensor, List]]:
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

        camera_scale_fac = self._get_downscale_factor()
        camera.rescale_output_resolution(1 / camera_scale_fac)

        colmap_camera = convert_to_colmap_camera(camera)
        self.last_size = (colmap_camera.image_height, colmap_camera.image_width)
        camera.rescale_output_resolution(camera_scale_fac)  # type: ignore

        if self.config.sh_degree > 0:
            sh_degree_to_use = min(
                self.step // self.config.sh_degree_interval, self.config.sh_degree
            )
        else:
            colors_crop = torch.sigmoid(colors_crop)
            sh_degree_to_use = None

        # TODO: novel view should be rendered with a perspective camera
        spherical = True  # camera.camera_type == CameraType.EQUIRECTANGULAR

        render_outputs = render(
            viewpoint_camera=colmap_camera,
            means=means_crop,
            features=colors_crop,
            scales=torch.exp(scales_crop),
            rotations=quats_crop / quats_crop.norm(dim=-1, keepdim=True),
            opacities=torch.sigmoid(opacities_crop),
            active_sh_degree=sh_degree_to_use,
            bg_color=self._get_background_color(),
            spherical=spherical,
        )

        if self.training and render_outputs["viewspace_points"].requires_grad:
            render_outputs["viewspace_points"].retain_grad()
        self.xys = render_outputs["viewspace_points"]
        self.radii = render_outputs["radii"]

        rgb = render_outputs["render"].permute(1, 2, 0).squeeze(0)
        fake_depth = torch.ones((*self.last_size, 1)).cuda() * 1000

        return {
            "rgb": rgb,  # type: ignore
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
