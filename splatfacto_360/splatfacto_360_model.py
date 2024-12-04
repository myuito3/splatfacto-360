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
from torchvision.utils import save_image

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

        c2w = camera.camera_to_worlds.clone()
        c2w[0, :3, :3] = c2w[0, :3, :3] @ rad2rotmat(deg2rad(180), 0, 0).to(c2w.device)
        camera.camera_to_worlds = c2w

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

        rgb = render_outputs["render"]
        if self.step % 50 == 0:
            save_image(rgb.cpu(), f"outputs/render_{self.step}.png")

        rgb = rgb.permute(1, 2, 0).squeeze(0)
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

    def get_metrics_dict(self, outputs, batch) -> Dict[str, torch.Tensor]:
        """Compute and returns metrics.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
        """
        gt_rgb = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )

        w_2 = gt_rgb.shape[1] // 2
        rotated_gt_rgb = torch.zeros_like(gt_rgb).to(gt_rgb)
        rotated_gt_rgb[:, :w_2, :] = gt_rgb[:, w_2:, :]
        rotated_gt_rgb[:, w_2:, :] = gt_rgb[:, :w_2, :]
        gt_rgb = rotated_gt_rgb

        metrics_dict = {}
        predicted_rgb = outputs["rgb"]
        metrics_dict["psnr"] = self.psnr(predicted_rgb, gt_rgb)

        metrics_dict["gaussian_count"] = self.num_points

        self.camera_optimizer.get_metrics_dict(metrics_dict)
        return metrics_dict

    def get_loss_dict(
        self, outputs, batch, metrics_dict=None
    ) -> Dict[str, torch.Tensor]:
        """Computes and returns the losses dict.

        Args:
            outputs: the output to compute loss dict to
            batch: ground truth batch corresponding to outputs
            metrics_dict: dictionary of metrics, some of which we can use for loss
        """
        gt_img = self.composite_with_background(
            self.get_gt_img(batch["image"]), outputs["background"]
        )
        pred_img = outputs["rgb"]

        w_2 = gt_img.shape[1] // 2
        rotated_gt_rgb = torch.zeros_like(gt_img).to(gt_img)
        rotated_gt_rgb[:, :w_2, :] = gt_img[:, w_2:, :]
        rotated_gt_rgb[:, w_2:, :] = gt_img[:, :w_2, :]
        gt_img = rotated_gt_rgb

        if self.step % 50 == 0:
            save_image(gt_img.permute(2, 0, 1).cpu(), f"outputs/gt_{self.step}.png")

        # Set masked part of both ground-truth and rendered image to black.
        # This is a little bit sketchy for the SSIM loss.
        if "mask" in batch:
            # batch["mask"] : [H, W, 1]
            mask = self._downscale_if_required(batch["mask"])
            mask = mask.to(self.device)
            assert mask.shape[:2] == gt_img.shape[:2] == pred_img.shape[:2]
            gt_img = gt_img * mask
            pred_img = pred_img * mask

        Ll1 = torch.abs(gt_img - pred_img).mean()
        simloss = 1 - self.ssim(
            gt_img.permute(2, 0, 1)[None, ...], pred_img.permute(2, 0, 1)[None, ...]
        )
        if self.config.use_scale_regularization and self.step % 10 == 0:
            scale_exp = torch.exp(self.scales)
            scale_reg = (
                torch.maximum(
                    scale_exp.amax(dim=-1) / scale_exp.amin(dim=-1),
                    torch.tensor(self.config.max_gauss_ratio),
                )
                - self.config.max_gauss_ratio
            )
            scale_reg = 0.1 * scale_reg.mean()
        else:
            scale_reg = torch.tensor(0.0).to(self.device)

        loss_dict = {
            "main_loss": (1 - self.config.ssim_lambda) * Ll1
            + self.config.ssim_lambda * simloss,
            "scale_reg": scale_reg,
        }

        if self.training:
            # Add loss from camera optimizer
            self.camera_optimizer.get_loss_dict(loss_dict)

        return loss_dict


import math


def deg2rad(deg):
    return deg * math.pi / 180.0


def rad2rotmat(radx, rady, radz):
    sx = math.sin(radx)
    sy = math.sin(rady)
    sz = math.sin(radz)
    cx = math.cos(radx)
    cy = math.cos(rady)
    cz = math.cos(radz)
    Rx = torch.tensor([[1, 0, 0], [0, cx, sx], [0, -sx, cx]])
    Ry = torch.tensor([[cy, 0, -sy], [0, 1, 0], [sy, 0, cy]])
    Rz = torch.tensor([[cz, sz, 0], [-sz, cz, 0], [0, 0, 1]])
    rotmat = Rx @ Ry @ Rz
    return rotmat
