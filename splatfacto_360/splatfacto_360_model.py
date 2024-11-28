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

from dataclasses import dataclass

try:
    from diff_gaussian_rasterization import (
        GaussianRasterizationSettings,
        GaussianRasterizer,
    )
except ImportError:
    print("Please install diff_gaussian_rasterization")

from nerfstudio.models.splatfacto import (
    SplatfactoModel,
    SplatfactoModelConfig,
)


@dataclass
class Splatfacto360ModelConfig(SplatfactoModelConfig):
    """Splatfacto-360 Model Config, nerfstudio's implementation of Gaussian Splatting"""


class Splatfacto360Model(SplatfactoModel):
    """Nerfstudio's implementation of Gaussian Splatting

    Args:
        config: Splatfacto-360 configuration to instantiate model
    """

    config: Splatfacto360ModelConfig
