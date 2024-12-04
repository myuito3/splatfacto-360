# splatfacto-360: A Nerfstudio Implementation of Splatfacto with 360-diff-gaussian-rasterization
This repository is an **UNOFFICIAL** implementation of incorporating the [360-diff-gaussian-rasterization](https://github.com/inuex35/360-diff-gaussian-rasterization) into nerfstudio's splatfacto. This rasterizer has been implemented by inuex35 and can rasterize a equirectangular image from 3D Gaussians.

## Installation

### Dependencies
We tested the code in the following configurations. Create an environment running this version of nerfstudio in advance.
- `torch == 2.0.1+cu118`
- `torchvision == 0.15.2+cu118`
- `nerfstudio == 1.1.3`

### Install requirements
```bash
pip install git+https://github.com/inuex35/360-diff-gaussian-rasterization.git
pip install tyro==0.8.14
```

### Clone and install this repository
```bash
git clone https://github.com/myuito3/splatfacto-360.git splatfacto_360
cd splatfacto_360
python -m pip install -e .
```

## Training

### Basic command
```bash
ns-train splatfacto-360 --data <data_folder>
```
