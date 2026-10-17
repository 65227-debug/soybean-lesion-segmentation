# Installation and recorded environment

Run commands from the repository root in an isolated Python environment.

The recorded working setup used Python 3.9, PyTorch 2.8.0, torchvision 0.23.0, MMEngine 0.10.7, MMCV 2.1.0 and MMSegmentation 1.2.2. It used CUDA 12.8 on an RTX 5090 D v2. The MMCV binary must be built for the selected PyTorch/CUDA combination; an arbitrary MMCV wheel is not guaranteed to work. See the included upstream installation instructions in `docs/en/get_started.md` and the dependency checks in `mmseg/__init__.py`.

```bash
conda create -n soybean-seg python=3.9 -y
conda activate soybean-seg
# Install a matching PyTorch/torchvision CUDA build for your machine first.
python -m pip install -r requirements/paper.txt
# Install MMCV 2.1.0 with compiled ops for that PyTorch/CUDA environment.
# Do not install mmcv and mmcv-lite simultaneously.
python -m pip install -e . --no-deps
python -c "import torch, mmcv, mmengine, mmseg; print(torch.__version__, mmcv.__version__, mmengine.__version__, mmseg.__version__)"
python scripts/check_release.py
```

The scientific Python versions recorded for this study are pinned in `requirements/paper.txt`. GPU packages are listed in `requirements/paper-gpu-reference.txt` as an environment reference; that file is not a universal binary installation recipe.

The teacher configuration loads a separately downloaded Pascal VOC DeepLabV3+–ResNet-50 checkpoint with the filename listed in `configs/dadou/deeplabv3plus_r50-d8_1xb4-60e_dadou-512x512.py`. Put it under `checkpoints/`. The student uses the OpenMMLab ImageNet MobileNetV3 initializer, and torchvision obtains the EfficientNet-B0 initializer separately. No pretrained or trained model is included here.

The publishing helper `scripts/publish_github.sh` requires Git and GitHub CLI (`gh`). It performs a content check, uses the current authenticated account, creates a new repository, commits the code, and pushes the main branch. An existing remote is never overwritten.
