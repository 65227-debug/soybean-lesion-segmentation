# Installation

Run commands from the repository root in an isolated Python environment.

The supported environment uses Python 3.9, PyTorch 2.8.0, torchvision 0.23.0, MMEngine 0.10.7, MMCV 2.1.0 and MMSegmentation 1.2.2. The MMCV binary must match the selected PyTorch/CUDA combination. Dependency compatibility is checked by `mmseg/__init__.py`.

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
python scripts/check_model_forward.py
```

Python dependencies are listed in `requirements/paper.txt`. GPU packages in `requirements/paper-gpu-reference.txt` are an environment reference, not a universal binary installation recipe. Supply compatible checkpoints when using the inference commands in the README.
