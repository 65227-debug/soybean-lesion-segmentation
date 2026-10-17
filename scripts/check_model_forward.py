#!/usr/bin/env python3
"""Check all three segmentation model graphs using zeros and no weights/data."""

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    import torch
    import torch.nn.functional as F
    from mmengine import Config
    from mmseg.registry import MODELS
    from mmseg.utils import register_all_modules
    import mmseg

    register_all_modules(init_default_scope=True)
    import projects.dadou_teacher  # noqa: F401
    if not Path(mmseg.__file__).resolve().is_relative_to(ROOT):
        raise RuntimeError("Model check must import this release's mmseg package")
    configurations = [
        ("student", "configs/soybean/student.py", 3),
        ("teacher", "configs/soybean/teacher.py", 3),
        ("leaf", "configs/soybean/leaf.py", 2),
    ]
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    reports = []
    with torch.no_grad():
        for name, path, classes in configurations:
            cfg = Config.fromfile(str(ROOT / path))
            # This is only an architecture check: never download pretrained weights.
            cfg.model.backbone.pop("init_cfg", None)
            if "pretrained" in cfg.model:
                cfg.model.pretrained = None
            model = MODELS.build(cfg.model).to(device).eval()
            image = torch.zeros(1, 3, 512, 512, device=device)
            logits = model.decode_head.forward(model.extract_feat(image))
            logits = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
            if tuple(logits.shape) != (1, classes, 512, 512) or not torch.isfinite(logits).all():
                raise AssertionError(f"Invalid output for {name}: {logits.shape}")
            reports.append({"component": name, "output_shape": list(logits.shape), "finite": True})
            del model, image, logits
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    print(json.dumps({
        "status": "PASS", "mmseg_version": mmseg.__version__,
        "release_source_used": True, "device": device, "models": reports,
        "weights_loaded": False, "pretrained_downloaded": False,
        "training_performed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
