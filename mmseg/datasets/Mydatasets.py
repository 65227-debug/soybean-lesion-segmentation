# 同济子豪兄 2023-6-25
from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class MyDataset(BaseSegDataset):
    # 类别和对应的 RGB配色
    METAINFO = {
        'classes': ['background', 'class_1', 'class_2'],  # 修改为你的实际类别名
        'palette': [[0, 0, 0], [255, 0, 0], [0, 255, 0]]  # RGB颜色，可自定义
    }

    # 指定图像扩展名、标注扩展名
    def __init__(self,
                 seg_map_suffix='.png',
                 reduce_zero_label=False,  # 如果0是背景且要忽略，改为True
                 **kwargs) -> None:
        super().__init__(
            seg_map_suffix=seg_map_suffix,
            reduce_zero_label=reduce_zero_label,
            **kwargs)