# Images and pixel labels

The Soybean dataset provides original images and annotations in flat directories:

```text
images/
masks/
annotations_json/
metadata.csv
```

It contains 1,567 distinct original images. Of these, 172 have manual pixel masks and LabelMe polygons; the remaining images have image-level labels. `metadata.csv` links the image and annotation filenames and records disease and scene metadata.

The separate 617-pair Soybean package contains only `images/` and `masks/`. Matching images and masks share the same filename stem.

| Value | Meaning |
| --- | --- |
| 0 | Background |
| 1 | Healthy leaf tissue |
| 2 | Lesion |
| 255 | Ignore |

The binary leaf model merges healthy leaf and lesion into the foreground leaf class. Exported probability maps use uint16 encoding: `round(probability × 65535)`.

The MH-SoyaHealthVision package retains its own directory layout and source notice.
