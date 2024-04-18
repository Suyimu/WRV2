# Raformer: Redundancy-Aware Transformer for Video Wire Inpainting


# WRV2
Wire Removal Video Datasets 2(WRV2)
The Wire Removal Video Dataset 2 (WRV2) is crafted to facilitate the development and evaluation of video inpainting algorithms that specifically target wire removal. This dataset includes a variety of scenes where wire removal is challenging yet critical for visual aesthetics and practical applications.

## Dataset folder structure

```
Wire Removal Video Datasets 2(WRV2)
|
├── train.json
├── test.json
├── JPEGImages
│   ├── yttl18-011_091
│   │   ├── 00000.jpg
│   │   ...
│   │   └── 0000N.jpg
│   ...
│   └── 12-042706_024
│       ├── 00000.jpg
│       ...
│       └── 0000N.jpg
│
└── test_masks
    ├── yttl18-011_091
    │   ├── 00000.png
    │   ...
    │   └── 0000N.png
    ...
    └── 12-042706_024
        ├── 00000.png
        ...
        └── 0000N.png

```

Annotations within this dataset are formatted as paletted binary images where the value 1 indicates the presence of a wire that needs to be inpainted, and 0 represents no wire.
## Video Demonstration

For a practical insight into the capabilities of the Raformer model and the challenges posed by wire artifacts in videos, we invite you to view our demonstration video. This video visually explains the preprocessing, challenges, and the effectiveness of the Raformer model in removing wires from video footage effectively.
<div><video controls src="video_wire_inpainting.mp4](https://github.com/Suyimu/WRV2/issues/1#issue-2249898516.mp4" muted="false"></video></div>
This demonstration video serves as a visual guide to understanding the types of scenarios included in the WRV2 dataset and showcases the effectiveness of our model in real-world applications.
## Looking for WRV?
If you are interested in exploring our previous dataset, the Wire Removal Video Dataset 1 (WRV1), please visit the following link for more information and resources:
### [https://github.com/Suyimu/Wire-removal-Dataset-for-video-inpainting]
