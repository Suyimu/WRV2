# Raformer: Redundancy-Aware Transformer for Video Wire Inpainting


# WRV2
Wire Removal Video Datasets 2(WRV2)


## Dataset folder structure

```
Wire Removal Video Datasets 2(WRV2)
|
├── train.json
├── test.json
├── JPEGImages
│   ├── yttl18-011_091
|       ├── 00000.jpg
|       ...
│       └── 0000N.jpg
│   ...
│   └── 12-042706_024
|        ├── 00000.jpg
|        ...
│        └── 0000N.jpg
│
└── test_masks
│   ├── yttl18-011_091
|       ├── 00000.png
|       ...
│       └── 0000N.png
│   ...
│   └── 12-042706_024
|        ├── 00000.png
|        ...
│        └── 0000N.png
```

Annotations are paletted binary images with values 0 or 1.
