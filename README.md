# Raformer: Redundancy-Aware Transformer for Video Wire Inpainting 🎬

[![Project Status: Active](https://img.shields.io/badge/Status-Active-brightgreen)](https://github.com/Suyimu/WRV2)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Welcome to the official repository for the **Raformer** project. This innovative model is engineered to address the challenge of removing wires from video sequences, showcasing its capabilities through extensive testing on our Wire Removal Video Dataset 2 (WRV2).

---

## 🚀 Latest Updates (2025.02.16)
#### **2025.02.16 - Public Beta Release!**
- **Model Inference Code & Pre-trained Weights** now available!  
  → Test Raformer on your own videos with just 4 steps!
- **Full Training Code** scheduled for release within 1 week after paper publication.

---

## ⚡ Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Suyimu/WRV2.git
cd Raformer
```

### 2. Environment Setup
```bash
# Create conda environment
conda create -n Raformer python=3.8 -y
conda activate Raformer

# Install dependencies
pip install -r requirements.txt
```

**System Requirements**:  
- CUDA >= 9.2  
- PyTorch >= 2.0  
- Torchvision >= 0.8.2  

### 3. Download Pre-trained Models
Download models from **[Release V0.1.0](https://pan.quark.cn/s/02a9c6a1d7a6)** (Password: `78NW`) and place them in `./weights/`.  

*First-time users*: Models will auto-download during initial inference if not found.

### 4. Run Inference
**Example Test** (using sample data in [`inputs/`](./inputs)):
```bash
python inference_Raformer.py \
  --video inputs/wire_removal/GT/8m56s \
  --mask inputs/wire_removal/Wire_Masks/8m56s
```

**Custom Video Processing**:  
Prepare your data as:
```
📁 your_video/
  ├── frame_0001.png
  ├── frame_0002.png
  └── ... (sequential frames)
📁 your_mask/  # Optional for multi-mask scenarios
  ├── mask_0001.png
  ├── mask_0002.png
  └── ... (binary masks, white=wire)
```
Run with:
```bash
python inference_Raformer.py --video path/to/frames --mask path/to/masks
```
---
# **Wire Removal Video Datasets 2 (WRV2)**

The WRV2 dataset is meticulously assembled to support developing and evaluating video inpainting algorithms aimed specifically at wire removal. This challenging task is critical for enhancing visual aesthetics in various scenes.


**To download the WRV2 dataset**, please visit this [download link](https://pan.quark.cn/s/63522988eedf).

For those requiring higher resolution for detailed analysis, a 4K high-definition version of the original videos is available. Due to the large size of these files, approximately 2TB, it is not feasible to offer direct downloads. Please contact the authors directly to access these files or discuss potential delivery methods.
## Dataset folder structure
The dataset is organized as follows:
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
For a detailed look at the dataset's structure and contents, please refer to the **example_for_WRV2**.

## Dataset Example
The following image illustrates various scenes from the WRV2 dataset, highlighting the diversity and complexity of environments in our wire removal challenges:
<div style="text-align: center; padding: 10px;">
    <img src="dataset_example.png" alt="Example Image 1: Original Image" style="width: 100%; height: auto;">
    <p><strong>Figure 1:</strong> Example Image 1 - Original Image from WRV2 Dataset</p>
</div>

## Video Demonstration

For a practical insight into the capabilities of the Raformer model and the challenges posed by wire artifacts in videos, we invite you to view our demonstration video. This video visually explains the preprocessing, challenges, and the effectiveness of the Raformer model in removing wires from video footage effectively.
</ol>
<div><video controls src="https://private-user-images.githubusercontent.com/101324047/323496512-324f307f-79ef-4aab-980c-8f0841d623cf.mp4" muted="false"></video></div>

## Looking for WRV?
If you are interested in exploring our previous dataset, the Wire Removal Video Dataset 1 (WRV1), please visit the following link for more information and resources:
### [https://github.com/Suyimu/Wire-removal-Dataset-for-video-inpainting]

## Citation
If this research benefits your work or involves the use of this dataset, please consider citing the following paper:
   ```bibtex
@misc{ji2024raformer,
      title={Raformer: Redundancy-Aware Transformer for Video Wire Inpainting}, 
      author={Zhong Ji and Yimu Su and Yan Zhang and Jiacheng Hou and Yanwei Pang and Jungong Han},
      year={2024},
      eprint={2404.15802},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License
This project is licensed under the [MIT License](LICENSE). Please refer to the LICENSE file for detailed terms.

## Acknowledgement

This code is based on [E<sup>2</sup>FGVI](https://github.com/MCG-NKU/E2FGVI) and [Propainter](https://github.com/sczhou/ProPainter).

