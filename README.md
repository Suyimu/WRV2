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

### 4. Run Inference
**Example Test** (using sample data in [`inputs/`](./Raformer/inputs)):
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
# 🎥 Wire Removal Video Dataset 2 (WRV2) 
<div align="center">
  <img src="dataset_example.png" width="80%" alt="WRV2 Dataset Samples">
  <p><em>Fig.1 - Example Image 1: Original Image</em></p>
</div>

## 📥 Download Links
| Version | Size | Link | Access |
|---------|------|------|--------|
| Standard | 200GB | [Quark Cloud](https://pan.quark.cn/s/63522988eedf) | Direct Download |
| 4K UHD | ~2TB | Contact Authors | Special Arrangement |

**Need higher resolution?**  
For research requiring ultra-high-definition analysis, contact us via [suyimu@tju.edu.cn] to discuss transfer options for our 4K master videos.

## 🗂 Dataset Structure
```bash
WRV2/
├── train.json       # Training metadata
├── test.json        # Testing metadata
├── JPEGImages/      # Original video frames
│   └── [video_id]/  
│       ├── 00000.jpg
│       └── ...      # Sequential frames
└── test_masks/      # Binary wire masks
    └── [video_id]/  
        ├── 00000.png  # 0=background, 1=wire
        └── ...      
```
Annotations within this dataset are formatted as paletted binary images where the value 1 indicates the presence of a wire that needs to be inpainted, and 0 represents no wire.
For a detailed look at the dataset's structure and contents, please refer to the **example_for_WRV2**.

## 🎬 Video Demonstration
For a practical insight into the capabilities of the Raformer model and the challenges posed by wire artifacts in videos, we invite you to view our demonstration video. This video visually explains the preprocessing, challenges, and the effectiveness of the Raformer model in removing wires from video footage effectively.
</ol>
<div><video controls src="https://private-user-images.githubusercontent.com/101324047/323496512-324f307f-79ef-4aab-980c-8f0841d623cf.mp4" muted="false"></video></div>
**Click to watch demonstration video**

## 📚 Related Resources
### Previous Version Dataset
[![WRV1 Badge](https://img.shields.io/badge/Dataset-WRV1-blue)](https://github.com/Suyimu/Wire-removal-Dataset-for-video-inpainting)  
Explore our initial wire removal benchmark dataset with different challenge configurations.

## 📜 Citation
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

## 📃 License
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
This dataset is released for **academic research only**. Commercial use requires written permission.

## 🙏 Acknowledgement
We build upon these excellent works:
- [E²FGVI](https://github.com/MCG-NKU/E2FGVI) - Efficient flow-guided video inpainting
- [ProPainter](https://github.com/sczhou/ProPainter) - Progressive video inpainting framework

<div align="center">
  <p>✨ The WRV2 project is maintained by <a href="https://mp.weixin.qq.com/s/ThSF4cpnCYPZ_DHiBtviBA">Multimedia Understanding Laboratory, Tianjin University</a> ✨</p>
  <a href="https://github.com/Suyimu/WRV2/stargazers"><img src="https://img.shields.io/github/stars/Suyimu/WRV2?style=social"></a>
  <a href="https://github.com/Suyimu/WRV2/issues"><img src="https://img.shields.io/github/issues/Suyimu/WRV2"></a>
</div>

