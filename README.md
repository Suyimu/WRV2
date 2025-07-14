# Raformer: Redundancy-Aware Transformer for Video Wire Inpainting 🎬

[![Project Status: Active](https://img.shields.io/badge/Status-Active-brightgreen)](https://github.com/Suyimu/WRV2)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

Welcome to the official repository for the **Raformer** project. This innovative model is engineered to address the challenge of removing wires from video sequences, showcasing its capabilities through extensive testing on our Wire Removal Video Dataset 2 (WRV2).

---


## 🚀 Latest Updates (2025.03.23)
#### **2025.03.22 - Official Paper Release & System Updates!**
- 🎉 **Paper Published in IEEE TIP!**  
  → Our work "*Raformer*" is now officially published in *IEEE Transactions on Image Processing*!  
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


##  **Training Workflow Overview**  
1. **Base train code**
 - Please refer to [ProPainter](https://github.com/sczhou/ProPainter)
2. **Critical Adaptation**
 - Replace ProPainter's default mask generator with the **wire-shaped mask function** from `Raformer/core/utils.py`.  
（→ Training implementation: **Modified ProPainter workflow** with wire-mask generation 🔄  ）
3. **Data Preparation**  
   - Configure the WRV2 dataset (no pre-generated masks required).
4. **Launch Training**  
   - Run 350K iterations with dynamic wire-mask generation enabled.  
*Designed for seamless integration and minimal code modification.*

  
# 🎥 Wire Removal Video Dataset 2 (WRV2) 
<div align="center">
  <img src="dataset_example.png" width="80%" alt="WRV2 Dataset Samples">
  <p><em>Fig.1 - Example Image 1: Original Image</em></p>
</div>

## 📥 Download Links
| Version | Size | Link | Access |
|---------|------|------|--------|
| Standard | 53GB | [Quark Cloud]() | ❗Download suspended |
| 4K UHD | ~2TB | Contact Authors | Special Arrangement |

**Need higher resolution?**  
For research requiring ultra-high-definition analysis, contact us via [suyimu@tju.edu.cn] to discuss transfer options for our 4K master videos.

We are very sorry that due to the company's copyright issues, the dataset has been temporarily suspended from open source. We are making every effort to communicate and strive for an early open source

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
@ARTICLE{10930654,
  author={Ji, Zhong and Su, Yimu and Zhang, Yan and Hou, Jiacheng and Pang, Yanwei and Han, Jungong},
  journal={IEEE Transactions on Image Processing}, 
  title={Raformer: Redundancy-Aware Transformer for Video Wire Inpainting}, 
  year={2025},
  volume={34},
  number={},
  pages={1795-1809},
  keywords={Wire;Transformers;Films;Visualization;Training;TV;Production;Manuals;Context modeling;Computational modeling;Video wire inpainting;video inpainting;video transformer;redundant elimination},
  doi={10.1109/TIP.2025.3550038}}
```

## 📃 License
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
This dataset is released for **academic research only**. Commercial use requires written permission.

## 🙏 Acknowledgement
We build upon these excellent works:
- [E²FGVI](https://github.com/MCG-NKU/E2FGVI) - Towards An End-to-End Framework for Flow-Guided Video Inpainting
- [ProPainter](https://github.com/sczhou/ProPainter) - ProPainter: Improving Propagation and Transformer for Video Inpainting

<div align="center">
  <p>✨ The WRV2 project is maintained by <a href="https://mp.weixin.qq.com/s/ThSF4cpnCYPZ_DHiBtviBA">Multimedia Understanding Laboratory, Tianjin University</a> ✨</p>
  <a href="https://github.com/Suyimu/WRV2/stargazers"><img src="https://img.shields.io/github/stars/Suyimu/WRV2?style=social"></a>
  <a href="https://github.com/Suyimu/WRV2/issues"><img src="https://img.shields.io/github/issues/Suyimu/WRV2"></a>
</div>

