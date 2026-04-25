# Attention-ResUNet for Automated Fetal Head Segmentation

> Official code for "Attention-ResUNet for Automated Fetal Head Segmentation", published at ANTIC 2025 (Springer). Best Paper Award, Image Processing Track.

[![arXiv](https://img.shields.io/badge/arXiv-2604.18148-b31b1b.svg)](https://arxiv.org/abs/2604.18148)
[![ANTIC 2025](https://img.shields.io/badge/ANTIC_2025-Best_Paper-gold)](https://arxiv.org/abs/2604.18148)
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](https://github.com/Ammar-ss/Attention-ResUnet/blob/main/LICENSE)

**Authors:** Ammar Bhilwarawala, Madhuchhanda Bandyopadhyay
**Affiliation:** School of Computer Engineering, KIIT University, Bhubaneswar, India

**Venue:** 5th International Conference on ANTIC 2025, Gwalior

---

## What this is

Fetal head circumference (HC) is one of the most important biometric markers in obstetric ultrasound as it tracks fetal growth, estimates gestational age, and flags intrauterine growth restriction. Manual measurement is time-consuming, operator-dependent, and subject to inter-observer variability. Automated segmentation of the fetal head from 2D ultrasound images is the prerequisite for reliable automated HC measurement.

This paper proposes Attention-ResUNet, an encoder-decoder architecture that combines multi-scale attention gates at four decoder levels with residual skip connections. The attention gates suppress irrelevant activations in the skip connections before they reach the decoder, directing the model toward the head boundary rather than surrounding tissue. The residual connections address gradient degradation in the deeper encoder layers.

Evaluated on the public HC18 benchmark, the model achieves DSC = 99.30% with zero false-negative segmentations, meaning that it never misses the fetal head entirely across the 200-image test set.

---

## Results

Evaluated on HC18 (n=200 test images).

| Metric | Attention-ResUNet | Best Baseline | Improvement |
|---|---|---|---|
| DSC (%) | 99.30 +/- 0.14 | 97.47 | +1.83 pp |
| Hausdorff Distance (mm) | 2.14 | 4.87 | -2.73 mm |
| False-negative segmentations | 0 | -- | -- |
| Inference time | 23 ms | -- | RTX 3080 |

Statistical significance: p < 0.001 (paired t-test vs all 5 baselines). Cohen's d up to 13.159.

**Explainability:** Grad-CAM spatial concentration index = 0.942 vs 0.784 for the baseline, confirming the attention gates are directing the model toward the correct anatomical region.

**Efficiency:** 14.7M parameters, 45 GFLOPs, 23 ms inference on RTX 3080.

Baselines compared: U-Net, Attention U-Net, ResU-Net, TransUNet, and Swin-UNet.

---

## Architecture

The model follows a standard encoder-decoder structure with two key modifications applied jointly:

**Multi-scale attention gates** are placed at all four decoder levels (512, 256, 128, 64 channels). Each gate takes the upsampled decoder feature map and the corresponding encoder skip connection as inputs, computes a sigmoid attention map, and multiplies it elementwise into the skip features before concatenation. This suppresses background activations, tissue structures that share texture with the head boundary, before they propagate into the decoder.

**Residual skip connections** in the encoder replace plain convolutional blocks with residual blocks. Each block adds a shortcut connection around two 3x3 convolutions with batch normalisation and ReLU. This addresses gradient degradation in the deeper encoder stages and improves feature reuse.

The design argument is that attention and residual learning are complementary: attention gates control *what* information flows through the skip connections, while residual connections control *how* well the encoder learns to represent it. The ablation confirms they are stronger together than either mechanism alone.

```
Input (1, 256, 256)
    |
Encoder:
  Block 1: ResConv (64ch)  --> AttGate 1
  Block 2: ResConv (128ch) --> AttGate 2
  Block 3: ResConv (256ch) --> AttGate 3
  Block 4: ResConv (512ch) --> AttGate 4
    |
Bottleneck (1024ch)
    |
Decoder:
  Up4 + AttGate4(skip) --> ResConv (512ch)
  Up3 + AttGate3(skip) --> ResConv (256ch)
  Up2 + AttGate2(skip) --> ResConv (128ch)
  Up1 + AttGate1(skip) --> ResConv (64ch)
    |
Output Conv (1ch) --> Sigmoid
```

---

## Dataset

**HC18** is a public benchmark for fetal head circumference measurement in 2D ultrasound images.

| Split | Images |
|---|---|
| Training | 999 |
| Test | 335 (200 used for evaluation in this work) |

Download: [grand-challenge.org/HC18](https://grand-challenge.org/HC18)

Images are 800x540 pixels, single-channel ultrasound scans from multiple acquisition centres. Preprocessing: resized to 256x256, normalised to [0,1], augmented with random horizontal/vertical flips, rotation (+/-15 degrees), and Gaussian noise during training only.

---

## Repo Contents

```
Attention-ResUNet/
├── attention-res-unet.ipynb   # Full training, evaluation, ablation, Grad-CAM
├── LICENSE                    # Apache 2.0
└── README.md
```

The notebook contains:
- Full model implementation (encoder, attention gates, residual blocks, decoder)
- Training loop with early stopping
- Evaluation on HC18 test set (DSC, HD, sensitivity, specificity)
- Four-configuration ablation (baseline U-Net / +residual / +attention / full model)
- Grad-CAM visualisation and spatial concentration index computation
- Comparison against all 5 baselines under identical training conditions

---

## How to Run

Designed to run on Kaggle with a T4 GPU (free tier).

**1. Get the dataset**

Download HC18 from [grand-challenge.org/HC18](https://grand-challenge.org/HC18). Upload to Kaggle dataset storage and update the path in Cell 2.

**2. Install dependencies**

```bash
pip install torch torchvision segmentation-models-pytorch \
    numpy pandas matplotlib scikit-learn opencv-python tqdm
```

**3. Run**

Open `attention-res-unet.ipynb` on Kaggle and run all cells. Training runs for up to 100 epochs with patience=15. Full training takes approximately 1.5 hours on a T4.

---

## Ablation Summary

| Configuration | DSC (%) | HD (mm) |
|---|---|---|
| Baseline U-Net | 96.81 | 5.92 |
| + Residual blocks only | 97.94 | 4.61 |
| + Attention gates only | 98.12 | 4.03 |
| Full Attention-ResUNet | **99.30** | **2.14** |

Both components contribute independently. Together they produce a result stronger than either alone -- the compositional argument central to the paper.

---

## Citation

If you use this code or build on this work, please cite:

```bibtex
@inproceedings{bhilwarawala2025attentionresunet,
  title     = {Attention-{ResUNet} for Automated Fetal Head Segmentation},
  author    = {Bhilwarawala, Ammar and Bandyopadhyay, Madhuchhanda},
  booktitle = {Proceedings of the 5th International Conference on ANTIC 2025},
  series    = {Lecture Notes in Computer Science},
  publisher = {Springer},
  year      = {2025},
  note      = {Best Paper Award, Image Processing Track}
}
```

arXiv preprint: [arxiv.org/abs/2604.18148](https://arxiv.org/abs/2604.18148)

---

## Related Work

This is the first paper in a line of work on fetal ultrasound analysis:

- **FETALFusion** (MICCAI 2026, under review) -- extends to multi-domain segmentation with a dual-path CNN-Mamba encoder and resolution-aware SSM scanning. DSC = 0.9635 on HC18+PSFHS joint training.
- **FETALFusion-v2** (in preparation, Medical Image Analysis) -- simultaneous segmentation and landmark heatmap regression for automated fetal biometry across 5 public datasets.

---

## Contact

Ammar Bhilwarawala · [ammarbhilwarawala@gmail.com](mailto:ammarbhilwarawala@gmail.com) · [Hugging Face](https://huggingface.co/Ammar-ss) · [LinkedIn](https://linkedin.com/in/ammar-bww)
