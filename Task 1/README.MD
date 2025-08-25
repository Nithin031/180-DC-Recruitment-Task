# Recruitment Task Report – Image Denoising and Flower Classification
**Author**: Nithin N  
**Department**: EEE, Sophomore, NITK  

---

## 1. Personal Journey
First of all, I would like to thank the team for providing such wonderful tasks. Most of them were completely new to me, and this gave me the opportunity to explore, experiment, and learn.  

For **Task 1 (Image Denoising)**, this was my very first time working on such a problem. I started with almost no prior knowledge and explored GitHub repositories and Kaggle kernels to understand the domain.  

- **Initial Attempts**: Began with a simple Autoencoder model and got a validation accuracy of 0.62.  
- **Experimentation**: Tried UNet, SCUNet, Converse CNN, and Noise2Void but performance was inconsistent (~0.59).  
- **Breakthrough**: Following team resources, I implemented **DnCNN**, reaching **0.7044 accuracy**.  
- **Final Model**: Returned to my baseline Autoencoder, added Residual connections, and achieved **0.76424 accuracy**.  

This journey taught me the importance of systematic experimentation, reproducibility, and proper evaluation rather than simply chasing scores.

---

## 2. Technical Pipeline
The solution is a **two-stage modular pipeline**:

1. **Denoising Stage** → Clean images reconstructed from noisy inputs.  
2. **Classification Stage** → CNN trained on clean flower images, later used to classify denoised test images.  

This separation ensures robustness and generalization.  

---

### 2.1 Data and Setup
- **Dataset Layout**:  
  - `train/noisy` & `train/clean` → paired training images.  
  - `test/noisy` → unlabeled test images.  
- **Preprocessing**: All resized to `128×128`.  
- **Reproducibility**: Fixed seeds, deterministic PyTorch.  
- **Environment**: CUDA GPU (if available).  
- **Outputs**: Weights, logs, denoised images, CSVs saved to `/kaggle/working`.

---

### 2.2 Stage 1 – Image Denoising
**Model**: Residual Denoising Autoencoder (light UNet style)  
- Encoder → Bottleneck → Decoder with Residual Blocks.  
- Skip connections for stability.  
- Final Conv + Sigmoid for RGB output.  

**Training Setup**:
- Loss: `MSE`  
- Optimizer: `Adam (lr=1e-3)`  
- Scheduler: `ReduceLROnPlateau`  
- Epochs: 150 max  
- Batch size: 32  
- Validation split: 80/20  

**Evaluation**:
- Metrics: `PSNR` & `SSIM`  
- Visual outputs: noisy → clean → denoised comparisons.  

---

### 2.3 Stage 2 – Flower Classification
**Model**: Compact CNN Classifier  
- Conv-BN-ReLU stacks (64 → 128 → 256 → 512).  
- Classifier: Linear(512→256→5), ReLU, Dropout(0.5).  

**Training Setup**:
- Loss: CrossEntropy with label smoothing (0.1).  
- Optimizer: AdamW (lr=1e-3, wd=1e-4).  
- Early stopping (patience=10).  
- Checkpointing best val accuracy.  

**Inference**:
- Classified **denoised test images**.  
- Class mapping:  
  - daisy=1, dandelion=2, roses=3, sunflowers=4, tulips=5  
- Submission: `test_labels.csv`  

---

## 3. Results & Observations

### 3.1 Denoising Performance
- **TEST mean PSNR**: `13.81 dB`  
- **TEST mean SSIM**: `0.2367`  

Note: We didn’t compute “before” PSNR/SSIM since comparing noisy with itself gives trivial (∞, 1.0).  

- Validation metrics were higher (since ground truth clean images exist).  
- Test metrics reflect divergence from noise, not true image quality.  

---

### 3.2 Classification Performance
- **Validation Accuracy**: `0.8273`  
- **Final Kaggle Score**: `0.76424`  
- **Consistency**: Stable predictions on denoised inputs.  

---

## 4. Limitations & Future Work
- **Loss Function**: Try perceptual loss (VGG/SSIM) for finer textures.  
- **Models**: Use Attention U-Net or SCUNet for detail recovery.  
- **Classifier**: Replace CNN with pretrained backbones (ResNet18, MobileNetV3).  

---

## 5. Conclusion
The project successfully demonstrates a **modular denoising + classification pipeline**:  

- Residual autoencoder effectively removes noise while retaining structure.  
- Compact CNN classifier achieves stable classification.  
- Metrics (PSNR = 13.81 dB, SSIM = 0.2367, Accuracy = 0.76424) validate performance.  

Overall, this work shows a practical and extendable approach to noisy flower image recognition, with clear potential for future enhancements.  
