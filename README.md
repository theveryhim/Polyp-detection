# Polyp detection 

In this repo, we are going to employ Machine learning methods to detect
colon tissues(polyp).

Here is a comprehensive summary table consolidating the performance metrics for all models and datasets provided in the README text.

## Summary

| Model | Dataset / Modality | Precision (P) | Recall (R) | mAP50 | mAP50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **YOLOv9m** | Trained (WLI) | 0.933 | **0.912** | **0.963** | **0.816** |
| | | | | | |
| **MambaYOLO** | Trained (WLI) | 0.8923 | 0.8278 | 0.9138 | 0.7264 |
| | Unseen (NBI-LCI...) | 0.8402 | 0.5672 | 0.6947 | 0.4972 |
| | | | | | |
| **YOLOv11** | Trained (WLI) | 0.8799 | 0.8638 | 0.9282 | 0.7067 |
| | Unseen (NBI-LCI...) | 0.8357 | 0.6675 | 0.7619 | 0.5383 |
| | External (REAL-Colon) | **0.4986** | 0.3413 | 0.3415 | 0.1922 |
| | | | | | |
| **YOLOv11-LoRA** | Trained (WLI) | 0.9007 | 0.8628 | 0.9351 | 0.7629 |
| | Unseen (NBI-LCI...) | 0.8608 | 0.6895 | 0.7985 | 0.6086 |
| | External (REAL-Colon) | 0.4906 | **0.3745** | **0.3665** | 0.1925 |
| | | | | | |
| **YOLOv12** | Trained (WLI) | **0.9418** | 0.8743 | 0.9503 | 0.7134 |
| | Unseen (NBI-LCI...) | **0.9438** | **0.8999** | **0.9616** | **0.7563** |
| | External (REAL-Colon) | 0.4705 | 0.3644 | 0.3530 | **0.2131** |

## Data
### PolypDB
- **WLI** (White Light Imaging): Standard endoscopic view
- **NBI** (Narrow Band Imaging): Enhanced vascular pattern visualization
- **LCI** (Linked Color Imaging): Improved color contrast
- **FICE** (Flexible Spectral Imaging Color Enhancement): Spectral enhancement
- **BLI** (Blue Laser Imaging): Surface structure enhancement
### REAL-Colon
REAL-Colon provides 60 full-resolution, real-world colonoscopy videos (2.7M frames) from multiple centers, with 350k expert-annotated polyp bounding boxes. Includes clinical metadata, acquisition details, and histopathology. Designed for robust CADe/CADx development and benchmarking. Released for non-commercial research. See the paper for details.

## YOLOv9-FineTune: 
- Train
```markdown
YOLOv9m summary (fused): 151 layers, 20,013,715 parameters, 0 gradients, 76.5 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95)
                 all        359        382      0.933      0.912      0.963      0.816
```
- detect polyps(bounding box) 
<p align="center">
    <img src="images/YOLOv9-FineTune-1.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv9-FineTune-2.png" alt="Descriptive Alt Text" class="fit-width-image">
</p>

## MambaYOLO-Train 

Mamba-YOLO merges the **state-space modeling efficiency of Mamba** with the **real-time detection strength of YOLOv8**.  
The architecture replaces the CSP backbone with a **Selective Scan (Mamba) block**, enabling long-range spatial dependency modeling at reduced computational cost.

This implementation targets **medical image analysis**, specifically **polyp detection** from multimodal colonoscopy datasets.

---

### Architecture

| Component | Description |
|------------|--------------|
| **Backbone** | Mamba-based state-space selective scan layers replacing CSP blocks |
| **Neck** | PANet-style feature pyramid |
| **Head** | YOLOv8 detection head (multi-scale anchors) |
| **Losses** | CIoU + BCE + objectness loss |
| **Training Framework** | Ultralytics YOLO API |
| **Hardware** | NVIDIA T4 (16 GB) × 2 |
| **Software Stack** | PyTorch 2.3.1 + CUDA 12.1, Python 3.11 |

---

### Training Methodology
| Parameter       | Value            |
| --------------- | ---------------- |
| Modality          | WLI              |
| Epochs          | 300              |
| Batch size      | 16               |
| Optimizer       | AdamW            |
| Image size      | 640×640          |
| Scheduler       | Cosine annealing |
| Mixed precision | AMP enabled      |

<p align="center">
    <img src="images/results.png" alt="Descriptive Alt Text" class="fit-width-image">
    <figcaption style="text-align: center;">Train metrix</figcaption>
</p>

### Inference: WLI(Training modality)
<p align="center" style="display: flex; flex-direction: row; gap: 2%">
  <img src="images/T_R_curve.png" alt="Image 1" style="width: 49%; height: auto;">
  <img src="images/T_PR_curve.png" alt="Image 2" style="width: 49%; height: auto;">
</p>
<p align="center" style="display: flex; flex-direction: row; gap: 2%">
  <img src="images/T_P_curve.png" alt="Image 2" style="width: 49%; height: auto;">
  <img src="images/T_F1_curve.png" alt="Image 2" style="width: 49%; height: auto;">
  <figcaption style="text-align: center;">Metrics of prediction on WLI</figcaption>
</p>
<p align="center" style="display: flex; flex-direction: row ;gap: 4%">
  <img src="images/val_batch0_labelsv1.jpg" alt="Image 1" style="width: 48%; height: auto;">
  <img src="images/val_batch0_predv1.jpg" alt="Image 2" style="width: 48%; height: auto;">
  <figcaption style="text-align: center;">GroundTruth/Prediction on WLI</figcaption>
</p>

```markdown
=== Test Set Metrics on trained modality(WLI)===
mAP50: 0.9138
mAP50-95: 0.7264
Precision: 0.8923
Recall: 0.8278
```

### Inference: NBI-LCI-FICE-BLI
<p align="center" style="display: flex; flex-direction: row; gap: 2%">
  <img src="images/I1_R_curve.png" alt="Image 1" style="width: 49%; height: auto;">
  <img src="images/I1_PR_curve.png" alt="Image 2" style="width: 49%; height: auto;">
</p>
<p align="center" style="display: flex; flex-direction: row; gap: 2%">
  <img src="images/I1_P_curve.png" alt="Image 2" style="width: 49%; height: auto;">
  <img src="images/I1_F1_curve.png" alt="Image 2" style="width: 49%; height: auto;">
  <figcaption style="text-align: center;">Metrics of prediction on NBI-LCI-FICE-BLI</figcaption>
</p>
<p align="center" style="display: flex; flex-direction: row ;gap: 4%">
  <img src="images/val_batch0_labelsv2.jpg" alt="Image 1" style="width: 48%; height: auto;">
  <img src="images/val_batch0_predv2.jpg" alt="Image 2" style="width: 48%; height: auto;">
  <figcaption style="text-align: center;">GroundTruth/Prediction on NBI-LCI-FICE-BLI</figcaption>
</p>

```markdown
=== NBI-LCI-FICE-BLI Metrics on unseen modalities===
mAP50: 0.6947
mAP50-95: 0.4972
Precision: 0.8402
Recall: 0.5672
```
### Inference: REAL-Colon
<p align="center" style="display: flex; flex-direction: row; gap: 2%">
  <img src="images/I2_R_curve.png" alt="Image 1" style="width: 49%; height: auto;">
  <img src="images/I2_PR_curve.png" alt="Image 2" style="width: 49%; height: auto;">
</p>
<p align="center" style="display: flex; flex-direction: row; gap: 2%">
  <img src="images/I2_P_curve.png" alt="Image 2" style="width: 49%; height: auto;">
  <img src="images/I2_F1_curve.png" alt="Image 2" style="width: 49%; height: auto;">
  <figcaption style="text-align: center;">Metrics of prediction on REAL-Colon</figcaption>
</p>
<p align="center" style="display: flex; flex-direction: row ;gap: 4%">
  <img src="images/val_batch2_labelsv3.jpg" alt="Image 1" style="width: 48%; height: auto;">
  <img src="images/val_batch2_predv3.jpg" alt="Image 2" style="width: 48%; height: auto;">
  <figcaption style="text-align: center;">GroundTruth/Prediction on REAL-Colon</figcaption>
</p>

## YOLOv11-FineTune
This project implements a robust polyp detection system using YOLO11 (You Only Look Once version 11) for medical image analysis. The model is trained on *WLI* modality.
### Training Configuration

#### Model Architecture
- **Base Model**: YOLO11 from Ultralytics
- **Input Resolution**: 640×640 pixels
- **Backbone**: CSPDarkNet
- **Neck**: PANet
- **Head**: Multi-scale detection

#### Hyperparameters
- **Epochs**: 50
- **Batch Size**: 16
- **Initial Learning Rate**: 0.001
- **Optimizer**: Auto-selected
- **Early Stopping Patience**: 10 epochs

#### Data Augmentation
- **Mosaic**: 0.8 probability
- **MixUp**: 0.1 probability
- **Copy-Paste**: 0.1 probability
- **Horizontal Flip**: 0.5 probability
- **Color Augmentation**: HSV adjustments
- **Spatial Transformations**: Rotation, translation, scaling, shearing

#### Detection Parameters
- **Training Confidence Threshold**: 0.1
- **IoU Threshold**: 0.4
- **Augmentation Focus**: Small object detection

### Inference: WLI 
```markdown
=== Test Set Metrics on trained modality(WLI)===
mAP50: 0.9282
mAP50-95: 0.7067
Precision: 0.8799
Recall: 0.8638
```
<p align="center">
    <img src="images/YOLOv11-FineTune-1.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv11-FineTune-2.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv11-FineTune-3.png" alt="Descriptive Alt Text" class="fit-width-image">
</p>

### Inference: NBI-LCI-FICE-BLI
```markdown
=== NBI-LCI-FICE-BLI Metrics on unseen modalities===
mAP50: 0.7619
mAP50-95: 0.5383
Precision: 0.8357
Recall: 0.6675
```
<p align="center">
    <img src="images/YOLOv11-FineTune-4.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv11-FineTune-5.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv11-FineTune-6.png" alt="Descriptive Alt Text" class="fit-width-image">
</p>

### Inference: REAL-Colon
<p align="center">
    <img src="images/YOLOv11-FineTune-7.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv11-FineTune-8.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv11-FineTune-9.png" alt="Descriptive Alt Text" class="fit-width-image">
</p>

```markdown
=== Metrics on unseen Dataset: REAL-Colon ===
mAP50: 0.3415
mAP50-95: 0.1922
Precision: 0.4986
Recall: 0.3413
```


## YOLOv11-FineTune-LoRA
Now, we turn to implementing bigger YOLO models using efficiency tricks. More specifically we mimic a LoRA(Low rank adaptation) style fine-tunning here: 
### Training Configuration

#### Model Architecture
- **Base Model**: YOLO11(L) from Ultralytics
- **Input Resolution**: 640×640 pixels
- **Backbone**: CSPDarkNet
- **Neck**: PANet
- **Head**: Multi-scale detection

#### Hyperparameters
- **Epochs**: 50
- **Batch Size**: 8
- **Freeze**: 10 ---> LoRA imitation
- **Initial Learning Rate**: 0.001
- **Optimizer**: Auto-selected
- **Early Stopping Patience**: 10 epochs

#### Data Augmentation
- **Mosaic**: 0.8 probability
- **MixUp**: 0.1 probability
- **Copy-Paste**: 0.1 probability
- **Horizontal Flip**: 0.5 probability
- **Color Augmentation**: HSV adjustments
- **Spatial Transformations**: Rotation, translation, scaling, shearing

#### Detection Parameters
- **Training Confidence Threshold**: 0.1
- **IoU Threshold**: 0.4
- **Augmentation Focus**: Small object detection

### Inference: WLI 
```markdown
=== Test Set Metrics ===
mAP50: 0.9351
mAP50-95: 0.7629
Precision: 0.9007
Recall: 0.8628
```
<p align="center">
    <img src="images/YOLOv11-FineTune-LoRA-1.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv11-FineTune-LoRA-2.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv11-FineTune-LoRA-3.png" alt="Descriptive Alt Text" class="fit-width-image">
</p>

### Inference: NBI-LCI-FICE-BLI
```markdown
=== Test Set Metrics ===
mAP50: 0.7985
mAP50-95: 0.6086
Precision: 0.8608
Recall: 0.6895
```
<p align="center">
    <img src="images/YOLOv11-FineTune-LoRA-4.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv11-FineTune-LoRA-5.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv11-FineTune-LoRA-6.png" alt="Descriptive Alt Text" class="fit-width-image">
</p>

### Inference: REAL-Colon
<p align="center">
    <img src="images/YOLOv11-FineTune-LoRA-7.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv11-FineTune-LoRA-8.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv11-FineTune-LoRA-9.png" alt="Descriptive Alt Text" class="fit-width-image">
</p>

```markdown
=== Metrics on unseen Dataset: REAL-Colon ===
mAP50: 0.3665
mAP50-95: 0.1925
Precision: 0.4906
Recall: 0.3745
```
<<<<<<< HEAD

## YOLOv12-FineTune
This time, We will try the most recent version of YOLO released by `Ultralytics` which is YOLOv12(s). 
### Training Configuration

#### Model Architecture
- **Base Model**: YOLO12 from Ultralytics
- **Input Resolution**: 640×640 pixels
- **Backbone**: Attention mechanisms and convolutional operations for feature extraction
- **Neck**: Residual Efficient Layer Aggregation Networks (R-ELAN)
- **Head**: Optimized attention architecture with Area Attention Mechanism, FlashAttention, and multi-scale detection

#### Hyperparameters
- **Epochs**: 50
- **Batch Size**: 16
- **Initial Learning Rate**: 0.001
- **Optimizer**: Auto-selected
- **Early Stopping Patience**: 10 epochs

#### Data Augmentation
- **Mosaic**: 0.8 probability
- **MixUp**: 0.1 probability
- **Copy-Paste**: 0.1 probability
- **Horizontal Flip**: 0.5 probability
- **Color Augmentation**: HSV adjustments
- **Spatial Transformations**: Rotation, translation, scaling, shearing

#### Detection Parameters
- **Training Confidence Threshold**: 0.1
- **IoU Threshold**: 0.4
- **Augmentation Focus**: Small object detection

### Inference: WLI 
```markdown
=== Test Set Metrics ===
mAP50: 0.9503
mAP50-95: 0.7134
Precision: 0.9418
Recall: 0.8743
```
<p align="center">
    <img src="images/YOLOv12-FineTune-1.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv12-FineTune-2.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv12-FineTune-3.png" alt="Descriptive Alt Text" class="fit-width-image">
</p>

### Inference: NBI-LCI-FICE-BLI
```markdown
=== Test Set Metrics ===
mAP50: 0.9616
mAP50-95: 0.7563
Precision: 0.9438
Recall: 0.8999
```
<p align="center">
    <img src="images/YOLOv12-FineTune-4.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv12-FineTune-5.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv12-FineTune-6.png" alt="Descriptive Alt Text" class="fit-width-image">
</p>

### Inference: REAL-Colon
<p align="center">
    <img src="images/YOLOv12-FineTune-7.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv12-FineTune-8.png" alt="Descriptive Alt Text" class="fit-width-image">
    <img src="images/YOLOv12-FineTune-9.png" alt="Descriptive Alt Text" class="fit-width-image">
</p>

```markdown
=== Metrics on unseen Dataset: REAL-Colon ===
mAP50: 0.3530
mAP50-95: 0.2131
Precision: 0.4705
Recall: 0.3644
```

=======
>>>>>>> c198478c873890a30547bc43d0c43818cef58b27
