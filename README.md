# 🏠 Multimodal House Price Prediction Using Satellite Imagery


> **A machine learning system that predicts house prices by combining traditional real estate features with satellite imagery analysis using deep learning.**

**Author:** Anshit Das  
**Date:** January 2026

---

## 📊 Key Results

| Model            | R² Score   | RMSE        | MAE         | Improvement   |
|------------------|----------  |------       |-----        |-------------  |
| **Tabular Only** | 0.9119     | $103,753    | $62,376     | Baseline      |
| **Multimodal**   | **0.9131** | **$103,071**| **$62,218** | **+0.13%**    |

**Key Achievement:** Successfully integrated satellite imagery with traditional features, achieving measurable improvement in prediction accuracy.

---

## 🎯 Project Overview

### Problem Statement
Traditional automated valuation models (AVMs) rely solely on structured data, missing crucial visual information that human appraisers consider—landscaping, neighborhood quality, building condition, and urban density.

### Solution
This project develops a **multimodal machine learning system** that combines:
- 📋 **Tabular Features** (20+ engineered features from property data)
- 🛰️ **Satellite Imagery** (ResNet50 deep learning embeddings)
- 🌳 **Geo-Visual Features** (vegetation, urban density, texture)
- 🚇 **Transport Data** (OpenStreetMap distance metrics)

### Architecture
```
Tabular Data + Satellite Images → Feature Engineering → XGBoost → Price Prediction
     ↓              ↓                      ↓
  (20+ feat)   (ResNet50)         (PCA: 2048D→26D)
                    ↓
              (Geo-Visual: 7D)
                    ↓
              (Transport: 6D)
                    ↓
             [60+ Total Features]
```

---

## 📂 Project Structure

```
CDC_Project_House_Prediction/
│
├── 📓 pre-processing.ipynb          # Data preprocessing & feature engineering
├── 📓 model_training (2).ipynb      # Model training & evaluation
├── 🐍 data_fetcher.py               # Download satellite images from Mapbox
├── 🐍 utils.py                      # Utility functions & helpers
├── 🐍 visualizations.py             # Grad-CAM & data visualization
├── 📄 requirements.txt              # Python dependencies
│
├── 📁 data/                         # (Not included - download separately)
│   ├── train.csv                    # house sales data
│   └── satellite_images/            # Downloaded satellite images
│
├── 📁 models/                       # (Generated after training)
│   └── multimodal_predictor.pkl     # Trained model (~425 MB)
│
└── 📁 outputs/                      # (Generated during execution)
    ├── gradcam/                     # Grad-CAM visualizations
    └── *.png                        # Analysis plots
```

---

## 🚀 Quick Start

### Prerequisites

- **Python 3.8+**
- **NVIDIA GPU** (recommended for CNN feature extraction)
- **8GB+ RAM**
- **Mapbox API Token** ([Get free token](https://account.mapbox.com/))

### Installation

1. **Clone the repository:**
```bash
git clone https://github.com/anshit0205/CDC_Project_House_Prediction.git
cd CDC_Project_House_Prediction
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Download dataset:**
   - Place `train.csv` in `data/` directory

4. **Configure Mapbox token:**
   - Edit `data_fetcher.py`
   - Replace `YOUR_MAPBOX_TOKEN_HERE` with your actual token

---

## 📖 Usage Guide

### Step 1: Download Satellite Images

```bash
python data_fetcher.py
```

**What it does:**
- Downloads 512×512px satellite images for each property
- Uses Mapbox Satellite API (zoom level 18)
- Saves images as `{property_id}_z18_s2.png`
- Implements rate limiting (0.2s delay)
- Resumes from interruption

**Expected output:**
```
📊 Loaded 16,209 properties from data/train.csv
Downloading images: 100%|████████████| 16209/16209 [2:15:30<00:00]
✅ Download complete!
   Downloaded: 16209
   Skipped: 0
   Failed: 0
```

### Step 2: Data Preprocessing & Feature Engineering

**Open and run:** `pre-processing.ipynb`

**What it does:**

#### 2.1 Tabular Feature Engineering
- Ratio features (bath/bed, sqft/bed, lot/living)
- Temporal features (house_age, years_since_reno)
- Quality composites (quality_area, condition_area)
- Neighborhood comparisons (relative_living_size)

#### 2.2 Image Feature Extraction (ResNet50)
```python
# Extract 2048-dimensional CNN embeddings
- Load ResNet50 (ImageNet pretrained)
- Extract from final pooling layer
- Process all 16,110 images
- Save embeddings to DataFrame
```
#### 2.3 Geo-Visual Features
Extract interpretable visual features:
- Green fraction (vegetation coverage)
- Impervious surfaces (roads, buildings)
- Edge density (urban complexity)
- Brightness statistics
- Texture features (GLCM)

#### 2.4 Transport Distance Features
Query OpenStreetMap for distances to:
- Metro/subway stations
- Railway stations  
- Airports

**Search radius:** 40 km (covers Seattle metro area)

#### 2.5 Output
Creates `processed_data.csv` with **2000+ features**:
- Original tabular: 19 columns
- Engineered tabular: 20+ features
- ResNet50 embeddings: 2048 dimensions
- Geo-visual: 7 features
- Transport: 6 features

**Expected output:**
```
✅ Preprocessing complete!
   Total samples: 16,110
   Total features: 2,000+
   File saved: data/processed_data.csv
```

### Step 3: Model Training & Evaluation

**Open and run:** `model_training (2).ipynb`

**What it does:**

#### 3.1 Automatic PCA Selection
```python
# Reduce 2048D → 26D while retaining 77.6% variance
- Fit PCA on image embeddings
- Find elbow point in variance curve
- Select optimal components: 26
```

**PCA Results:**
```
Elbow detected at 26 components
Variance explained: 77.6%
Dimension reduction: 98.7% (2048 → 26)
```

#### 3.2 Train Two Models
**Baseline (Tabular Only):**
- Features: Tabular + Transport + Zipcode (26 features)
- Purpose: Establish baseline performance

**Multimodal (Full):**
- Features: All above + Image PCA + Geo-visual (60+ features)
- Purpose: Evaluate image contribution

#### 3.3 XGBoost Configuration
```python
xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.1,
    'n_estimators': 1000,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'device': 'cuda'  # GPU acceleration
}
```

#### 3.4 Evaluation Metrics
- R² Score (coefficient of determination)
- RMSE (root mean squared error)
- MAE (mean absolute error)
- Feature importance analysis

**Expected output:**
```
================================================================================
FINAL RESULTS
================================================================================

TABULAR ONLY MODEL:
RMSE: $103,753.43
MAE: $62,376.39
R²: 0.9119

MULTIMODAL MODEL (Tabular + Images):
PCA components: 26
RMSE: $103,070.82
MAE: $62,218.05
R²: 0.9131

📊 IMPROVEMENT:
RMSE: +0.66%
R²: +0.13%

✅ Model saved to: models/multimodal_predictor.pkl
```

### Step 4: Visualization & Analysis

**Run:** `visualizations.py`

```bash
python visualizations.py
```

**What it generates:**

#### 4.1 Grad-CAM Visualizations
- Shows which image regions CNN focuses on
- Heatmaps overlay on original satellite images
- Validates model learns relevant features (buildings, vegetation, roads)

#### 4.2 Data Analysis Plots
- Price distribution (original & log-transformed)
- Feature correlation heatmap
- Spatial distribution map
- Feature importance chart

**Output location:** `outputs/`

---

## 📊 Results Analysis

### Model Performance Comparison

| Metric | Tabular Only | Multimodal | Improvement |
|--------|--------------|------------|-------------|
| **R² Score** | 0.9119 | **0.9131** | +0.13% ↑ |
| **RMSE** | $103,753 | **$103,071** | -0.66% ↓ |
| **MAE** | $62,376 | **$62,218** | -0.25% ↓ |
| **Features** | 26 | 60+ | +130% |

**Interpretation:**
- **91.31%** of price variance explained by multimodal model
- **Consistent improvement** across all metrics
- Image features provide **measurable value** despite location dominance

### Top 10 Feature Importances

| Rank | Feature | Type | Importance | 
|------|---------|------|------------|
| 1 | zipcode_te | Location | 25.4% |
| 2 | quality_area | Engineered | 18.8% |
| 3 | grade | Tabular | 11.2% |
| 4 | sqft_living | Tabular | 6.1% |
| 5 | lat | Geospatial | 4.9% |
| 6 | condition_area | Engineered | 3.3% |
| 7 | view_score | Engineered | 2.2% |
| 8 | view | Tabular | 2.0% |
| 9 | long | Geospatial | 1.9% |
| 10 | waterfront | Tabular | 1.8% |
| ... | img_pca_3 | **Image** | **1.4%** ← |

**Key Insights:**
- **Location dominates** (zipcode: 25.4%)
- **Quality × size** is critical (18.8%)
- **Image features present** in top 20 (multiple PCA components)
- **Engineered features** outperform raw features

### Feature Category Breakdown

| Category | Total Importance | Features |
|----------|------------------|----------|
| Location (zipcode, lat, long) | 32.2% | 3 |
| Engineered Composites | 24.3% | ~8 |
| Raw Tabular | 18.9% | ~12 |
| **Image Features (PCA)** | **14.1%** | **26** |
| Geo-Visual | 6.8% | 7 |
| Transport | 3.7% | 6 |

### Grad-CAM Findings

**What the CNN looks at:**

1. **Buildings & Structures** - Main architectural features (red hot spots)
2. **Vegetation & Green Space** - Landscaping quality (yellow regions)
3. **Roads & Infrastructure** - Accessibility patterns
4. **Urban vs Suburban** - Context-aware density detection

**Validation:** Model focuses on economically relevant features, not noise.

---

## 🛠️ Technical Details

### Key Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Deep Learning** | PyTorch + ResNet50 | Image embedding extraction |
| **Gradient Boosting** | XGBoost | Regression modeling |
| **Dimensionality Reduction** | scikit-learn PCA | 2048D → 26D compression |
| **Geospatial** | OSMnx + BallTree | Transport distance calculation |
| **Image Processing** | PIL, OpenCV, scikit-image | Geo-visual feature extraction |
| **Visualization** | Matplotlib, Seaborn | Analysis & Grad-CAM |

### Model Architecture

```python
# Image Feature Pipeline
Satellite Image (512×512)
    ↓
ResNet50 (Pretrained on ImageNet)
    ↓
2048D Embedding Vector
    ↓
PCA (Automatic Selection)
    ↓
26D Reduced Embedding
    ↓
[Combined with Tabular Features]
    ↓
XGBoost Regressor
    ↓
Predicted Log Price
    ↓
Expm1 Transform → Final Price
```

---

## 📦 Dependencies

### Core Libraries
```
python>=3.8
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=2.0.0
```

### Deep Learning
```
torch>=2.0.0
torchvision>=0.15.0
```

### Image Processing
```
Pillow>=9.0.0
opencv-python>=4.5.0
scikit-image>=0.19.0
```

### Geospatial
```
osmnx>=1.2.0
geopandas>=0.12.0
```

### Visualization
```
matplotlib>=3.5.0
seaborn>=0.12.0
```

**Install all:**
```bash
pip install -r requirements.txt
```

---

## 🎓 Methodology Highlights

### 1. **Automatic PCA Selection**
- **Challenge:** 2048D embeddings too high-dimensional
- **Solution:** Elbow method to find optimal components
- **Result:** 26 components retain 77.6% variance
- **Benefit:** 98.7% dimension reduction, faster training

### 2. **Transport Radius (40 km)**
- **Context:** Seattle area = 217 km²
- **Calculation:** `radius = sqrt(217/π) × 2 × 1.5 + buffer = 40 km`
- **Rationale:** Complete coverage of metro area + suburbs
- **Validation:** All properties within range of infrastructure

### 3. **Log Price Transform**
- **Issue:** Right-skewed price distribution ($75K - $7.7M)
- **Solution:** `y = log(price + 1)`
- **Benefit:** Stabilizes variance, improves convergence

---

### Future Enhancements

- [ ] Try EfficientNet, Vision Transformers
- [ ] Multi-scale imagery (different zoom levels)
- [ ] Street view integration
- [ ] Transfer learning across cities
- [ ] Uncertainty quantification

```
## 📚 References

1. He, K., et al. (2016). "Deep Residual Learning for Image Recognition." *CVPR*.
2. Chen, T., & Guestrin, C. (2016). "XGBoost: A Scalable Tree Boosting System." *KDD*.
3. Selvaraju, R. R., et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks." *ICCV*.

---
