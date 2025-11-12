# Detecting Orange Diseases Using K-Nearest Neighbors for Sustainable Citrus Farming

A machine learning project that uses K-Nearest Neighbors (KNN) algorithm to classify citrus diseases from leaf images, supporting sustainable farming practices through early disease detection.

## 🎯 Project Overview

This project implements a KNN-based classification system to detect three types of citrus conditions:
- **Citrus Canker** - A bacterial disease causing lesions on leaves and fruit
- **Healthy** - Normal healthy citrus leaves
- **Melanose** - A fungal disease causing brown spots on fruit and leaves

The system achieves **82.82% accuracy** on test data, providing farmers with a reliable tool for early disease detection.

## � Dataset

**Source**: [Orange Fruit Dataset on Kaggle](https://www.kaggle.com/datasets/mohammedarfathr/orange-fruit-daatset)

The orange fruit dataset was gathered using a phone camera in the HEIC (highly enhanced image container) format and was converted to JPEG format. This dataset was collected in research carried out to determine the dominant diseases affecting oranges in the eastern parts of Uganda. It can be used for artificial intelligence projects and agricultural research.

**Key Information**:
- **Format**: JPEG images (converted from HEIC)
- **Origin**: Eastern Uganda
- **Purpose**: Research on dominant orange diseases
- **License**: CC BY-SA 4.0
- **Applications**: Computer Vision, Deep Learning, Image Classification, Neural Networks

## �📊 Model Performance

```
Model: KNN
Overall Accuracy: 82.82%

Class-wise Performance:
┌───────────────┬───────────┬────────┬──────────┬─────────┐
│ Class         │ Precision │ Recall │ F1-Score │ Support │
├───────────────┼───────────┼────────┼──────────┼─────────┤
│ Citrus Canker │   96.86%  │ 55.38% │  70.47%  │   390   │
│ Healthy       │   77.27%  │ 95.90% │  85.58%  │   390   │
│ Melanose      │   81.86%  │ 97.18% │  88.86%  │   390   │
└───────────────┴───────────┴────────┴──────────┴─────────┘
```

## 🚀 Features

- **Image Classification**: Automated disease detection from citrus leaf images
- **Data Balancing**: Tools to balance dataset distribution across classes
- **Performance Testing**: Comprehensive performance evaluation including:
  - Feature extraction time
  - Classification time
  - CPU and RAM usage monitoring
- **Jupyter Notebook**: Interactive exploration and visualization
- **Pre-trained Model**: Ready-to-use KNN model (`KNN.joblib`)

## 📁 Project Structure

```
orange_8/
├── README.md                           # Project documentation
├── data_balance.py                     # Dataset balancing utility
├── test_knn_performance.py             # Performance testing script
├── model/
│   ├── KNN.joblib                      # Trained KNN model
│   └── classification_report_KNN.txt   # Detailed performance report
├── graph_and_result/
│   └── classification_report_KNN.txt   # Classification results
└── notebook/
    └── knn-classification.ipynb        # Jupyter notebook for exploration
```

## 🛠️ Requirements

```bash
# Core dependencies
opencv-python
numpy
scikit-learn
joblib
psutil
tqdm
jupyter  # For notebook usage
```

## 💻 Installation

1. Clone the repository:
```bash
git clone https://github.com/hjolly16/OrangeKNN.git
cd orange_8
```

2. Install dependencies:
```bash
pip install opencv-python numpy scikit-learn joblib psutil tqdm jupyter
```

## 📖 Usage

### 1. Testing Model Performance

Run performance tests on a folder of test images:

```bash
python test_knn_performance.py
```

This script will:
- Load the pre-trained KNN model
- Process test images from a specified folder
- Measure processing time, CPU, and RAM usage
- Generate detailed performance metrics

### 2. Balancing Dataset

Balance image distribution across different disease classes:

```bash
python data_balance.py
```

Configure the script to:
- Specify the base path to your dataset
- Set target count for each class
- Automatically balance by removing excess images

### 3. Interactive Exploration

Launch the Jupyter notebook for interactive analysis:

```bash
jupyter notebook notebook/knn-classification.ipynb
```

## 🔬 Methodology

### Feature Extraction
The system uses computer vision techniques to extract relevant features from citrus leaf images, capturing patterns that distinguish between healthy and diseased leaves.

### Classification
K-Nearest Neighbors (KNN) algorithm is employed to classify images based on extracted features. The model learns from labeled training data to predict disease presence in new images.

### Performance Monitoring
The testing script provides real-time monitoring of:
- **Processing Time**: Time taken for feature extraction and classification
- **Resource Usage**: CPU and memory consumption (baseline-adjusted)
- **Batch Statistics**: Aggregate performance metrics across test sets

## 📈 Model Insights

**Strengths:**
- High precision for Citrus Canker detection (96.86%)
- Excellent recall for Melanose detection (97.18%)
- Strong overall F1-score for Melanose (88.86%)

**Areas for Improvement:**
- Citrus Canker recall (55.38%) could be improved
- Consider ensemble methods or deep learning for better performance

## 🌱 Impact on Sustainable Farming

Early disease detection enables:
- **Reduced Pesticide Use**: Target treatments only where needed
- **Higher Crop Yields**: Early intervention prevents disease spread
- **Cost Savings**: Minimize crop losses and treatment costs
- **Environmental Protection**: Precision agriculture reduces chemical runoff

## 🔮 Future Enhancements

- [ ] Implement deep learning models (CNN) for improved accuracy
- [ ] Add real-time detection via mobile application
- [ ] Expand disease categories
- [ ] Include severity level classification
- [ ] Integrate with IoT sensors for automated monitoring
---