# Smart Home & Weather Analysis Pipeline (Deep Learning)

## 🎓 Submission Details

- **Project Title**: Smart Home & Weather Analysis Pipeline
- **Brief Description**: An end-to-end Deep Learning pipeline that integrates and analyzes two logically related datasets: Smart Home Energy Consumption and Weather Type Classification. The pipeline applies standard preprocessing, builds a 1D CNN model, and evaluates tasks using complete metrics and carbon tracking.

### Team Members
| Student Name | Student ID |
| :--- | :--- |
| Kirolos Mourice | 192100132 |
| Mohamed Saad | 192100137 |
| Sherif Diaa | 192100037 |
| Mostafa Abdallah | 192100058 |

---

## 📌 Project Overview
This project presents an end-to-end Deep Learning pipeline that integrates and analyzes two distinct but logically related datasets: **Smart Home Energy Consumption** and **Weather Type Classification**. The code handles advanced data preprocessing, builds state-of-the-art Deep Learning models (using TensorFlow/Keras), and rigorously evaluates performance across both classification and regression tasks. 

Notably, this system computes exact computational complexity (FLOPs) and actively monitors its own environmental footprint (Energy Consumption and Carbon Emissions) during model training and inference.

---

## 💾 Datasets

We combine two prominent datasets to establish a powerful predictive pipeline. **Note:** if you are running this on Kaggle, you must add these exact datasets to your input:

1. **[Weather Type Classification Dataset](https://www.kaggle.com/datasets/nikhil7280/weather-type-classification/versions/1)**:
   Contains meteorological features such as *Temperature, Humidity, Wind Speed, Precipitation, Atmospheric Pressure,* and *Visibility*. This dataset is utilized for predicting the weather conditions type.
2. **[Smart Home Dataset (HomeC)](https://www.kaggle.com/datasets/taranvee/smart-home-dataset-with-weather-information/versions/1)**:
   A detailed dataset tracking energy consumption across various home appliances, combined with localized weather information, timestamped to track hourly and monthly variations.

---

## ⚙️ Data Preprocessing & Engineering

Robust preprocessing techniques are critical to the accuracy of our DL models. The following steps are strictly applied to the data:

- **Missing Values Handling**: Initial parsing drops invalid formats and reconstructs continuous timestamps for time-series structural integrity.
- **Outlier Detection & Removal (Z-Score)**: Statistical identification bounds variables within a $Z < 3$ threshold to eliminate statistical anomalies and noise.
- **Feature Scaling**: Features undergo standard normal scaling via `StandardScaler` to ensure zero mean and unit variance, a requisite for rapidly converging Deep Neural Networks.
- **Data Augmentation**: We apply noise injection (Gaussian distribution $N(0, 0.02)$) to synthetically expand the minority/training distributions. This significantly mitigates model overfitting and boosts generalization.

---

## 🧠 Deep Learning Architecture & Training

The core methodology harnesses **1D Convolutional Neural Networks (CNN)** tailored for structured tabular and sequential features.

- **Architecture Details**:
  - `Conv1D` Layers for feature extraction.
  - `BatchNormalization` and `MaxPooling1D` for spatial hierarchy representation and scaling.
  - `Dropout` layers ($0.3$ and $0.4$) deployed as regularization to penalize overconfidence.
  - Fully Connected (`Dense`) output layers customized per task (e.g., `softmax` for categorical choices, `linear` for regression tasks).
- **Training Mechanics**:
  - Optimizers: Adaptive moment estimation (Adam).
  - Callbacks: Incorporates `EarlyStopping` (to halt when validation loss plateaus) and `ReduceLROnPlateau` (to dynamically decay learning rate).
  - Visualizing the **Learning Curves** (Loss and Accuracy curves across epochs) verifies model fitting.

---

## 📈 Testing & Evaluation Metrics

The system performs extreme validation for its output. Evaluation is broken down depending on the predictive problem nature:

### Classification Metrics (Weather Type Prediction)
The predictions are quantified using the following rigorous indicators:
- **Accuracy (ACC)**
- **Receiver Operating Characteristic - Area Under Curve (ROC-AUC)** 
- **F1-Score (Macro)**
- **Cohen’s Kappa Coefficient**
- **Matthews Correlation Coefficient (MCC)**
- **Recall & Precision**

### Regression Metrics (Energy Consumption/Target Prediction)
The numerical models utilize minimizing errors:
- **Mean Absolute Error (MAE)**
- **Mean Absolute Percentage Error (MAPE)**
- **Root Mean Squared Error (RMSE)**
- **Coefficient of Determination ($R^2$)**

---

## 🌍 Hardware & Environmental Impact

As Artificial Intelligence scales, computation sustainability becomes vital. This project embraces *Green AI* techniques:

- **CodeCarbon Integration (`EmissionsTracker`)**: The pipeline actively tracks kWh energy consumption and exact equivalents of $CO_2$ emissions generated by utilizing GPU processors.
- **Computational Complexity (FLOPS)**: The code manually computes the *Floating Point Operations per Second* (FLOPs) specifically for the 1D Deep Learning Architecture, enabling a firm grasp on the theoretical computational limits and real-time efficiency of the inference step.

---

## 🚀 How to Run

### Running on Kaggle
1. Create a new notebook on Kaggle.
2. Under the **"Input"** section on the right panel, click **Add Data** and add the two datasets listed in the [Datasets](#-datasets) section above via their URLs.
3. **Copy and paste** the entire source code from `main.py` into a single code cell.
4. Run the cell to execute the entire pipeline, view the evaluation outputs, and generate the saved model files.

### 📝 Outputs
   - The CLI/Cell terminal will log the GPU allocations and Carbon tracking.
   - Classification/Regression metrics will format directly onto the terminal.
   - Model weights are persistently saved (e.g., `weather_cnn_model.keras`).
