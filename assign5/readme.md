# Machine Learning Lab Assignment 5: Optical Character Recognition (OCR)

This directory contains the code, analysis, and visual outputs for Assignment 5 of the Machine Learning course. The objective of this experiment is to implement and evaluate Artificial Neural Network models, specifically the Perceptron Learning Algorithm (PLA) and various configurations of the Multilayer Perceptron (MLP), for an Optical Character Recognition (OCR) task.

## Files in this Directory

*   **`assign_5.ipynb`**: The main Jupyter Notebook containing the complete data pipeline, from image loading and preprocessing to model training, evaluation, and visualization.
*   **`Experiment 5_Dataset/`**: Directory containing the dataset, which includes images of English characters and a CSV file (`english.csv`) mapping images to their labels.
*   **`Experiment_5.pdf` & `ML_assign_05.pdf`**: The assignment problem statement and lab manual documentation.
*   **Visualizations and Plots (*.png)**: The notebook generates several plots that are saved as PNG files to analyze the training progress and classification performance of the developed models.

## Project Workflow and Implementation Details

Based on the contents of the `assign_5.ipynb` notebook and the generated files, the assignment follows this workflow:

### 1. Data Loading and Exploration
*   **Dataset**: The project uses an English character dataset consisting of 3,410 images. The dataset characterizes 62 distinct classes corresponding to English alphanumeric characters (0-9, A-Z, a-z). Each class has 55 samples.
*   **Data File**: The `english.csv` file provides the mappings between the image file paths (`Img/imgXXX-YYY.png`) and their corresponding target labels.
*   **Image Processing**: Images are loaded using the OpenCV library (`cv2`) and preprocessed (likely resized, converted to grayscale, and flattened into 1D arrays) to be fed into the neural network models.

### 2. Model Implementations
The core of the assignment focuses on training artificial neural networks to classify the character images.

#### Perceptron Learning Algorithm (PLA)
*   A basic single-layer perceptron model is trained on the dataset.
*   **Output**: The training loss over iterations is tracked and saved as `loss_pla.png`. The ROC curve evaluating the PLA's classification is saved as `ROC_CURVE_PLA.png`.

#### Multilayer Perceptron (MLP) Classifier
*   The `MLPClassifier` from `scikit-learn` is utilized to train deeper feedforward neural networks.
*   To understand the effect of network depth and capacity, three different hidden layer architectures are evaluated:
    1.  **Single Hidden Layer**: 128 neurons (`MLP(128)`).
    2.  **Two Hidden Layers**: 256 neurons in the first layer and 128 in the second (`MLP(256-128)`).
    3.  **Three Hidden Layers**: 512, 256, and 128 neurons respectively (`MLP(512-256-128)`).

### 3. Model Evaluation
*   **Metrics**: The models are evaluated using standard classification metrics including `accuracy_score`, `precision_score`, `recall_score`, and `f1_score`, compiled into a detailed `classification_report`.
*   **ROC Curves & AUC**: Because this is a multi-class problem (62 classes), labels are binarized using `label_binarize` to compute the Receiver Operating Characteristic (ROC) curves and Area Under the Curve (AUC) for model evaluation.
*   **Grid Search**: `GridSearchCV` is imported and likely used for hyperparameter tuning of the MLP configurations (e.g., tuning learning rates, activation functions, or solvers).

## Image Content Summary

The scripts output various performance plots, summarized below:

| File Name | Description |
| :--- | :--- |
| `loss_pla.png` | A line plot showing the training loss curve for the standard Perceptron Learning Algorithm (PLA) over the training epochs. |
| `ROC_CURVE_PLA.png` | The Receiver Operating Characteristic (ROC) curve demonstrating the performance boundaries of the PLA model. |
| `Training Loss vs Epochs - MLP(128).png` | The training loss curve showing the convergence of the Multilayer Perceptron with a single hidden layer of 128 neurons. |
| `ROC CURVE - MLP(128).png` | The ROC curve and AUC score for the MLP configuration with 1 hidden layer (128 neurons). |
| `Training Loss vs Epochs - MLP(256-128).png` | The training loss curve demonstrating the optimization progress of the 2-hidden-layer MLP (256, 128). |
| `ROC_CURVE - MLP(256-128).png` | The ROC curve and AUC score for the MLP configuration with 2 hidden layers (256, 128 neurons). |
| `Training Loss vs Epochs - MLP(512-256-128).png` | The training loss curve representing the training trajectory of the deepest MLP network (3 hidden layers: 512, 256, 128). |
| `ROC_CURVE - MLP(512-256-128).png` | The ROC curve and AUC score for the deepest MLP configuration with 3 hidden layers (512, 256, 128 neurons). |
