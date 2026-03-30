# Machine Learning Lab Assignment 4: Email Spam Classification

This directory contains the code, analysis, and visual outputs for Assignment 4 of the Machine Learning course. The primary objective of this assignment is to build and evaluate machine learning models (Logistic Regression and Support Vector Machines) for classifying emails as either "Spam" or "Ham" (Not Spam) using Natural Language Processing (NLP) techniques.

## Files in this Directory

*   **`ml_lab_4.ipynb`**: The main Jupyter Notebook containing the entire implementation, from data loading and preprocessing to model training, hyperparameter tuning, and evaluation.
*   **`ml_04.pdf`**: The assignment problem statement or lab manual (PDF).
*   **Visualizations and Plots (*.png)**: The notebook generates several plots that are saved as PNG files to analyze the data and model performance.

## Project Workflow and Implementation Details

Based on the contents of the `ml_lab_4.ipynb` notebook, the workflow of the assignment is as follows:

### 1. Data Loading and Exploration
*   **Dataset**: The project uses an email dataset (`spam_ham_dataset.csv`) containing 5,171 emails. Each email is labeled as 'spam' (1) or 'ham' (0).
*   **Data Cleaning**: Unnecessary columns (`Unnamed: 0`) are dropped. The dataset consists of columns: `label`, `text`, and `label_num`.
*   **Class Distribution**: The dataset contains 3,672 'ham' emails and 1,499 'spam' emails. This distribution is plotted and saved as `class_distribution.png`.
*   **Text Analysis**: The length of the emails (character count) and the word counts are analyzed for both spam and ham emails, with histograms generated and saved as `text_length_analysis.png` and text boxplots saved as `text_boxplots.png`.

### 2. Text Preprocessing and Feature Extraction
*   **TF-IDF Vectorization**: The raw email text is converted into numerical features using `TfidfVectorizer` from `scikit-learn`.
*   **Parameters applied**: Stop words are removed (`stop_words='english'`), `max_features` is set to 5000, and an `ngram_range` of (1, 2) is used (unigrams and bigrams).
*   The resulting TF-IDF feature matrix has a shape of `(5171, 5000)`.

### 3. Model Training and Hyperparameter Tuning

#### Logistic Regression
*   A Logistic Regression model is trained and optimized using `GridSearchCV`.
*   **Hyperparameter Tuning**: Parameters tested include `C` values `[0.1, 1, 10, 100]` and `penalty` types `['l1', 'l2']` (with solver `liblinear`).
*   **Results**: The best parameters found are `{'C': 10, 'penalty': 'l2', 'solver': 'liblinear'}`, yielding an impressive CV Accuracy of ~98.19%.
*   Performance visualizations for the hyperparameter combinations are exported as `lr_hyperparam_performance.png`, `enhanced_lr_performance.png`, `lr_performance.png`, and `lr_white_performance.png`.

#### Support Vector Machine (SVM)
*   Various SVM kernels are compared: `linear`, `poly`, and `rbf`. The `rbf` (Radial Basis Function) kernel performed the best (Accuracy: ~98.74%, compared to linear at ~98.65% and poly at ~90.24%). This comparison is available in `svm_kernel_comparison.png`.
*   **Hyperparameter Tuning**: A grid search is performed on the `rbf` kernel SVM with parameters `C`: `[0.1, 1, 10, 100]` and `gamma`: `['scale', 'auto']`.
*   **Results**: The best parameters found are `{'C': 10, 'gamma': 'scale'}`, achieving a peak CV accuracy of ~98.50%.
*   Visualizations showing the SVM training vs CV accuracy curve based on C are saved in `svm_hyperparam_performance.png` and `svm_simple_plot.png`.

### 4. Cross-Validation and Model Evaluation
*   **5-Fold Cross Validation**: Both optimized models undergo 5-fold Cross-Validation to ensure generalization. The CV results and analysis are saved in `cv_results.png`.
*   **Metrics Calculated**: The models are assessed using Accuracy, Precision, Recall, and F1-Score. Both models score highly, generally in the 97%-98% threshold range across all metrics.
*   **Confusion Matrices**: Confusion matrices mapping True Positives, True Negatives, False Positives, and False Negatives for the models are exported to `confusion_matrices.png`.
*   **ROC Curves & AUC**: Receiver Operating Characteristic (ROC) plots comparing the trade-off between the True Positive Rate and False Positive Rate are plotted and saved as `roc_curves.png`. Individual model ROC curves are available at `roc_lr.png` and `roc_svm.png`.

## Image Content Summary
Here is a table explaining what each generated image represents:

| File Name | Description |
| :--- | :--- |
| `class_distribution.png` | A bar chart showing the absolute counts of 'Ham' vs 'Spam' instances in the dataset. |
| `text_length_analysis.png` | Histograms depicting the distribution of character counts and word counts grouped by class (Spam vs. Ham). |
| `text_boxplots.png` | Boxplots displaying the variance and outliers in text lengths for the two classes. |
| `lr_hyperparam_performance.png` | A plot demonstrating how changing Logistic Regression hyperparameters (like `C` and `penalty`) affects Cross-Validation Accuracy. |
| `lr_performance.png` / `enhanced...` / `lr_white...` | Visual summaries of the Logistic Regression model's overall performance metrics including train/test curves. |
| `svm_kernel_comparison.png` | Bar charts comparing Accuracy and Training Time across different SVM kernels (`linear`, `poly`, `rbf`). |
| `svm_hyperparam_performance.png` | A line plot visualizing the accuracy curve representing the Trade-off parameter `C` against CV and train accuracy for the SVM model. |
| `svm_simple_plot.png` | Simplified performance metric charts for the SVM model. |
| `cv_results.png` | Graphical representation of the 5-Fold Cross-Validation accuracy scores for both Logistic Regression and SVM across each fold. |
| `confusion_matrices.png` | Visual heatmaps showing the confusion matrices (Actual vs Predicted classifications) for both models. |
| `roc_curves.png` / `roc_lr.png` / `roc_svm.png` | Plots showing the ROC curve and the AUC (Area Under the Curve) scores indicating the classification capability of the models. |
