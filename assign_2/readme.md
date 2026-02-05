Binary Email Classification using Naive Bayes and K-Nearest Neighbors

This project implements and evaluates multiple machine learning models for a binary classification problem using the Spambase dataset.

Project Workflow Overview

Data loading and exploration
Data preprocessing and feature scaling
Model implementation
Model evaluation using metrics and visualizations
Hyperparameter tuning and optimization
Bias–variance and overfitting analysis

Data Loading and Exploration

The dataset is loaded for analysis.
Input features consist of numerical values extracted from email content.
The target variable is binary.

0 represents Non-Spam
1 represents Spam

Initial exploration includes dataset shape inspection and class distribution analysis to check for imbalance.

Data Preprocessing

The following preprocessing steps are implemented.

Train–Test Split

The dataset is split into training and testing sets.
Stratification ensures the class distribution is preserved in both sets.

Feature Scaling

StandardScaler is applied to normalize features.

Model Implementation

Naive Bayes Classifiers

Three variants of Naive Bayes are implemented.

Gaussian Naive Bayes
This model assumes features follow a normal distribution and works well for continuous features.

Multinomial Naive Bayes
This model is designed for frequency-based features and is commonly used in text classification tasks.

Bernoulli Naive Bayes
This model assumes binary feature values and captures the presence or absence of features.

Each model is trained on the training dataset, evaluated on the test dataset, and compared using consistent evaluation metrics.

K-Nearest Neighbors (KNN)

A baseline KNN classifier is implemented using scikit-learn.
The model predicts labels based on majority voting among nearest neighbors.
Distance-based weighting is also evaluated.

Different values of k are tested, and performance trends are observed using accuracy plots.

Hyperparameter Tuning

To optimize KNN performance, two approaches are used.

Grid Search
An exhaustive search is performed over the number of neighbors, distance metric, and neighbor search algorithm.

Randomized Search
Random sampling from the hyperparameter space is used as a faster alternative to grid search.

The best hyperparameters are selected based on cross-validation accuracy.

Neighbor Search Algorithms

Two spatial indexing methods are compared: KDTree and BallTree.

Both are evaluated in terms of accuracy, training time, prediction time, and memory usage.

Model Evaluation

The models are evaluated using accuracy, precision, recall, F1-score, specificity, false positive rate, and ROC–AUC.

Visual evaluations include confusion matrices for each classifier, ROC curves to analyze class separation, accuracy versus k plots for KNN, and training versus validation accuracy plots to detect overfitting.

Overfitting and Underfitting Analysis

Small values of k in KNN lead to overfitting due to high model variance.
Large values of k cause underfitting due to excessive smoothing.
Hyperparameter tuning helps achieve better generalization.

Bias–Variance Analysis

Naive Bayes exhibits high bias due to its strong independence assumption.
KNN has high variance for small k values.
Proper tuning balances the bias–variance trade-off effectively.

Key Observations

Naive Bayes models are computationally efficient but less expressive.
KNN achieves higher accuracy after tuning.
Visualization plays a key role in diagnosing model behavior.

Technologies Used

Python
NumPy
Pandas
Scikit-learn
Matplotlib
Seaborn
