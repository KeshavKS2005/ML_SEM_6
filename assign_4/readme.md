Binary Classification using Logistic Regression and Support Vector Machine



Objective:



The objective of this experiment is to classify emails as spam or ham using two binary classification techniques:



Logistic Regression



Support Vector Machine (SVM)



The experiment also focuses on understanding the effect of hyperparameter tuning on the classification performance of both models.



Dataset Description:



The experiment uses the Spambase dataset, which consists of numerical features extracted from email content. Each email is labeled as either spam or non-spam (ham).



The dataset is commonly used for email spam classification tasks and is suitable for evaluating binary classifiers.



Theory Overview:



Logistic Regression:

Logistic Regression is a probabilistic binary classification algorithm. It uses the sigmoid function to estimate the probability that a given input belongs to the positive class (spam). A threshold value, typically 0.5, is used to convert probabilities into class labels.



Regularization Techniques:



L1 Regularization (Lasso): Encourages sparsity by shrinking some coefficients to zero and helps in feature selection.



L2 Regularization (Ridge): Penalizes large weights and improves model generalization.



Important Logistic Regression Hyperparameters:





Solver: liblinear and saga



Support Vector Machine (SVM):

SVM is a margin-based classifier that finds the optimal hyperplane separating two classes by maximizing the margin. Only a subset of data points, called support vectors, influence the decision boundary.



SVM Kernels:



Linear Kernel



Polynomial Kernel



Radial Basis Function (RBF) Kernel



Sigmoid Kernel



Important SVM Hyperparameters:



C: Controls margin width and misclassification



Gamma: Controls the influence of individual data points



Degree: Used for polynomial kernel



Hyperparameter Tuning:



Two tuning strategies are discussed:



Grid Search: Exhaustively searches through all parameter combinations using cross-validation



Implementation Steps:

Load the dataset

Handle missing values

Standardize numerical features

Perform Exploratory Data Analysis (EDA)

Split data into training and testing sets

Train baseline Logistic Regression model

Tune Logistic Regression hyperparameters

Train SVM with different kernels

Tune SVM hyperparameters

Evaluate models using performance metrics

Perform 5-Fold Cross-Validation

Exploratory Data Analysis:



Class distribution was analyzed to ensure there is no severe imbalance between spam and ham emails.



Text length and word count distributions were studied.



Boxplots were used to visualize spread and outliers in text length.



Hyperparameter Search Space:



Logistic Regression:



Regularization: L1, L2



C values: 0.01, 0.1, 1, 10, 100



Solvers: liblinear, saga



Support Vector Machine:



Kernels: Linear, Polynomial, RBF, Sigmoid



C values: 0.1, 1, 10, 100



Gamma: scale, auto



Polynomial Degree: 2, 3, 4



Best Hyperparameters Obtained:



Logistic Regression:

C = 10

Penalty = L2

Solver = liblinear

Best Cross-Validation Accuracy = 0.9819



Support Vector Machine:

Kernel = RBF

C = 10

Gamma = scale

Best Cross-Validation Accuracy = 0.9850



Performance Evaluation:



Final Test Set Results:



Logistic Regression:

Accuracy = 0.9855

Precision = 0.9766

Recall = 0.9733

F1 Score = 0.9750



Support Vector Machine (RBF):

Accuracy = 0.9874

F1 Score = 0.9785



Confusion matrices were plotted for both models to visualize classification performance.



ROC curves and AUC values were generated, showing excellent discriminative ability for both classifiers.



Bias–Variance Analysis:



Both models achieved accuracy greater than 98% on validation and test datasets, indicating low bias.

The closeness between cross-validation and test scores indicates low variance and good generalization.

SVM with RBF kernel showed a slight improvement over Logistic Regression.

