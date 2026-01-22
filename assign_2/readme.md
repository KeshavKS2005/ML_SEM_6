\# Binary Email Classification using Naive Bayes and K-Nearest Neighbors



This project implements and evaluates multiple machine learning models for a \*\*binary classification problem\*\* using the \*\*Spambase dataset\*\*. 

---



\## 1. Project Workflow Overview







1\. Data loading and exploration  

2\. Data preprocessing and feature scaling  

3\. Model implementation  

4\. Model evaluation using metrics and visualizations  

5\. Hyperparameter tuning and optimization  

6\. Bias–variance and overfitting analysis  





\## 2. Data Loading and Exploration



\- The dataset is loaded .

\- Input features consist of numerical values extracted from email content.

\- The target variable is binary:

&nbsp; - `0` → Non-Spam  

&nbsp; - `1` → Spam  



Initial exploration includes:

\- Dataset shape inspection

\- Class distribution analysis to check imbalance





---



\## 3. Data Preprocessing



The following preprocessing steps are implemented:



\### Train–Test Split

\- The dataset is split into training and testing sets.

\- Stratification ensures the class distribution is preserved in both sets.



\### Feature Scaling

\- \*\*StandardScaler\*\* is applied to normalize features.



---



\## 4. Model Implementation



\### 4.1 Naive Bayes Classifiers



Three variants of Naive Bayes are implemented:



\- \*\*Gaussian Naive Bayes\*\*

&nbsp; - Assumes features follow a normal distribution.

&nbsp; - Works well for continuous features.



\- \*\*Multinomial Naive Bayes\*\*

&nbsp; - Designed for frequency-based features.

&nbsp; - Commonly used in text classification tasks.



\- \*\*Bernoulli Naive Bayes\*\*

&nbsp; - Assumes binary feature values.

&nbsp; - Captures presence or absence of features.



Each model is:

\- Trained on the training dataset

\- Evaluated on the test dataset

\- Compared using consistent evaluation metrics



---



\### 4.2 K-Nearest Neighbors (KNN)



\- A baseline KNN classifier is implemented using \*\*scikit-learn\*\*.

\- The model predicts labels based on majority voting among nearest neighbors.

\- Distance-based weighting is also evaluated.



Key aspects:

\- Different values of `k` are tested

\- Performance trends are observed using accuracy plots



---



\## 5. Hyperparameter Tuning



To optimize KNN performance:



\### Grid Search

\- Exhaustive search over:

&nbsp; - Number of neighbors (`k`)

&nbsp; - Distance metric

&nbsp; - Neighbor search algorithm



\### Randomized Search

\- Random sampling from the hyperparameter space

\- Faster alternative to grid search



The best hyperparameters are selected based on \*\*cross-validation accuracy\*\*.



---



\## 6. Neighbor Search Algorithms



Two spatial indexing methods are compared:



\- \*\*KDTree\*\*

\- \*\*BallTree\*\*

Both are evaluated in terms of:

\- Accuracy

\- Training time

\- Prediction time

\- Memory usage



---



\## 7. Model Evaluation



The models are evaluated using the following metrics:



\- Accuracy

\- Precision

\- Recall

\- F1-Score

\- Specificity

\- False Positive Rate

\- ROC–AUC



\### Visual Evaluations

\- Confusion matrices for each classifier

\- ROC curves to analyze class separation

\- Accuracy vs. `k` plots for KNN

\- Training vs. validation accuracy to detect overfitting



---



\## 8. Overfitting and Underfitting Analysis



\- \*\*Small values of `k`\*\* in KNN lead to overfitting due to high model variance.

\- \*\*Large values of `k`\*\* cause underfitting due to excessive smoothing.

\- Hyperparameter tuning helps achieve better generalization.



---



\## 9. Bias–Variance Analysis



\- Naive Bayes exhibits \*\*high bias\*\* due to its strong independence assumption.

\- KNN has \*\*high variance\*\* for small `k` values.

\- Proper tuning balances the bias–variance trade-off effectively.



---



\## 10. Key Observations



\- Naive Bayes models are computationally efficient but less expressive.

\- KNN achieves higher accuracy after tuning.

\- Visualization plays a key role in diagnosing model behavior.



---



\## 11. Technologies Used



\- Python

\- NumPy

\- Pandas

\- Scikit-learn

\- Matplotlib

\- Seaborn





