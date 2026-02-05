Regression Analysis using Linear and Regularized Models

Data Loading

The dataset is loaded from a CSV source containing both numerical and categorical features related to loan applicants.
The target variable is the loan amount sanctioned.
All financial features are represented in USD.

Data Preprocessing

Categorical Encoding
Non-numeric features such as profession, employment type, and property details are converted into numerical form using label encoding techniques.

Feature Scaling
Numerical features are standardized using StandardScaler.
This step is critical for Ridge, Lasso, and Elastic Net models, as they are sensitive to feature magnitude.

Exploratory Data Analysis (EDA)

Exploratory analysis is performed to understand data distribution and relationships.
This includes distribution analysis of the target variable (loan amount), feature-wise distributions, correlation mapping, and scatter plots between key features and the target variable.

These visualizations help justify preprocessing choices and model selection.

Train–Test Split

The dataset is split into a training set and a test set.
The training set is used to fit the models, while the test set is used for final evaluation.
This ensures that model performance is evaluated on unseen data, providing a measure of generalization.

Baseline Model: Linear Regression

A standard Linear Regression model is trained as a baseline to compare against regularized models.

Regularized Regression Models

Several regularized models are implemented to control model complexity.

Ridge Regression
This model applies L2 regularization and helps reduce large coefficient values.

Lasso Regression
This model applies L1 regularization and forces some coefficients to zero, enabling feature selection.

Elastic Net Regression
This model combines L1 and L2 penalties to balance coefficient shrinkage and sparsity.

Hyperparameter Tuning

To obtain optimal model performance, Grid Search with five-fold cross-validation is used.
The search is performed over predefined values of the regularization strength (alpha) and the l1 ratio for Elastic Net.
The best parameters are selected based on cross-validated R squared score.

This step ensures that the models are neither over-regularized nor under-regularized.

Model Evaluation

Each model is evaluated using multiple regression metrics.
These include Mean Absolute Error, Mean Squared Error, Root Mean Squared Error, R squared score, and training time to assess computational efficiency.

Evaluation is performed using both cross-validation results and the final test dataset.

Model Interpretation

To understand model behavior, the implementation includes predicted versus actual plots, residual analysis, and learning curves.
These analyses highlight the effect of regularization on coefficients, changes in feature importance, and bias–variance trade-offs.

Overfitting and Bias–Variance Analysis

The code explicitly analyzes the difference between training and validation errors, the impact of increasing regularization strength, variance reduction using Ridge and Elastic Net, and feature sparsity introduced by Lasso.
This theoretical analysis is supported by experimental results.

Technologies Used

Python
NumPy
Pandas
Matplotlib
Seaborn
Scikit-learn
