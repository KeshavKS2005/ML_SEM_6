\# Regression Analysis using Linear and Regularized Models



\### 1. Data Loading

The dataset is loaded from a CSV source containing both \*\*numerical and categorical features\*\* related to loan applicants.  

The target variable is the \*\*loan amount sanctioned\*\*.

The features are all present in USD.



---



\### 2. Data Preprocessing





\- \*\*Categorical Encoding\*\*

&nbsp; - Non-numeric features such as profession, employment type, property details, etc., are converted into numerical form using label encoding techniques.

\- \*\*Feature Scaling\*\*

&nbsp; - Numerical features are standardized using `StandardScaler`.

&nbsp; - This step is critical for Ridge, Lasso, and Elastic Net models, as they are sensitive to feature magnitude.



---



\### 3. Exploratory Data Analysis (EDA)

The code performs exploratory analysis to understand data distribution and relationships:



\- Distribution of the target variable (Loan Amount)

\- Feature distributions

\- Correlation Map

&nbsp;

\- Scatter plots between key features and the target





These visualizations help justify preprocessing choices and model selection.



---



\### 4. Train–Test Split

The dataset is split into:

\- \*\*Training set\*\*: Used to fit models

\- \*\*Test set\*\*: Used for final evaluation



This ensures that model performance is evaluated on \*\*unseen data\*\*, a measure of generalization.



---



\### 5. Baseline Model: Linear Regression

A standard \*\*Linear Regression\*\* model is trained as a baseline:





\### 6. Regularized Regression Models

The following regularized models are implemented to control model complexity:



\- \*\*Ridge Regression\*\*

&nbsp; - Applies L2 regularization

&nbsp; - Reduces large coefficients





\- \*\*Lasso Regression\*\*

&nbsp; - Applies L1 regularization

&nbsp; - Forces some coefficients to zero

&nbsp; 

\- \*\*Elastic Net Regression\*\*

&nbsp; - Combines L1 and L2 penalties

&nbsp; - Balances coefficient shrinkage and sparsity



---



\### 7. Hyperparameter Tuning

To obtain optimal model performance, \*\*Grid Search with 5-Fold Cross-Validation\*\* is used:



\- Searches over predefined values of:

&nbsp; - `alpha` (regularization strength)

&nbsp; - `l1\_ratio` (Elastic Net)

\- Selects the best parameters based on \*\*cross-validated R² score\*\*



This step ensures models are neither over-regularized nor under-regularized.



---



\### 8. Model Evaluation

Each model is evaluated using multiple regression metrics:



\- \*\*MAE (Mean Absolute Error)\*\* – average magnitude of errors

\- \*\*MSE (Mean Squared Error)\*\* – penalizes large errors

\- \*\*RMSE (Root Mean Squared Error)\*\* – interpretable error scale

\- \*\*R² Score\*\* – variance explained by the model

\- \*\*Training Time\*\* – computational efficiency



Evaluation is performed on:

\- Cross-validation results

\- Final test dataset



---



\### 9. Model Interpretation

To understand model behavior, the code includes:



\- \*\*Predicted vs Actual plots\*\*

\- \*\*Residual analysis\*\*

\- \*\*Learning curves\*\* 



This highlights:

\- Effect of regularization on coefficients

\- Feature importance changes

\- Bias–variance trade-offs



---



\### 10. Overfitting and Bias–Variance Analysis

The code explicitly analyzes:



\- Difference between training and validation errors

\- Impact of increasing regularization strength

\- Variance reduction using Ridge and Elastic Net

\- Feature sparsity introduced by Lasso



This theoretical analysis is supported by experimental results.



---



\## Technologies Used

\- Python

\- NumPy

\- Pandas

\- Matplotlib

\- Seaborn

\- scikit-learn



---



\## Files in This Repository

\- `ml\_lab\_3.ipynb` – Complete implementation with outputs and visualizations

\- `ML\_03\_final.pdf` – Final lab report

\- `README.md` – Project documentation





