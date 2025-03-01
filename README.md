# Linear Regression Project: Housing Prices Prediction


## Overview
This project is developed as part of the MATH 372: Linear Regression Analysis final project. The objective is to create an end-to-end linear regression function that fits a model using techniques learned in class. The project uses multiple linear regression methods to predict housing prices in California. The primary goal is to help users select between predictive or explanatory models, using cross-validation and grid-search when necessary.

The function will take a predictor matrix (or data frame) **_X_** and a continuous-valued response **_y_** as inputs and will:
- Preprocess data and handle missing observations
- Develop a predictive model for y
- Fit a parsimonious explanatory model that balances explanation and prediction
___________________

### Dataset
The dataset used in this project is from Kaggle's California Housing Prices dataset. It contains various housing features across districts in California, including attributes like median income, housing age, and proximity to the ocean.


Dataset Source: Kaggle - California Housing Prices

https://www.kaggle.com/datasets/camnugent/california-housing-prices

___________________

### Objectives
1. Data Preprocessing
- Handle missing data through imputation or removal.
- Detect and remove outliers and influential points.
- Scale features for model training.

2. Model Development
- Implement and compare Lasso, Ridge, and OLS regression models.
- Use cross-validation and grid search for hyperparameter tuning where applicable.
- Fit a parsimonious model for explanatory purposes or a predictive model, based on user preference.

3. Model Diagnostics and Evaluation
- Evaluate model performance using various metrics such as RMSE, AIC, BIC, and Adjusted R^2.
- Perform hypothesis tests for normality, homoscedasticity, and linearity.
- Check for outliers, influential points, and high-leverage points using appropriate plots (e.g., leverage vs. Cook’s distance).
- Assess if any transformation is needed on the response variable **_y_**

4. Model Selection
- Perform model selection based on performance metrics, including MSE, AIC, BIC, Mallow’s Cp, and Adjusted R^2.
- Use formal F-tests for nested models, when appropriate.
- Provide diagnostic plots for visual assessment.

___________________
### Models Implemented
- OLS Regression (Ordinary Least Squares): Baseline model.
- Ridge Regression: L2 regularization for penalizing large coefficients.
- Lasso Regression: L1 regularization for promoting sparsity in coefficients.
___________________
### Results and Analysis
1. Model Selection Summary
- OLS Regression was chosen as the final model based on the lowest test RMSE (73,559.74) and balanced performance across complexity metrics.
- Ridge and Lasso models were compared but did not outperform OLS in terms of test error and explained variance.

2. Diagnostics
- Normality of Residuals: Residuals were tested and found to deviate from normality.
- Homoscedasticity: Breusch-Pagan test revealed heteroscedasticity in residuals.
- Outliers and Influential Points: Leverage plots and Cook’s Distance identified influential data points.

3. Model Evaluation
- AIC: 398,486.02
- BIC: 398,539.70
- Adjusted R^2 : 0.57
- RMSE: 73,559.74
___________________

### Technologies Used
Python Libraries:
- Pandas
- Numpy
- Scikit-learn
- Statsmodels
- Matplotlib
- Seaborn


__________________

### Explanatory Plots
- Leverage Values: Identifies high-leverage points.
- Cook's Distance vs. Leverage: Visualizes influential points.
- Residuals Distribution: Tests for normality of residuals.
- Q-Q Plot of Residuals: Assesses normality of residuals.
- Residuals vs. Predicted Values: Visualizes homoscedasticity.


__________________
### Conclusion
- The final model selected, OLS Regression, demonstrated the best trade-off between complexity and predictive power.
- The diagnostics and performance evaluation suggest further refinement of the model by considering potential transformations of the response variable **_y_** and improving homoscedasticity.

