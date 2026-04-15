import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import cross_val_score
# Import the library that contains all the functions/modules related to the regression model


X = pd.read_csv(r"X.csv")
egg = pd.read_csv(r"y.csv")
# Join X and y on their indices
X['egg_price'] = egg

# Add a new column 'Y' which is the 'egg_price' column shifted down by 1 entry
X['Y'] = X['egg_price'].shift(-1)
X_train, X_test, y_train, y_test = train_test_split(X.drop(columns=['Y', 'Year-Month']), X['Y'], test_size=0.25, shuffle = False)

# We must add an intercept as the standard model doesn't automatically fit one
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)
lr = LinearRegression()
lr.fit(X_train, y_train)
pred = lr.predict(X_test)

# Calculate the mean squared error and R-squared value  
mse = mean_squared_error(y_test.iloc[:-1], pred[:-1])
r2 = r2_score(y_test.iloc[:-1], pred[:-1])

# Calculate the residuals
residuals = y_test.iloc[:-1] - pred[:-1]

# Create a DataFrame for the residuals  
residuals_df = pd.DataFrame({'Actual': y_test.iloc[:-1], 'Predicted': pred[:-1], 'Residuals': residuals})
# Perform cross-validation
cv_scores = cross_val_score(lr, X_train, y_train, cv=5, scoring='neg_mean_squared_error')

# Convert negative MSE to positive for interpretation
cv_mse_scores = -cv_scores
feature_importance = pd.Series(lr.coef_, index=X_train.columns)
feature_importance.sort_values()