# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("50_Startups.csv")
print(df.head())

print(df.info())
print(df.describe())  # Summary statistics

correlation_matrix = df[['R&D Spend', 'Marketing Spend', 'Administration', 'Profit']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

predictors = ['R&D Spend', 'Marketing Spend', 'Administration']
for col in predictors:
    plt.scatter(df[col], df['Profit'])
    plt.xlabel(col)
    plt.ylabel("Profit")
    plt.title(f"{col} vs Profit")
    plt.show()

X = df[['R&D Spend', 'Marketing Spend', 'Administration']]  # Independent variables
y = df['Profit']  # Dependent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
r2_train = r2_score(y_train, y_train_pred)

rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
r2_test = r2_score(y_test, y_test_pred)

print(f"Training RMSE: {rmse_train}, Training R²: {r2_train}")
print(f"Testing RMSE: {rmse_test}, Testing R²: {r2_test}")


"""
Findings:
1. The correlation matrix helps identify variables that strongly impact profit.
2. 'R&D Spend' and 'Marketing Spend' are highly correlated with profit.
3. Scatter plots confirm (almost) linear relationships.
4. The linear regression model provides RMSE and R² values to measure accuracy.
5. If R² is low, consider feature engineering or polynomial regression.
"""