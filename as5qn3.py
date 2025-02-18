import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

file_path = "Auto.csv"

try:
    df = pd.read_csv(file_path, na_values="?")
except FileNotFoundError:
    print("Error: Auto.csv not found. Ensure the file is in the correct directory.")
    raise


df.dropna(inplace=True)
df.drop(columns=['name', 'origin'], inplace=True)


X = df.drop(columns=['mpg'])
y = df['mpg']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

alphas = np.logspace(-3, 3, 50)
ridge_scores = []
lasso_scores = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    ridge_scores.append(r2_score(y_test, ridge.predict(X_test)))

    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(X_train, y_train)
    lasso_scores.append(r2_score(y_test, lasso.predict(X_test)))

best_alpha_ridge = alphas[np.argmax(ridge_scores)]
best_alpha_lasso = alphas[np.argmax(lasso_scores)]

plt.figure(figsize=(10,5))
plt.plot(alphas, ridge_scores, label="Ridge R²", marker='o', linestyle='dashed')
plt.plot(alphas, lasso_scores, label="LASSO R²", marker='s', linestyle='dotted')
plt.xscale('log')
plt.xlabel("Alpha")
plt.ylabel("R² Score")
plt.title("R² Score vs. Alpha (Ridge & LASSO)")
plt.legend()
plt.grid()
plt.show()


print(f"Best Ridge Alpha: {best_alpha_ridge:.4f}, Best R²: {max(ridge_scores):.4f}")
print(f"Best LASSO Alpha: {best_alpha_lasso:.4f}, Best R²: {max(lasso_scores):.4f}")

"""
Findings:
1. Ridge and LASSO Regression were applied with a range of alpha values.
2. The best alpha values were identified based on the highest R² score.
3. The R² score decreases when alpha is too high, causing over-regularization.
4. Ridge regression tends to perform better when features are highly correlated.
5. LASSO regression can shrink some coefficients to zero, effectively performing feature selection.
"""
