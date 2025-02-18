import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


data = load_diabetes(as_frame=True)
df = data['frame']
y = df['target']

print(data.keys())
print(data["DESCR"])
print(data.DESCR)
print(df)
plt.hist(df['target'], 25)
plt.xlabel('target')
plt.show()

sns.heatmap(data=df.corr().round(2), annot=True)
plt.show()

plt.subplot(1, 2, 1)
plt.scatter(df['bmi'], df['target'])
plt.xlabel('bmi')
plt.ylabel('target')
plt.subplot(1, 2, 2)
plt.scatter(df['s5'], df['target'])
plt.xlabel('s5')
plt.ylabel('target')
plt.show()

x = pd.DataFrame(df[['bmi', 's5']], columns=['bmi', 's5'])
print(x)
print(y)

X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=5, test_size=0.2)
print(X_train.shape)
print(X_test.shape)

lin = LinearRegression()
lin.fit(X_train, y_train)

y_train_predict = lin.predict(X_train)
rmse_train = np.sqrt(mean_squared_error(y_train, y_train_predict))
r2_train = r2_score(y_train, y_train_predict)

y_test_predict = lin.predict(X_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_test_predict))
r2_test = r2_score(y_test, y_test_predict)

print(f"RMSE Train with 'bmi', 's5': {rmse_train:.2f}")
print(f"R² Train with 'bmi', 's5': {r2_train:.2f}")
print(f"RMSE Test with 'bmi', 's5': {rmse_test:.2f}")
print(f"R² Test with 'bmi', 's5': {r2_test:.2f}")

x_new = pd.DataFrame(df[['bmi', 's5', 'age']], columns=['bmi', 's5', 'age'])

X_train, X_test, y_train, y_test = train_test_split(x_new, y, random_state=5, test_size=0.2)

lin.fit(X_train, y_train)

y_train_predict = lin.predict(X_train)
rmse_train_new = np.sqrt(mean_squared_error(y_train, y_train_predict))
r2_train_new = r2_score(y_train, y_train_predict)

y_test_predict = lin.predict(X_test)
rmse_test_new = np.sqrt(mean_squared_error(y_test, y_test_predict))
r2_test_new = r2_score(y_test, y_test_predict)

print(f"RMSE Train with 'bmi', 's5', 'age': {rmse_train_new:.2f}")
print(f"R² Train with 'bmi', 's5', 'age': {r2_train_new:.2f}")
print(f"RMSE Test with 'bmi', 's5', 'age': {rmse_test_new:.2f}")
print(f"R² Test with 'bmi', 's5', 'age': {r2_test_new:.2f}")

x_more_features = pd.DataFrame(df[['bmi', 's5', 'age', 'bp']], columns=['bmi', 's5', 'age', 'bp'])

X_train, X_test, y_train, y_test = train_test_split(x_more_features, y, random_state=5, test_size=0.2)

lin.fit(X_train, y_train)

y_train_predict = lin.predict(X_train)
rmse_train_more = np.sqrt(mean_squared_error(y_train, y_train_predict))
r2_train_more = r2_score(y_train, y_train_predict)


y_test_predict = lin.predict(X_test)
rmse_test_more = np.sqrt(mean_squared_error(y_test, y_test_predict))
r2_test_more = r2_score(y_test, y_test_predict)

print(f"RMSE Train with 'bmi', 's5', 'age', 'bp': {rmse_train_more:.2f}")
print(f"R² Train with 'bmi', 's5', 'age', 'bp': {r2_train_more:.2f}")
print(f"RMSE Test with 'bmi', 's5', 'age', 'bp': {rmse_test_more:.2f}")
print(f"R² Test with 'bmi', 's5', 'age', 'bp': {r2_test_more:.2f}")


"""
- Initially, using 'bmi' and 's5' provided a baseline model. 
- Adding 'age' improved the model because age correlates well with diabetes progression.
- Adding more variables, like 'bp', further improved the performance, but diminishing returns might occur after a certain point. 
- The model might start overfitting if too many features are added, which can cause the test set performance to drop.
"""

