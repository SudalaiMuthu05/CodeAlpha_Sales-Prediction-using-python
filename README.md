# ✅ TASK 4: Sales Prediction using Python
# Internship Task - CodeAlpha | Author: Krithick

# --------------------------------------------------------
# ✅ Step 1: Import Required Libraries
# --------------------------------------------------------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --------------------------------------------------------
# ✅ Step 2: Load Dataset
# --------------------------------------------------------

df = pd.read_csv("Advertising.csv")  # Make sure this file is in your data/ folder
print(" Preview:\n", df.head())

# --------------------------------------------------------
# ✅ Step 3: Clean the Data
# --------------------------------------------------------

# Check for nulls or duplicates
print("\n Null Values:\n", df.isnull().sum())
print("\n Duplicates:", df.duplicated().sum())

# If there's an 'Unnamed: 0' column (index column), drop it — not needed here though
# df.drop(['Unnamed: 0'], axis=1, inplace=True)

# --------------------------------------------------------
# ✅ Step 4: Data Transformation & Feature Selection
# --------------------------------------------------------

# Visualize correlations
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title(" Feature Correlation Heatmap")
plt.show()

# Drop weakly correlated features (optional)
df.drop(['Newspaper'], axis=1, inplace=True)

# Define input features (X) and target variable (y)
X = df[['TV', 'Radio']]
y = df['Sales']

# --------------------------------------------------------
# ✅ Step 5: Train/Test Split
# --------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --------------------------------------------------------
# ✅ Step 6: Build & Train the Regression Model
# --------------------------------------------------------

model = LinearRegression()
model.fit(X_train, y_train)

# Show learned coefficients
coefficients = pd.DataFrame(model.coef_, X.columns, columns=['Coefficient'])
print("\n Feature Impact:\n", coefficients)

# --------------------------------------------------------
# ✅ Step 7: Predict & Evaluate
# --------------------------------------------------------

y_pred = model.predict(X_test)

# Evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\n Model Evaluation:")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R² Score:", round(r2, 2))

# --------------------------------------------------------
# ✅ Step 8: Visualize Actual vs Predicted
# --------------------------------------------------------

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, c='darkblue', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel("Actual Sales")
plt.ylabel("Predicted Sales")
plt.title("Actual vs Predicted Sales")
plt.grid(True)
plt.show()

# --------------------------------------------------------
# ✅ Step 9: Business Insights
# --------------------------------------------------------

"""
✅ TV had the highest positive impact on Sales.
✅ Newspaper had weak correlation and was dropped to improve accuracy.
✅ R² Score of {:.2f} shows the model explains most of the variation in sales.
✅ Actionable Insight: Invest more in TV and Radio, and minimize Newspaper ads for better ROI.
""".format(r2)
