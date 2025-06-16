# linear_regression.py

"""
Linear Regression on Salary Dataset

Predict salary based on years of experience using Linear Regression.
Author: Chukwuka Chijioke Jerry
"""

# =========================
# 📦 Import Libraries
# =========================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# =========================
# 📥 Load Dataset
# =========================
df = pd.read_csv("Salary_Data.csv")
X = df[['YearsExperience']]  # Feature
y = df['Salary']             # Target

# =========================
# 🧠 Train Model
# =========================
model = LinearRegression()
model.fit(X, y)

# =========================
# 🔍 Model Coefficients
# =========================
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")

# =========================
# 📈 Visualization
# =========================
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Prediction')
plt.title("Linear Regression: Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.savefig("linear_regression.png")
plt.show()
