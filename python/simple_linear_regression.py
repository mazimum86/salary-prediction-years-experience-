# linear_regression.py

"""
📊 Simple Linear Regression on Salary Dataset

This script trains a Linear Regression model to predict salary based on years of experience.
It saves the predicted results and visualizes both the training and test set results.

Author: Chukwuka Chijioke Jerry
"""

# =========================
# 📦 Import Libraries
# =========================
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import os

# =========================
# 📥 Load Dataset
# =========================
df = pd.read_csv("../data/Salary_Data.csv")  # Ensure CSV is in 'data/' folder
X = df[['YearsExperience']]
y = df['Salary']

# =========================
# ✂️ Split the Dataset
# =========================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# =========================
# 🧠 Train the Model
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# 🔍 Model Coefficients
# =========================
print(f"Intercept: {model.intercept_:.2f}")
print(f"Coefficient: {model.coef_[0]:.2f}")

# =========================
# 📁 Create Output Folders
# =========================
os.makedirs("../outputs", exist_ok=True)
os.makedirs("../plots", exist_ok=True)

# =========================
# 🧾 Save Predictions
# =========================
predictions = model.predict(X_test)
results = pd.DataFrame({
    'YearsExperience': X_test.values.flatten(),
    'ActualSalary': y_test.values,
    'PredictedSalary': predictions
})
results.to_csv("../outputs/SLR_predicted_salaries.csv", index=False)

# =========================
# 📈 Plot - Training Set
# =========================
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, color='blue', label='Actual (Train)')
plt.plot(X_train, model.predict(X_train), color='red', label='Prediction')
plt.title("Simple Linear Regression (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../plots/simple_linear_regression_plot_train")
plt.show()

# =========================
# 📈 Plot - Test Set
# =========================
plt.figure(figsize=(8, 5))
plt.scatter(X_test, y_test, color='green', label='Actual (Test)')
plt.plot(X_test, model.predict(X_test), color='red', label='Prediction')
plt.title("Simple Linear Regression (Test Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("../plots/simple_linear_regression_plot_test")
plt.show()

# ✅ Done
print("✅ Model training complete. Results saved to 'outputs/' and plots to 'plots/'.")
