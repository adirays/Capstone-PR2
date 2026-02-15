# eda.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("C:/Users/91810/Downloads/heart_disease_dataset.csv")

# 1️⃣ Basic Information
print("First 5 Rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nStatistical Summary:")
print(df.describe())

print("\nMissing Values:")
print(df.isnull().sum())

# 2️⃣ Check Target Distribution
if "target" in df.columns:
    plt.figure()
    sns.countplot(x="target", data=df)
    plt.title("Target Variable Distribution")
    plt.show()

# 3️⃣ Correlation Matrix
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# 4️⃣ Histograms of Numerical Columns
df.hist(figsize=(12, 10))
plt.tight_layout()
plt.show()
