# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")

# Step 2: Load Data
df = pd.read_csv("train.csv")
print("Shape of data:", df.shape)
print("\nFirst 5 rows:\n", df.head())

# Step 3: Basic Info
print("\nInfo:\n")
print(df.info())
print("\nMissing values:\n", df.isnull().sum())

# Step 4: Data Cleaning
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)
df.drop('Cabin', axis=1, inplace=True)
df.drop(['PassengerId', 'Name', 'Ticket'], axis=1, inplace=True)

# Step 5: EDA Visualizations
# 1. Survival Count
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.show()

# 2. Gender vs Survival
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival by Gender")
plt.show()

# 3. Pclass vs Survival
sns.countplot(x='Pclass', hue='Survived', data=df)
plt.title("Survival by Class")
plt.show()

# 4. Age Distribution
sns.histplot(df['Age'], kde=True, bins=30)
plt.title("Age Distribution")
plt.show()

# 5. Age vs Survival
sns.boxplot(x='Survived', y='Age', data=df)
plt.title("Age vs Survival")
plt.show()

# Step 6: Correlation Heatmap
corr = df.corr(numeric_only=True)
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Step 7: Export Cleaned Data
df.to_csv("cleaned_titanic.csv", index=False)
print("✅ Cleaned data saved as 'cleaned_titanic.csv'")

print("\n--- Key Insights ---")
print("1. Females had higher survival rates than males.")
print("2. First class passengers were more likely to survive.")
print("3. Most missing data came from Cabin (dropped).")
from ucimlrepo import fetch_ucirepo 
  
# fetch dataset 
bank_marketing = fetch_ucirepo(id=222) 
  
# data (as pandas dataframes) 
X = bank_marketing.data.features 
y = bank_marketing.data.targets 
  
# metadata 
print(bank_marketing.metadata) 
  
# variable information 
print(bank_marketing.variables) 
