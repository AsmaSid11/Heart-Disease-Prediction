"""
Heart Disease Prediction - EDA Demo
Demonstrating NumPy, Pandas, and Matplotlib

Dataset: Heart Disease Dataset from Kaggle
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for better visualizations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ============================================================================
# PART 1: PANDAS - Data Loading and Exploration
# ============================================================================
print("="*70)
print("PART 1: PANDAS - DATA LOADING AND EXPLORATION")
print("="*70)

# Load the dataset
# Assuming the file is named 'heart.csv' - adjust the path as needed
df = pd.read_csv('heart.csv')

print("\n1.1 First Few Rows:")
print(df.head())

print("\n1.2 Dataset Information:")
print(df.info())

print("\n1.3 Statistical Summary:")
print(df.describe())

print("\n1.4 Shape of Dataset:")
print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

print("\n1.5 Missing Values:")
print(df.isnull().sum())

print("\n1.6 Target Variable Distribution:")
print(df['target'].value_counts())

# ============================================================================
# PART 2: NUMPY - Numerical Computations
# ============================================================================
print("\n" + "="*70)
print("PART 2: NUMPY - NUMERICAL COMPUTATIONS")
print("="*70)

# Convert pandas columns to numpy arrays for demonstration
age = df['age'].values
chol = df['chol'].values
trestbps = df['trestbps'].values

print("\n2.1 Basic NumPy Statistics:")
print(f"Mean Age: {np.mean(age):.2f}")
print(f"Median Cholesterol: {np.median(chol):.2f}")
print(f"Std Dev of Blood Pressure: {np.std(trestbps):.2f}")

print("\n2.2 Percentiles:")
print(f"25th percentile of Age: {np.percentile(age, 25):.2f}")
print(f"75th percentile of Age: {np.percentile(age, 75):.2f}")

print("\n2.3 Correlation between Age and Cholesterol:")
correlation = np.corrcoef(age, chol)[0, 1]
print(f"Correlation coefficient: {correlation:.4f}")

print("\n2.4 Vectorized Operations (Age Categories):")
# Using NumPy's where for conditional operations
age_categories = np.where(age < 40, 'Young',
                          np.where(age < 55, 'Middle-aged', 'Senior'))
unique, counts = np.unique(age_categories, return_counts=True)
for cat, count in zip(unique, counts):
    print(f"{cat}: {count}")

# Calculate correlation matrix using NumPy
print("\n2.5 Correlation Matrix (NumPy):")
numeric_cols = df.select_dtypes(include=[np.number]).columns
correlation_matrix = np.corrcoef(df[numeric_cols].T)
print(f"Shape: {correlation_matrix.shape}")

# ============================================================================
# PART 3: PANDAS - Advanced Data Analysis
# ============================================================================
print("\n" + "="*70)
print("PART 3: PANDAS - ADVANCED DATA ANALYSIS")
print("="*70)

print("\n3.1 Group By Analysis - Heart Disease by Sex:")
print(df.groupby('sex')['target'].value_counts().unstack(fill_value=0))

print("\n3.2 Age Groups Analysis:")
df['age_group'] = pd.cut(df['age'], bins=[0, 40, 55, 70, 100], 
                          labels=['Young', 'Middle-aged', 'Senior', 'Elderly'])
print(df.groupby('age_group')['target'].mean())

print("\n3.3 Chest Pain Type Analysis:")
print(df.groupby('cp')['target'].agg(['count', 'mean']))

print("\n3.4 High Cholesterol Cases (>240):")
high_chol = df[df['chol'] > 240]
print(f"Number of cases: {len(high_chol)}")
print(f"Heart disease rate: {high_chol['target'].mean():.2%}")

# ============================================================================
# PART 4: MATPLOTLIB - Data Visualization
# ============================================================================
print("\n" + "="*70)
print("PART 4: MATPLOTLIB - DATA VISUALIZATION")
print("="*70)

# Create a figure with multiple subplots
fig = plt.figure(figsize=(16, 12))

# 4.1 Age Distribution
plt.subplot(3, 3, 1)
plt.hist(df['age'], bins=20, color='skyblue', edgecolor='black', alpha=0.7)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.grid(axis='y', alpha=0.3)

# 4.2 Heart Disease Distribution
plt.subplot(3, 3, 2)
target_counts = df['target'].value_counts()
plt.bar(['No Disease', 'Disease'], target_counts.values, color=['lightcoral', 'lightgreen'])
plt.xlabel('Target')
plt.ylabel('Count')
plt.title('Heart Disease Distribution')
for i, v in enumerate(target_counts.values):
    plt.text(i, v + 5, str(v), ha='center', fontweight='bold')

# 4.3 Cholesterol Distribution by Target
plt.subplot(3, 3, 3)
plt.boxplot([df[df['target']==0]['chol'], df[df['target']==1]['chol']], 
            labels=['No Disease', 'Disease'])
plt.ylabel('Cholesterol Level')
plt.title('Cholesterol by Heart Disease')
plt.grid(axis='y', alpha=0.3)

# 4.4 Age vs Max Heart Rate (Scatter)
plt.subplot(3, 3, 4)
colors = ['red' if x == 1 else 'blue' for x in df['target']]
plt.scatter(df['age'], df['thalach'], c=colors, alpha=0.5)
plt.xlabel('Age')
plt.ylabel('Max Heart Rate')
plt.title('Age vs Max Heart Rate')
plt.legend(['Disease', 'No Disease'])

# 4.5 Chest Pain Type Distribution
plt.subplot(3, 3, 5)
cp_counts = df['cp'].value_counts().sort_index()
plt.bar(cp_counts.index, cp_counts.values, color='coral')
plt.xlabel('Chest Pain Type')
plt.ylabel('Count')
plt.title('Chest Pain Type Distribution')
plt.xticks(cp_counts.index)

# 4.6 Correlation Heatmap
plt.subplot(3, 3, 6)
corr_features = ['age', 'trestbps', 'chol', 'thalach', 'target']
corr_matrix = df[corr_features].corr()
im = plt.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
plt.colorbar(im)
plt.xticks(range(len(corr_features)), corr_features, rotation=45)
plt.yticks(range(len(corr_features)), corr_features)
plt.title('Correlation Heatmap')
for i in range(len(corr_features)):
    for j in range(len(corr_features)):
        plt.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                ha='center', va='center', color='black', fontsize=8)

# 4.7 Blood Pressure Distribution
plt.subplot(3, 3, 7)
plt.hist([df[df['target']==0]['trestbps'], df[df['target']==1]['trestbps']], 
         bins=15, label=['No Disease', 'Disease'], alpha=0.7)
plt.xlabel('Resting Blood Pressure')
plt.ylabel('Frequency')
plt.title('Blood Pressure Distribution')
plt.legend()
plt.grid(axis='y', alpha=0.3)

# 4.8 Age Group vs Heart Disease
plt.subplot(3, 3, 8)
age_disease = df.groupby('age_group')['target'].mean() * 100
plt.bar(range(len(age_disease)), age_disease.values, color='mediumpurple')
plt.xlabel('Age Group')
plt.ylabel('Heart Disease Rate (%)')
plt.title('Heart Disease Rate by Age Group')
plt.xticks(range(len(age_disease)), age_disease.index, rotation=45)
for i, v in enumerate(age_disease.values):
    plt.text(i, v + 1, f'{v:.1f}%', ha='center', fontweight='bold')

# 4.9 Sex vs Heart Disease
plt.subplot(3, 3, 9)
sex_disease = df.groupby('sex')['target'].value_counts().unstack()
sex_disease.plot(kind='bar', ax=plt.gca(), color=['lightcoral', 'lightgreen'])
plt.xlabel('Sex (0=Female, 1=Male)')
plt.ylabel('Count')
plt.title('Heart Disease by Sex')
plt.legend(['No Disease', 'Disease'])
plt.xticks(rotation=0)

plt.tight_layout()
plt.savefig('heart_disease_eda.png', dpi=300, bbox_inches='tight')
print("\n✓ Visualizations saved as 'heart_disease_eda.png'")
plt.show()

# ============================================================================
# SUMMARY STATISTICS
# ============================================================================
print("\n" + "="*70)
print("SUMMARY: KEY FINDINGS")
print("="*70)

print(f"\n→ Total Patients: {len(df)}")
print(f"→ Heart Disease Cases: {df['target'].sum()} ({df['target'].mean()*100:.1f}%)")
print(f"→ Average Age: {df['age'].mean():.1f} years")
print(f"→ Average Cholesterol: {df['chol'].mean():.1f} mg/dl")
print(f"→ Average Max Heart Rate: {df['thalach'].mean():.1f} bpm")
print(f"→ Strongest Correlation with Target: {df.corr()['target'].abs().sort_values(ascending=False).index[1]}")

print("\n" + "="*70)
print("EDA COMPLETE!")
print("="*70)