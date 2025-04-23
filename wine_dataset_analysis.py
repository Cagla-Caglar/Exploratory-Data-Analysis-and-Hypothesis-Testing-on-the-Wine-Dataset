import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from scipy.stats import boxcox

# Read the dataset from the local project directory
file_path = "data/wine.csv"
df = pd.read_csv(file_path)

# Display basic information about the dataset
print(df.head(10))
print(df.describe())
print(df.info())

# Plot histograms before transformation
for col in df.columns[:-1]:
    sns.displot(data=df, x=col, kde=True)
    plt.title(f'Histogram of {col} before transformation')
    plt.show()

# Check for missing values
print(df.isnull().sum())

# Apply Box-Cox transformation to numerical features (excluding target variable)
numerical_features = df.select_dtypes(include=[np.number]).columns
for col in numerical_features:
    if col != 'quality':
        df[col] = df[col] + 0.01  
        df[col], lambda_value = boxcox(df[col])

# Plot histograms after transformation
for col in df.columns[:-1]:
    sns.displot(data=df, x=col, kde=True)
    plt.title(f'Histogram of {col} after transformation')
    plt.show()

# Separate the dataset by wine color
x_red = df[df['color'] == 'red'].drop(columns=['color'])
x_white = df[df['color'] == 'white'].drop(columns=['color'])

# Perform a two-sample t-test on pH values
x_red_sample_pH = x_red["pH"].sample(n=1000)
x_white_sample_pH = x_white["pH"].sample(n=1000)
t_stat, p_value = stats.ttest_ind(x_red_sample_pH, x_white_sample_pH)
print(f"T-Statistic: {t_stat}, P-Value: {p_value}")

# One-hot encode the 'color' column
ohe = OneHotEncoder()
ct = ColumnTransformer(transformers=[('encode', OneHotEncoder(), ['color'])], remainder='passthrough')
xff = ct.fit_transform(df)

print(xff.shape)
