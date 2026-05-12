
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load Excel file
file_path = "demo_Correct_Updated_Student_Dataset_Final.xlsx"
data = pd.read_excel(file_path)

# Display the first 5 rows
print(data.head())

print(data.info())
print(data.describe())
'''
# Data Distribution
plt.figure(figsize=(12, 6))
sns.histplot(data["Placement Status"].astype(str), kde=False, bins=3)
plt.title("Placement Status Distribution")
plt.show()

'''

# Find and print feature correlation
numeric_data = data.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_data.corr()
print("Feature Correlation Matrix:", correlation_matrix)

# Find and print correlation between Placement and numerical features
if data['Placement Status'].dtype == 'object':
    data['Placement Status'] = LabelEncoder().fit_transform(data['Placement Status'])  # Ensure Placement is numeric

numeric_data = data.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_data.corr()['Placement Status'].drop('Placement Status') # Exclude self-correlation


# Sort the correlation values
sorted_correlation = correlation_matrix.sort_values(ascending=False)

# Print sorted correlation values
print("Sorted correlation with Placement Status:")
print(sorted_correlation)
'''

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())'''
'''
# Skills and Placement Status relation (Java, Python, C++, etc.)
skills_columns = ['Java', 'Python', 'C++', 'ML', 'AI', 'SQL', 'Tableau', 'JavaScript', 'DSA', 'ReactJS']

for col in skills_columns:
    sns.barplot(x='Placement Status', y=col, data=data)
    plt.title(f'{col} by Placement Status')
    plt.show()

'''
# Plotting count distribution for Gender
sns.countplot(data=data, x='Gender')
plt.title('Gender Distribution')
plt.show()



# Histograms for numerical features
data[['Graduation Marks', 'Technical Score (out of 20)', 'Quants']].hist(bins=15, figsize=(12, 8))
plt.suptitle('Histograms of Numerical Features')
plt.show()



# Boxplot for outliers in Graduation Marks
sns.boxplot(x=data['Graduation Marks'])
plt.title('Boxplot of Graduation Marks')
plt.show()

# Violin plot for Technical Score by Placement Status
sns.violinplot(x='Placement Status', y='Technical Score (out of 20)', data=data)
plt.title('Technical Score by Placement Status')
plt.show()


print(data['Placement Status'].head())
print(data['Placement Status'].isnull().sum())

