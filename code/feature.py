import pandas as pd

# Load your processed data
data = pd.read_excel("processed_student_data.xlsx")

# Encode target variable if not already encoded
from sklearn.preprocessing import LabelEncoder
data['Placement Status'] = LabelEncoder().fit_transform(data['Placement Status'])

# Select numeric featurescls

numeric_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_features.remove('Placement Status')

# Calculate correlation with target
correlations = data[numeric_features + ['Placement Status']].corr()['Placement Status'].abs()

# Set a correlation threshold (e.g., 0.1)
threshold = 0.1

# Select features above the threshold
selected_features = correlations[correlations > threshold].index.drop('Placement Status').tolist()

print("Selected Features based on correlation threshold of", threshold, ":")
print(selected_features)
