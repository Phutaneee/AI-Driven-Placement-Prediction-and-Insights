import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load Excel file
file_path = "demo_Correct_Updated_Student_Dataset_Final.xlsx"
data = pd.read_excel(file_path)

# Display the first 5 rows
print(data.head())
print(data.info())
print(data.describe())

# Print the 'Placement Status' and 'Name of Company' columns
print(data[['Placement Status', 'Name of Company']].head(30))  # Display first 10 rows

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

# Handle missing values
missing_value_replacement = 'Not Placed'  
data.fillna(data.mean(numeric_only=True), inplace=True)
data['Name of Company'].fillna(missing_value_replacement, inplace=True)
data['Nature of Job'].fillna(missing_value_replacement, inplace=True)
data['Comapny Domain'].fillna(missing_value_replacement, inplace=True)
data['Industry Location'].fillna(missing_value_replacement, inplace=True)
data['Department Alocated'].fillna(missing_value_replacement, inplace=True)
data['Job Role'].fillna(missing_value_replacement, inplace=True)

# Check for missing values
print("\nMissing Values:")
print(data.isnull().sum())

print(data.columns)

# Drop the 'High School Marks' column
data = data.drop(columns=['High School Marks'])
data = data.drop(columns=['Technical Score'])

# Verify the change by printing the first few rows
print(data.columns)


data['Placement Status'] = LabelEncoder().fit_transform(data['Placement Status'])  
print(data['Placement Status'].head())

# Count the occurrences of each placement status
placement_counts = data['Placement Status'].value_counts()

# Print the counts
print(placement_counts)

# Save to a new Excel file if needed
data.to_excel("processed_student_data.xlsx", index=False)