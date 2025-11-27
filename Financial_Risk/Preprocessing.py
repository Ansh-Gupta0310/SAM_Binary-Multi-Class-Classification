import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Load the Datasets
train_df = pd.read_csv('train_updated.csv')
test_df = pd.read_csv('test_updated.csv')
sample_sub = pd.read_csv('sample_submission_updated.csv')

# 2. Separate Features and Target
# Drop ProfileID as it is an identifier, not a feature
X = train_df.drop(['RiskFlag', 'ProfileID'], axis=1)
y = train_df['RiskFlag']
X_test = test_df.drop(['ProfileID'], axis=1)

# 3. Define Mappings for Categorical Features
# Ordinal Mapping for Qualification
qual_map = {
    'High School': 0, 
    "Bachelor's": 1, 
    "Master's": 2, 
    'PhD': 3
}

# Binary Mapping for Yes/No columns
binary_map = {'Yes': 1, 'No': 0}

# 4. Apply Mappings
def apply_mappings(df):
    df = df.copy()
    df['QualificationLevel'] = df['QualificationLevel'].map(qual_map)
    
    # Binary columns
    binary_cols = ['OwnsProperty', 'FamilyObligation', 'JointApplicant']
    for col in binary_cols:
        df[col] = df[col].map(binary_map)
        
    return df

X = apply_mappings(X)
X_test = apply_mappings(X_test)

# 5. One-Hot Encoding for Nominal Categorical Features
# (WorkCategory, RelationshipStatus, FundUseCase)
X = pd.get_dummies(X, columns=['WorkCategory', 'RelationshipStatus', 'FundUseCase'])
X_test = pd.get_dummies(X_test, columns=['WorkCategory', 'RelationshipStatus', 'FundUseCase'])

# CRITICAL STEP: Align columns between train and test
# This ensures that if a category (e.g., 'FundUseCase_Other') is missing in one set, 
# the model structure remains consistent.
X, X_test = X.align(X_test, join='left', axis=1)
X_test = X_test.fillna(0) # Fill missing columns (from alignment) with 0

# 6. Scaling (Standardization)
# Essential for SVM and Neural Networks
scaler = StandardScaler()

# Fit on training data ONLY, then transform both
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for readability (optional, but helpful)
X_final = pd.DataFrame(X_scaled, columns=X.columns)
X_test_final = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# 7. Split Training Data for Validation
# Since we have an imbalanced dataset, stratify=y ensures the split 
# preserves the % of class 1 in both train and validation sets.
X_train, X_val, y_train, y_val = train_test_split(
    X_final, y, test_size=0.2, random_state=42, stratify=y
)

print("Preprocessing Complete.")
print(f"Training set shape: {X_train.shape}")
print(f"Validation set shape: {X_val.shape}")
print(f"Test set shape: {X_test_final.shape}")

# ... (Your previous preprocessing code) ...

# 8. Save the processed datasets to CSV files
# index=False ensures we don't save the row numbers as a separate column
print("Saving processed datasets...")

X_train.to_csv('X_train_processed.csv', index=False)
y_train.to_csv('y_train_processed.csv', index=False)

X_val.to_csv('X_val_processed.csv', index=False)
y_val.to_csv('y_val_processed.csv', index=False)

# Save the processed test set
X_test_final.to_csv('test_processed.csv', index=False)

# Optional: Save the Test ProfileIDs separately if you haven't already
# (Useful for creating the final submission file later)
test_df[['ProfileID']].to_csv('test_ids.csv', index=False)

print("Files saved successfully.")
# Now you can proceed to train your models:
# 1. Logistic Regression
# 2. SVM
# 3. Neural Network