import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load datasets
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# 1. Handle Target Variable
# Drop rows in train where target is missing
train = train.dropna(subset=['spend_category'])
y_train = train['spend_category']

# Separate features
X_train = train.drop(['spend_category', 'trip_id'], axis=1)
X_test = test.drop(['trip_id'], axis=1)

# Identify numerical and categorical columns
# We explicitly define them to ensure correct processing types
numeric_features = ['num_females', 'num_males', 'mainland_stay_nights', 'island_stay_nights']
categorical_features = [col for col in X_train.columns if col not in numeric_features]

# Preprocessing Pipeline

# Numeric: Impute with median, then Scale
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical: Impute with 'missing', then OneHotEncode
# handle_unknown='ignore' ensures the model doesn't crash if test data has new categories
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) 
])

# Combine transformers
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    verbose_feature_names_out=False
)

# Fit on train and transform both
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Get feature names
feature_names = preprocessor.get_feature_names_out()

# Convert back to DataFrames
X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

# Add target back to train data for training
X_train_df['spend_category'] = y_train.values

# Save to CSV
X_train_df.to_csv('preprocessed_train.csv', index=False)
X_test_df.to_csv('preprocessed_test.csv', index=False)

print("Preprocessing complete.")