import pandas as pd
import numpy as np
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold

# 1. Load Data
print("Loading data...")
train_df = pd.read_csv('train_updated.csv')
test_df = pd.read_csv('test_updated.csv')
test_ids = pd.read_csv('test_ids.csv')

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING (The Secret Sauce)
# ---------------------------------------------------------
def create_features(df):
    # Avoid division by zero by adding a small number (e.g., +1)
    
    # Ratio of Loan to Income (High ratio = High Risk)
    df['Loan_to_Income'] = df['RequestedSum'] / (df['AnnualEarnings'] + 1)
    
    # Estimated Monthly Installment (EMI)
    df['EMI'] = df['RequestedSum'] / (df['RepayPeriod'] + 1)
    
    # EMI to Monthly Income Ratio
    monthly_income = df['AnnualEarnings'] / 12
    df['EMI_to_Income'] = df['EMI'] / (monthly_income + 1)
    
    # Trust per year of age (Is he trustworthy for his age?)
    df['Trust_per_Year'] = df['TrustMetric'] / (df['ApplicantYears'] + 1)
    
    # Accounts per year of work
    df['Accounts_per_WorkYear'] = df['ActiveAccounts'] / (df['WorkDuration'] + 1)
    
    return df

print("Creating new features...")
train_df = create_features(train_df)
test_df = create_features(test_df)

# ---------------------------------------------------------
# 3. PREPROCESSING (Updated for new features)
# ---------------------------------------------------------
target = train_df['RiskFlag']
train_features = train_df.drop(['RiskFlag', 'ProfileID'], axis=1)
test_features = test_df.drop(['ProfileID'], axis=1)

# Identify columns again because we added new ones
cat_cols = train_features.select_dtypes(include=['object']).columns
num_cols = train_features.select_dtypes(exclude=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
    ],
    verbose_feature_names_out=False
)

print("Preprocessing data...")
X = preprocessor.fit_transform(train_features)
X_test = preprocessor.transform(test_features)
y = target.values

# ---------------------------------------------------------
# 4. TRAIN MODEL 1: XGBOOST (Standard)
# ---------------------------------------------------------
print("\nTraining XGBoost...")
xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)
xgb_model.fit(X, y)
# Get Probabilities (Confidence) instead of just 0/1
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------
# 5. TRAIN MODEL 2: CATBOOST (Standard)
# ---------------------------------------------------------
print("\nTraining CatBoost...")
cb_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='Accuracy',
    verbose=100,
    random_state=42
)
cb_model.fit(X, y)
# Get Probabilities
cb_probs = cb_model.predict_proba(X_test)[:, 1]

# ---------------------------------------------------------
# 6. BLENDING (Weighted Average)
# ---------------------------------------------------------
print("\nBlending predictions...")
# Since CatBoost performed slightly better for you, let's give it slightly more weight
# 80% CatBoost + 20% XGBoost
final_probs = (0.6 * cb_probs) + (0.4 * xgb_probs)

# Apply Threshold (0.5 is standard, but you can tweak this to 0.6 to be safer)
final_predictions = (final_probs >= 0.5).astype(int)

# ---------------------------------------------------------
# 7. SAVE SUBMISSION
# ---------------------------------------------------------
submission = pd.DataFrame({
    'ProfileID': test_ids['ProfileID'],
    'RiskFlag': final_predictions
})

filename = 'submission_advanced_blend.csv'
submission.to_csv(filename, index=False)
print(f"Done! Saved to '{filename}'")
print("Summary of improvements:")
print("1. Added 5 new engineered features (Loan_to_Income, EMI, etc.)")
print("2. Trained both XGBoost and CatBoost on this richer data.")
print("3. Blended their predictions (60% CatBoost, 40% XGBoost) for stability.")