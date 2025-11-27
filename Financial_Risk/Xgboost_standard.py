import pandas as pd
import xgboost as xgb

# 1. Load Preprocessed Data
print("Loading preprocessed data...")
X = pd.read_csv('train_X_preprocessed.csv')
y = pd.read_csv('train_y_preprocessed.csv')
X_test = pd.read_csv('test_X_preprocessed.csv')
test_ids = pd.read_csv('test_ids.csv')

# 2. Train Standard XGBoost (NO Balancing)
# We removed 'scale_pos_weight' to let the model learn naturally.
print("Training Standard XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=300,       # Increased trees for better learning
    learning_rate=0.05,     # Slower learning rate for higher precision
    max_depth=8,            # Deeper trees to find complex patterns
    subsample=0.8,          # Use 80% of data per tree (prevents overfitting)
    colsample_bytree=0.8,   # Use 80% of features per tree
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
    # CRITICAL: scale_pos_weight IS REMOVED
)

model.fit(X, y)

# 3. Predict on Test Data
print("Generating predictions...")
final_predictions = model.predict(X_test)

# 4. Save Submission
submission = pd.DataFrame({
    'ProfileID': test_ids['ProfileID'],
    'RiskFlag': final_predictions
})

submission.to_csv('submission_xgboost_standard.csv', index=False)
print("Done! Saved to 'submission_xgboost_standard.csv'")