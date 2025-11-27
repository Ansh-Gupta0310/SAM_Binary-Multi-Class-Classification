import pandas as pd
from catboost import CatBoostClassifier

# 1. Load Preprocessed Data
print("Loading preprocessed data...")
X = pd.read_csv('train_X_preprocessed.csv')
y = pd.read_csv('train_y_preprocessed.csv')
X_test = pd.read_csv('test_X_preprocessed.csv')
test_ids = pd.read_csv('test_ids.csv')

# 2. Train Standard CatBoost
print("Training Standard CatBoost...")
cb_model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    loss_function='Logloss',
    eval_metric='Accuracy', # Optimize for pure Accuracy now
    verbose=100,
    random_state=42
    # Removed: auto_class_weights='Balanced'
)

cb_model.fit(X, y)

# 3. Predict and Save
final_predictions = cb_model.predict(X_test)

submission = pd.DataFrame({
    'ProfileID': test_ids['ProfileID'],
    'RiskFlag': final_predictions
})

submission.to_csv('submission_catboost_standard.csv', index=False)
print("Done! Saved to 'submission_catboost_standard.csv'")