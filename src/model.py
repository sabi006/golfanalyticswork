
import pandas as pd
import joblib
import os
#jev laya 
from sklearn.model_selection import GroupShuffleSplit
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Load dataset
df = pd.read_csv("data/golf_data.csv")

# 2. Select features and target
X = df[["club", "accuracy", "wind_speed"]]
y = df["distance"]

# 3. Split data by players
splitter = GroupShuffleSplit(
    n_splits=1,
    test_size=0.2,
    random_state=42
)

train_index, test_index = next(
    splitter.split(X, y, groups=df["player_id"])
)

X_train = X.iloc[train_index]
X_test = X.iloc[test_index]

y_train = y.iloc[train_index]
y_test = y.iloc[test_index]

# 4. Prepare categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ("club", OneHotEncoder(handle_unknown="ignore"), ["club"])
    ],
    remainder="passthrough"
)

# 5. Create ML model
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# 6. Train model
model.fit(X_train, y_train)

# 7. Make predictions
predictions = model.predict(X_test)

# 8. Evaluate model
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
rmse = mse ** 0.5
r2 = r2_score(y_test, predictions)

print("MODEL EVALUATION")
print("-------------------------")
print("MAE:", round(mae, 2))
print("MSE:", round(mse, 2))
print("RMSE:", round(rmse, 2))
print("R2 Score:", round(r2, 4))

# 9. Save trained model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/golf_model.pkl")

print("\nModel saved successfully!")
