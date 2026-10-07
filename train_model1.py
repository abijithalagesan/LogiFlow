import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    balanced_accuracy_score,
    f1_score,
)

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

DATA_FILE = "dataset/route_features.csv"

df = pd.read_csv(DATA_FILE)

print("Dataset shape:", df.shape)

# --------------------------------------------------
# 2. Feature engineering
# --------------------------------------------------

df["date"] = pd.to_datetime(df["date"], errors="coerce")
df["departure_time_utc"] = pd.to_datetime(
    df["departure_time_utc"],
    format="%H:%M:%S",
    errors="coerce"
)

df["day_of_week"] = df["date"].dt.dayofweek
df["departure_hour"] = df["departure_time_utc"].dt.hour

# Features used for the model
feature_columns = [
    "station_code",
    "executor_capacity_cm3",
    "num_stops",
    "num_dropoffs",
    "num_zones",
    "avg_lat",
    "avg_lng",
    "day_of_week",
    "departure_hour",
]

X = df[feature_columns]
y = df["route_score"]

print("\nClasses:")
print(y.value_counts())

# --------------------------------------------------
# 3. Train / test split
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# --------------------------------------------------
# 4. Preprocessing
# --------------------------------------------------

categorical_features = ["station_code"]

numeric_features = [
    "executor_capacity_cm3",
    "num_stops",
    "num_dropoffs",
    "num_zones",
    "avg_lat",
    "avg_lng",
    "day_of_week",
    "departure_hour",
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features,
        ),
        (
            "numeric",
            "passthrough",
            numeric_features,
        ),
    ]
)

# --------------------------------------------------
# 5. Random Forest classifier
# --------------------------------------------------

model = RandomForestClassifier(
    n_estimators=400,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2,
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ]
)

# --------------------------------------------------
# 6. Train
# --------------------------------------------------

print("\nTraining Model 1...")

pipeline.fit(X_train, y_train)

print("Training completed.")

# --------------------------------------------------
# 7. Evaluation
# --------------------------------------------------

y_pred = pipeline.predict(X_test)

balanced_acc = balanced_accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average="macro")

print("\n" + "=" * 60)
print("MODEL 1 - ROUTE QUALITY CLASSIFICATION")
print("=" * 60)

print(f"\nBalanced Accuracy: {balanced_acc:.4f}")
print(f"Macro F1 Score:    {macro_f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        labels=["High", "Medium", "Low"],
        zero_division=0,
    )
)

print("Confusion Matrix:")
print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=["High", "Medium", "Low"],
    )
)

# --------------------------------------------------
# 8. Save trained model
# --------------------------------------------------

os.makedirs("models", exist_ok=True)

model_file = "models/route_quality_model.joblib"

joblib.dump(pipeline, model_file)

print("\nModel saved to:")
print(model_file)