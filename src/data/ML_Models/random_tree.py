from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.data.load_data import load_data
from src.data.preprocess import (
    split_data,
    handle_missing_values,
    one_hot_encode_data,
    ordinal_encode_data,
    identify_features,
    standardize_data
)


def create_model():
    model = RandomForestClassifier(
        n_estimators=100,
        max_features="sqrt",
        random_state=42,
        oob_score=True
    )
    return model


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)

    print("\nRandom Forest trained successfully!")

    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print("\nTest Accuracy:", accuracy)

    print("OOB Score:", model.oob_score_)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    return y_pred


def main():

    # --------------------------------------------------
    # 1. Load Dataset
    # --------------------------------------------------

    df = load_data()

    print("Original Dataset Shape:")
    print(df.shape)

    # --------------------------------------------------
    # 2. Split Dataset
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = split_data(
        df,
        target_column="PlacementStatus",
        drop_columns=[
            "StudentID",
            "Salary Package",
            "IsAnomaly"
        ]
    )

    print("\nTraining Shape:")
    print(X_train.shape)

    print("\nTesting Shape:")
    print(X_test.shape)

    # --------------------------------------------------
    # 3. Identify Features
    # --------------------------------------------------

    numerical_features, categorical_features = identify_features(X_train)

    print("\nNumerical Features:")
    print(numerical_features)

    print("\nCategorical Features:")
    print(categorical_features)

    # --------------------------------------------------
    # 4. Define Encoding Features
    # --------------------------------------------------

    one_hot_features = [
        "Gender",
        "City",
        "Stream",
        "Specialisation",
        "Hostel",
        "HistoryOfBacklogs"
    ]

    ordinal_features = [
        "CollegeTier",
        "CGPA_Tier"
    ]

    # --------------------------------------------------
    # 5. Handle Missing Values
    # --------------------------------------------------

    X_train, X_test, imputer = handle_missing_values(
        X_train,
        X_test,
        numerical_features
    )

    print("\nMissing Value Handling Completed")

    print(
        X_train[numerical_features].isnull().sum()
    )

    # --------------------------------------------------
    # 6. Standardization
    # --------------------------------------------------

    X_train, X_test, scaler = standardize_data(
        X_train,
        X_test,
        numerical_features
    )

    print("\nStandardization Completed")

    # --------------------------------------------------
    # 7. One-Hot Encoding
    # --------------------------------------------------

    X_train, X_test, one_hot_encoder = one_hot_encode_data(
        X_train,
        X_test,
        one_hot_features
    )

    print("\nOne-Hot Encoding Completed")

    # --------------------------------------------------
    # 8. Ordinal Encoding
    # --------------------------------------------------

    X_train, X_test, ordinal_encoder = ordinal_encode_data(
        X_train,
        X_test,
        ordinal_features
    )

    print("\nOrdinal Encoding Completed")

    # --------------------------------------------------
    # 9. Final Dataset Shape
    # --------------------------------------------------

    print("\nFinal Training Shape:")
    print(X_train.shape)

    print("\nFinal Testing Shape:")
    print(X_test.shape)

    # --------------------------------------------------
    # 10. Create Model
    # --------------------------------------------------

    model = create_model()

    # --------------------------------------------------
    # 11. Train Model
    # --------------------------------------------------

    model = train_model(
        model,
        X_train,
        y_train
    )

    # --------------------------------------------------
    # 12. Evaluate Model
    # --------------------------------------------------

    y_pred = evaluate_model(
        model,
        X_test,
        y_test
    )

    # --------------------------------------------------
    # 13. Save Preprocessed Data
    # --------------------------------------------------

    train_output = X_train.copy()
    test_output = X_test.copy()

    train_output["PlacementStatus"] = y_train
    test_output["PlacementStatus"] = y_test

    train_output.to_csv(
        r"C:\Users\malli\Downloads\PlacementPredictionSystem-1\PlacementPredictionSystem_malli_updated\Placementpredictionsystem\src\data\preprocessed_test.csv",
        index=False
    )

    test_output.to_csv(
        r"C:\Users\malli\Downloads\PlacementPredictionSystem-1\PlacementPredictionSystem_malli_updated\Placementpredictionsystem\src\data\preprocessed_test.csv",
        index=False
    )

    print("\nPreprocessed files saved successfully.")


if __name__ == "__main__":
    main()