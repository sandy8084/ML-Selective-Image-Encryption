import os
import subprocess
import pandas as pd
import joblib


def get_labels(image_path):

    # -----------------------------
    # Base paths
    # -----------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    matlab_folder = os.path.join(BASE_DIR, "matlab")

    # -----------------------------
    # Ensure results folder exists
    # -----------------------------
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # -----------------------------
    # Output CSV path (UPDATED)
    # -----------------------------
    output_csv = os.path.join(results_dir, "features.csv")

    # -----------------------------
    # Convert paths for MATLAB
    # -----------------------------
    image_path_abs = os.path.abspath(image_path).replace("\\", "/")
    output_csv_abs = output_csv.replace("\\", "/")
    matlab_folder = matlab_folder.replace("\\", "/")

    print("...Running MATLAB script for feature extraction...")

    subprocess.run([
        "matlab",
        "-batch",
        f"cd('{matlab_folder}'); extract_features('{image_path_abs}', '{output_csv_abs}')"
    ])

    # -----------------------------
    # Check output
    # -----------------------------
    if not os.path.exists(output_csv):
        raise Exception("Feature extraction failed. MATLAB output not found.")

    print(" MATLAB feature extraction complete")

    # -----------------------------
    # Load features
    # -----------------------------
    X_new = pd.read_csv(output_csv)

    # -----------------------------
    # Load scaler
    # -----------------------------
    scaler = joblib.load(os.path.join(BASE_DIR, "models", "scaler.pkl"))

    X_new = pd.DataFrame(
        scaler.transform(X_new),
        columns=scaler.feature_names_in_
    )

    # -----------------------------
    # Load model
    # -----------------------------
    svm_model = joblib.load(os.path.join(BASE_DIR, "models", "svm_trained_model.pkl"))

    # -----------------------------
    # Predict
    # -----------------------------
    y_pred = svm_model.predict(X_new)

    # -----------------------------
    # Save predictions (UPDATED)
    # -----------------------------
    pred_path = os.path.join(results_dir, "predicted_labels.csv")

    pd.DataFrame(y_pred, columns=["Encrypt"]).to_csv(pred_path, index=False)

    print(f"Predictions saved at: {pred_path}")

    return y_pred