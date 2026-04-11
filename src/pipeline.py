from feature_pipeline import get_labels
from encryption import encrypt_image
from decryption import decrypt_image
import os


def run_pipeline(image_path):

    print("Starting Full Pipeline...")

    # -----------------------------
    # Base path
    # -----------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # -----------------------------
    # STEP 1: Get labels (ML)
    # -----------------------------
    get_labels(image_path)

    # -----------------------------
    # STEP 2: Encrypt image
    # -----------------------------
    encrypted_image = encrypt_image(image_path)

    # -----------------------------
    # STEP 3: Decrypt image
    # -----------------------------
    decrypted_image = decrypt_image(encrypted_image)

    print("Pipeline completed successfully")


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    image_path = os.path.join(BASE_DIR, "Test_images", "16_Walter.tiff")

    run_pipeline(image_path)