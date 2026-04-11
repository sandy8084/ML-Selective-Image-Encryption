import cv2
import numpy as np
import pandas as pd
import os
def encrypt_image(image_path):

    print("...Starting Encryption Pipeline...")
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(BASE_DIR, "results")
    labels_path = os.path.join(results_dir, "predicted_labels.csv")
    labels = pd.read_csv(labels_path)["Encrypt"].values
    # -----------------------------
    # Paths
    # -----------------------------
    henon_x_file = os.path.join(BASE_DIR, "keys", "henon_x.txt")
    henon_y_file = os.path.join(BASE_DIR, "keys", "henon_y.txt")

    output_encrypted_image = os.path.join(results_dir, "s1_encrypted_image.tif")
    output_confused_image = os.path.join(results_dir, "s2_encrypted_image.tif")
    output_final_dna_encrypted = os.path.join(results_dir, "final_encrypted_image.tif")

    # -----------------------------
    # Load Image
    # -----------------------------
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Error: Unable to load {image_path}")

    image_size = image.shape
    block_size = 8

    # -----------------------------
    # Load Henon_x
    # -----------------------------
    with open(henon_x_file, "r") as f:
        henon_x = [int(line.strip()) for line in f]

    # -----------------------------
    # Stage 1: XOR Diffusion
    # -----------------------------
    encrypted_image = np.copy(image)
    key_index = 0
    block_idx = 0

    for i in range(0, image_size[0], block_size):
        for j in range(0, image_size[1], block_size):

            if labels[block_idx] == 1:
                block = encrypted_image[i:i+block_size, j:j+block_size].flatten()
                block_key = np.array(henon_x[key_index:key_index+64], dtype=np.uint8)

                encrypted_block = np.bitwise_xor(block, block_key)

                encrypted_image[i:i+block_size, j:j+block_size] = encrypted_block.reshape((block_size, block_size))

                key_index += 64

            block_idx += 1

    cv2.imwrite(output_encrypted_image, encrypted_image)
    print("***Stage 1 Done***")

    # -----------------------------
    # Stage 2: Confusion
    # -----------------------------
    with open(henon_y_file, "r") as f:
        henon_y = [int(line.strip()) for line in f]

    image_flat = encrypted_image.flatten()

    permutation_order = np.argsort(henon_y[:len(image_flat)])
    confused_flat = image_flat[permutation_order]
    confused_image = confused_flat.reshape(image_size)

    cv2.imwrite(output_confused_image, confused_image)
    print("*** Stage 2 Done***")

    # -----------------------------
    # Stage 3: DNA Encryption
    # -----------------------------
    image = confused_image
    rows, cols = image.shape

    list1, list2 = [], []

    for i in range(rows):
        for j in range(cols):
            pixel = image[i][j]
            b4 = (pixel >> 4) & 1
            b5 = (pixel >> 5) & 1
            b6 = (pixel >> 6) & 1
            b7 = (pixel >> 7) & 1

            list1.append(str(b4) + str(b5))
            list2.append(str(b6) + str(b7))

    # CLS chaotic key
    u = 0.4678
    x = [0.1]

    for j in range(rows * cols):
        temp = (u * x[-1] * (1 - x[-1])) + ((4 - u) * (np.sin(np.pi * x[-1])) / 4)
        temp %= 1
        temp *= 1e9
        temp %= 255
        x.append(temp)

    key_bin_list = []

    for num in x[1:]:
        bin_str = bin(int(num))[2:].zfill(8)
        key_bin_list.append(bin_str[:4])

    final_binary_result = []

    for i in range(len(list1)):
        data_bin = list1[i] + list2[i]
        key_bin = key_bin_list[i]

        a1 = int(data_bin[:2], 2)
        a2 = int(data_bin[2:], 2)
        b1 = int(key_bin[:2], 2)
        b2 = int(key_bin[2:], 2)

        r1 = (a1 + b1) % 4
        r2 = (a2 + b2) % 4

        encrypted_bin = format(r1, '02b') + format(r2, '02b')
        final_binary_result.append(encrypted_bin)

    new_image = np.zeros((rows, cols), dtype=np.uint8)

    index = 0
    for i in range(rows):
        for j in range(cols):
            original_pixel = image[i, j]

            b0 = original_pixel & 1
            b1 = original_pixel & 2
            b2 = original_pixel & 4
            b3 = original_pixel & 8

            mod_bits = final_binary_result[index]
            index += 1

            b4 = int(mod_bits[0]) << 4
            b5 = int(mod_bits[1]) << 5
            b6 = int(mod_bits[2]) << 6
            b7 = int(mod_bits[3]) << 7

            new_pixel = b0 | b1 | b2 | b3 | b4 | b5 | b6 | b7
            new_image[i, j] = new_pixel

    cv2.imwrite(output_final_dna_encrypted, new_image)

    print("...Encryption Completed Successfully...")

    return new_image