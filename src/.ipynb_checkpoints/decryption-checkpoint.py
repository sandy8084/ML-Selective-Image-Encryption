import os
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt


def decrypt_image(encrypted_image):

    print("Starting Decryption Pipeline...")

    # -----------------------------
    # Base paths
    # -----------------------------
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(BASE_DIR, "results")
    keys_dir = os.path.join(BASE_DIR, "keys")

    os.makedirs(results_dir, exist_ok=True)

    # -----------------------------
    # File paths
    # -----------------------------
    henon_x_file = os.path.join(keys_dir, "henon_x.txt")
    henon_y_file = os.path.join(keys_dir, "henon_y.txt")
    labels_file = os.path.join(results_dir, "predicted_labels.csv")

    output_dna = os.path.join(results_dir, "s1_dna_decrypted_image.tif")
    output_deconfused = os.path.join(results_dir, "s2_deconfused_image.tif")
    output_final = os.path.join(results_dir, "final_decrypted_image.tif")

    # -----------------------------
    # Load image
    # -----------------------------
    image = encrypted_image
    rows, cols = image.shape
    image_size = image.shape

    # ====================================================
    # Stage 1: DNA Decryption
    # ====================================================
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

    u = 0.4678
    x = [0.1]

    for _ in range(rows * cols):
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

        r1 = (a1 - b1) % 4
        r2 = (a2 - b2) % 4

        decrypted_bin = format(r1, '02b') + format(r2, '02b')
        final_binary_result.append(decrypted_bin)

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

    cv2.imwrite(output_dna, new_image)
    print("**stage 1 decryption done**")
    # ====================================================
    # Stage 2: Deconfusion
    # ====================================================
    with open(henon_y_file, "r") as f:
        henon_y = [int(line.strip()) for line in f]

    image_flat = new_image.flatten()

    permutation_order = np.argsort(henon_y[:len(image_flat)])
    inverse_permutation = np.argsort(permutation_order)

    deconfused_flat = image_flat[inverse_permutation]
    deconfused_image = deconfused_flat.reshape(image_size)

    cv2.imwrite(output_deconfused, deconfused_image)
    print("** stage 2 decryption done**")
    # ====================================================
    # Stage 3: XOR Reverse
    # ====================================================
    labels = pd.read_csv(labels_file)["Encrypt"].values

    with open(henon_x_file, "r") as f:
        henon_x = [int(line.strip()) for line in f]

    decrypted_image = np.copy(deconfused_image)

    block_size = 8
    key_index = 0
    block_idx = 0

    for i in range(0, image_size[0], block_size):
        for j in range(0, image_size[1], block_size):

            if labels[block_idx] == 1:
                block = decrypted_image[i:i+block_size, j:j+block_size].flatten()
                block_key = np.array(henon_x[key_index:key_index+64], dtype=np.uint8)

                decrypted_block = np.bitwise_xor(block, block_key)

                decrypted_image[i:i+block_size, j:j+block_size] = decrypted_block.reshape((block_size, block_size))

                key_index += 64

            block_idx += 1

    cv2.imwrite(output_final, decrypted_image)

    print("Decryption completed")

    return decrypted_image