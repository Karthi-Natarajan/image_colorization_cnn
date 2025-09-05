import cv2
import numpy as np
from tensorflow.keras.models import load_model
from model import build_model
import os

IMG_SIZE = 128
NUM_SAMPLES = 5

# ----------------------------
# Load model
# ----------------------------
try:
    model = load_model("colorization_model.h5")
    print("Loaded trained model successfully!")
except:
    print("Trained model not found! Building new model...")
    model = build_model(img_size=IMG_SIZE)

# ----------------------------
# Load grayscale dataset
# ----------------------------
gray_file = "dataset/gray_imgs.npy"
gray_data = np.load(gray_file, mmap_mode='r')  
gray_data_subset = gray_data[:NUM_SAMPLES]

output_folder = "colorized_samples"
os.makedirs(output_folder, exist_ok=True)

# ----------------------------
# Colorize images
# ----------------------------
for i, gray_img in enumerate(gray_data_subset):
    L = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE)) / 100.0
    L_input = L.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    pred_ab = model.predict(L_input)[0]
    pred_ab = np.clip(pred_ab, -1, 1)
    a = pred_ab[:,:,0] * 128
    b = pred_ab[:,:,1] * 128
    
    lab_out = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
    lab_out[:,:,0] = L * 100
    lab_out[:,:,1] = a
    lab_out[:,:,2] = b
    
    colorized = cv2.cvtColor(lab_out.astype(np.uint8), cv2.COLOR_LAB2BGR)
    save_path = os.path.join(output_folder, f"colorized_{i+1}.png")
    cv2.imwrite(save_path, colorized)
    print(f"Saved colorized image: {save_path}")

print(f"All {NUM_SAMPLES} images colorized and saved in '{output_folder}'")
