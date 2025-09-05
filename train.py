import numpy as np
import cv2
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from model import build_model
from tensorflow.keras.utils import Sequence
import os

IMG_SIZE = 128
BATCH_SIZE = 16
EPOCHS = 50
LIMIT = 500  # subset for quick testing

gray_file = "dataset/gray_imgs.npy"
lab_file = "dataset/lab_imgs.npy"

# ---------------------------
# Data generator
# ---------------------------
class NPYDataGenerator(Sequence):
    def __init__(self, gray_file, lab_file, batch_size=16, img_size=128, limit=None):
        self.gray_data = np.load(gray_file, mmap_mode='r')
        self.lab_data = np.load(lab_file, mmap_mode='r')
        if limit is not None:
            self.gray_data = self.gray_data[:limit]
            self.lab_data = self.lab_data[:limit]
        self.batch_size = batch_size
        self.img_size = img_size
        self.indices = np.arange(len(self.gray_data))

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        X_batch = []
        y_batch = []
        for i in batch_indices:
            gray_img = cv2.resize(self.gray_data[i], (self.img_size, self.img_size)) / 100.0  # L channel
            lab_img = cv2.resize(self.lab_data[i], (self.img_size, self.img_size))
            a = lab_img[:,:,1] / 128.0
            b = lab_img[:,:,2] / 128.0
            X_batch.append(gray_img.reshape(self.img_size, self.img_size, 1))
            y_batch.append(np.stack([a,b], axis=-1))
        return np.array(X_batch, dtype=np.float32), np.array(y_batch, dtype=np.float32)

dataset = NPYDataGenerator(gray_file, lab_file, batch_size=BATCH_SIZE, img_size=IMG_SIZE, limit=LIMIT)

# ---------------------------
# Build model
# ---------------------------
model = build_model(img_size=IMG_SIZE)

# ---------------------------
# Checkpoint
# ---------------------------
checkpoint = ModelCheckpoint("colorization_model.h5", monitor='loss', verbose=1, save_best_only=True, mode='min')

# ---------------------------
# Callback to save sample outputs after each epoch
# ---------------------------
class SampleOutputCallback(Callback):
    def __init__(self, gray_data, output_folder="sample_outputs", num_samples=5):
        super().__init__()
        self.gray_data = gray_data[:num_samples]
        self.output_folder = output_folder
        self.num_samples = num_samples
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

    def on_epoch_end(self, epoch, logs=None):
        for i, gray_img in enumerate(self.gray_data):
            L = cv2.resize(gray_img, (IMG_SIZE, IMG_SIZE)) / 100.0
            L_input = L.reshape(1, IMG_SIZE, IMG_SIZE, 1)

            pred_ab = self.model.predict(L_input, verbose=0)[0]
            pred_ab = np.clip(pred_ab, -1, 1)
            a = pred_ab[:,:,0] * 128
            b = pred_ab[:,:,1] * 128

            lab_out = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32)
            lab_out[:,:,0] = L * 100
            lab_out[:,:,1] = a
            lab_out[:,:,2] = b

            colorized = cv2.cvtColor(lab_out.astype(np.uint8), cv2.COLOR_LAB2BGR)
            save_path = os.path.join(self.output_folder, f"epoch{epoch+1}_sample{i+1}.png")
            cv2.imwrite(save_path, colorized)

        print(f"Saved sample outputs for epoch {epoch+1} in '{self.output_folder}'")

sample_callback = SampleOutputCallback(gray_data=np.load(gray_file, mmap_mode='r'), num_samples=5)

# ---------------------------
# Train
# ---------------------------
model.fit(dataset, epochs=EPOCHS, verbose=1, callbacks=[checkpoint, sample_callback])
print("Training done. Model saved as colorization_model.h5")
