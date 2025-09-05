# Image Colorization using CNN

This project implements a **Convolutional Neural Network (CNN)** for automatic image colorization.  
It takes grayscale images as input and predicts their colored versions.

---

## Features
- Deep learning model built using **TensorFlow/Keras**  
- Pre-trained weights supported (`.h5` model files)  
- Flask web app interface for uploading and testing images  
- Example dataset and training script included  
- Sample outputs are provided in the `colorized_samples/` folder  

---

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Karthi-Natarajan/image_colorization_cnn.git
cd image_colorization_cnn

python -m venv venv
source venv/bin/activate    # Linux/Mac
venv\Scripts\activate       # Windows


pip install -r requirements.txt

#To run train.py
python train.py

#To run app.py
python app.py
