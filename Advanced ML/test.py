import os
import numpy as np
from keras.models import load_model
from utils import load_data

# Configuration
model_path = './models/best_model.h5'  # Replace with correct file name
dataset_name = 'fer2013'

# Load test data
_, val_data = load_data(dataset_name)
val_faces, val_emotions = val_data

# Load model
model = load_model(model_path)

# Evaluate
loss, accuracy = model.evaluate(val_faces, val_emotions, verbose=1)
print(f'\nValidation Loss: {loss:.4f}')
print(f'Validation Accuracy: {accuracy:.4f}')

# Optional: predict and print some examples
predictions = model.predict(val_faces[:5])
print("Predictions:", np.argmax(predictions, axis=1))
print("Ground Truth:", np.argmax(val_emotions[:5], axis=1))
