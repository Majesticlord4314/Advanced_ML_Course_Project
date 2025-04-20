import numpy as np
from keras.utils import to_categorical

def load_data(dataset_name):
    # Dummy dataset for structure (replace with actual loading logic)
    if dataset_name == 'fer2013':
        # Example: load from numpy or preprocess here
        train_faces = np.random.rand(1000, 48, 48, 1)
        train_labels = np.random.randint(0, 7, 1000)
        val_faces = np.random.rand(200, 48, 48, 1)
        val_labels = np.random.randint(0, 7, 200)

        train_emotions = to_categorical(train_labels, 7)
        val_emotions = to_categorical(val_labels, 7)

        return (train_faces, train_emotions), (val_faces, val_emotions)

    else:
        raise ValueError("Unknown dataset:", dataset_name)
