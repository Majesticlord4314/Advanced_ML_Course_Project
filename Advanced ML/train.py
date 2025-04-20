import os
from keras.callbacks import CSVLogger, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from model import build_model  # your model architecture in model.py
from utils import load_data  # your data loader
import tensorflow as tf

# Hyperparameters
dataset_name = 'fer2013'
num_epochs = 30
batch_size = 64
patience = 10
base_path = './'
trained_models_path = os.path.join(base_path, 'models/')
os.makedirs(trained_models_path, exist_ok=True)

# Data preparation
train_data, val_data = load_data(dataset_name)
train_faces, train_emotions = train_data

# Model
model = build_model()  # returns compiled model

# Callbacks
log_file_path = os.path.join(base_path, dataset_name + '_training.log')
csv_logger = CSVLogger(log_file_path, append=False)
early_stop = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=int(patience / 4), verbose=1)
checkpoint_path = os.path.join(trained_models_path, '{epoch:02d}-{val_loss:.2f}.h5')
model_checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_loss', save_best_only=True, verbose=1)

callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

# Data augmentation
data_generator = ImageDataGenerator(horizontal_flip=True)
train_generator = data_generator.flow(train_faces, train_emotions, batch_size=batch_size)

# Training
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_faces) // batch_size,
    epochs=num_epochs,
    validation_data=val_data,
    callbacks=callbacks,
    verbose=1
)
