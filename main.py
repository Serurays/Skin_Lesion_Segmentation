import os
import matplotlib.pyplot as plt
from unet import unet_model
from utils import load_data, plot_metrics, dice_coefficient, mean_iou, predict_and_overlay
import tensorflow as tf

physical_devices = tf.config.list_physical_devices('GPU')

try:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    pass

# Define paths
data_folder = 'data'
train_images_dir = os.path.join(data_folder, 'train_images')
train_masks_dir = os.path.join(data_folder, 'train_masks')
val_images_dir = os.path.join(data_folder, 'val_images')
val_masks_dir = os.path.join(data_folder, 'val_masks')

# Load train and validation data
image_size = (128, 128)  # Resize images to this size
X_train, Y_train = load_data(train_images_dir, train_masks_dir, image_size)
X_val, Y_val = load_data(val_images_dir, val_masks_dir, image_size)

# Build the model
model = unet_model(input_size=(128, 128, 3))

# Compile model with improved metrics
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=[dice_coefficient, mean_iou]
)

# Train the model
history = model.fit(
    X_train, Y_train,
    validation_data=(X_val, Y_val),
    epochs=30,
    batch_size=16
)

os.makedirs("models", exist_ok=True)
model.save("models/unet.keras")
model.save_weights("models/unet_weights.weights.h5")

print(history.history.keys())

# Plot metrics
plot_metrics(history)

predict_and_overlay(predictor, "data/train_images", "new_data/train")
