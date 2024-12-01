from tensorflow.keras.utils import load_img, img_to_array, array_to_img
import numpy as np
import os
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image

def load_data(image_dir, mask_dir, image_size=(128, 128)):
    image_files = sorted(os.listdir(image_dir))
    mask_files = sorted(os.listdir(mask_dir))
    
    images = []
    masks = []
    
    for img_file, mask_file in zip(image_files, mask_files):
        if img_file.split('.')[0] == mask_file.split('_segmentation')[0]:
            img = img_to_array(load_img(os.path.join(image_dir, img_file), target_size=image_size)) / 255.0
            mask = img_to_array(load_img(os.path.join(mask_dir, mask_file), target_size=image_size, color_mode="grayscale")) / 255.0
            images.append(img)
            masks.append(mask)
        else:
            print(f"Skipping unmatched pair: {img_file} and {mask_file}")
    
    return np.array(images), np.array(masks)


def plot_metrics(history, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)
    
    if 'dice_coefficient' in history.history and 'val_dice_coefficient' in history.history:
        plt.figure(figsize=(6, 4))
        plt.plot(history.history['dice_coefficient'], label='Train Dice Coefficient')
        plt.plot(history.history['val_dice_coefficient'], label='Val Dice Coefficient')
        plt.title('Dice Coefficient')
        plt.xlabel('Epochs')
        plt.ylabel('Dice Coefficient')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'dice_coefficient.png'))
        plt.close()
    
    if 'mean_iou' in history.history and 'val_mean_iou' in history.history:
        plt.figure(figsize=(6, 4))
        plt.plot(history.history['mean_iou'], label='Train IoU')
        plt.plot(history.history['val_mean_iou'], label='Val IoU')
        plt.title('IoU')
        plt.xlabel('Epochs')
        plt.ylabel('IoU')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'iou.png'))
        plt.close()

    if 'loss' in history.history and 'val_loss' in history.history:
        plt.figure(figsize=(6, 4))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(output_dir, 'loss.png'))
        plt.close()

# Define Dice coefficient
def dice_coefficient(y_true, y_pred, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (tf.keras.backend.sum(y_true_f) + tf.keras.backend.sum(y_pred_f) + smooth)

def mean_iou(y_true, y_pred):
    y_true = tf.keras.backend.flatten(y_true)
    y_pred = tf.keras.backend.flatten(y_pred)
    y_pred = tf.cast(y_pred > 0.5, tf.int32)
    y_true = tf.cast(y_true, tf.int32)

    cm = tf.math.confusion_matrix(y_true, y_pred, num_classes=2)
    intersection = tf.linalg.diag_part(cm)
    union = tf.reduce_sum(cm, axis=0) + tf.reduce_sum(cm, axis=1) - intersection
    iou = intersection / tf.maximum(union, 1)
    return tf.reduce_mean(iou)

def predict_and_overlay(model, image_dir, new_data_dir, image_size=(128, 128)):
    os.makedirs(new_data_dir, exist_ok=True)
    
    image_files = sorted(os.listdir(image_dir))
    
    for img_file in image_files:
        img = img_to_array(load_img(os.path.join(image_dir, img_file), target_size=image_size)) / 255.0
        img_expanded = np.expand_dims(img, axis=0)

        predicted_mask = model.predict(img_expanded)[0]
        binary_mask = (predicted_mask > 0.5).astype(np.uint8)

        overlay = img * binary_mask 

        overlay_img = array_to_img(overlay)
        save_path = os.path.join(new_data_dir, img_file)
        overlay_img.save(save_path)
        print(f"Saved overlayed image: {save_path}")
