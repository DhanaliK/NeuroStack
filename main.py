import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions

from tensorflow.keras.preprocessing import image as keras_image
from tensorflow.keras.models import Model

IMG_WIDTH, IMG_HEIGHT = 224, 224
LAST_CONV_LAYER_NAME = "conv5_block3_out"
CLASSIFIER_LAYER_NAME = "predictions"

def load_and_preprocess_image(img_path):
    img = keras_image.load_img(img_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    img_array = keras_image.img_to_array(img)
    return preprocess_input(np.expand_dims(img_array, axis=0)), img_array

def create_dummy_mask(image_array):
    hsv = cv2.cvtColor(image_array.astype(np.uint8), cv2.COLOR_RGB2HSV)
    lower = np.array([10, 40, 40])
    upper = np.array([30, 255, 255])
    lesion_mask = cv2.inRange(hsv, lower, upper)
    return (lesion_mask > 0).astype(np.uint8)

def load_or_create_mask(mask_path, img_array):
    if os.path.exists(mask_path):
        mask = keras_image.load_img(mask_path, target_size=(IMG_WIDTH, IMG_HEIGHT), color_mode='grayscale')
        mask_array = keras_image.img_to_array(mask) / 255.0
        return (mask_array > 0.5).astype(np.uint8).squeeze()
    else:
        print(f"Mask not found, generating dummy mask: {mask_path}")
        return create_dummy_mask(img_array)

def make_gradcam_heatmap(img_array_preprocessed, model):
    grad_model = Model(inputs=[model.inputs],
                       outputs=[model.get_layer(LAST_CONV_LAYER_NAME).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array_preprocessed)
        pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]
    grads = tape.gradient(class_channel, conv_outputs)[0]
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1))
    heatmap = tf.reduce_sum(tf.multiply(pooled_grads, conv_outputs[0]), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / np.max(heatmap + 1e-10)
    return cv2.resize(heatmap, (IMG_WIDTH, IMG_HEIGHT)), int(pred_index.numpy())


def binarize_heatmap(heatmap, threshold=0.5):
    return (heatmap > threshold).astype(np.uint8)

def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    return np.sum(intersection) / np.sum(union)

def visualize(image, mask, heatmap, bin_heatmap, iou, save_path=None):
    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(image / 255.0); axs[0].set_title("Original"); axs[0].axis('off')
    axs[1].imshow(mask, cmap='gray'); axs[1].set_title("Lesion Mask"); axs[1].axis('off')
    axs[2].imshow(image / 255.0); axs[2].imshow(heatmap, cmap='jet', alpha=0.5); axs[2].set_title("Grad-CAM"); axs[2].axis('off')
    axs[3].imshow(bin_heatmap, cmap='gray'); axs[3].set_title(f"Binarized\nIoU={iou:.2f}"); axs[3].axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

def process_images(image_dir, mask_dir, output_dir):
    model = ResNet50(weights='imagenet')
    results = []
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(image_dir):
        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname.replace(".jpg", "_mask.png"))
        if not fname.lower().endswith(('.jpg', '.png')):
            continue

        img_pre, img_raw = load_and_preprocess_image(img_path)
        true_mask = load_or_create_mask(mask_path, img_raw)
        heatmap, pred_idx = make_gradcam_heatmap(img_pre, model)
        bin_heat = binarize_heatmap(heatmap)
        iou = calculate_iou(bin_heat, true_mask)

        results.append({"filename": fname, "predicted_class": decode_predictions(model.predict(img_pre), top=1)[0][0][1], "IoU": round(iou, 4)})
        visualize(img_raw, true_mask, heatmap, bin_heat, iou, save_path=os.path.join(output_dir, f"{fname}_viz.png"))
        print(f"{fname} => IoU: {iou:.3f}")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(output_dir, "iou_scores.csv"), index=False)
    print("\n✅ Results saved to:", output_dir)

if __name__ == "__main__":
    image_dir = "./sample_dataset/images"
    mask_dir = "./sample_dataset/masks"
    output_dir = "./results/visuals"
    process_images(image_dir, mask_dir, output_dir)
