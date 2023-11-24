import os
import cv2
import numpy as np
import tensorflow as tf
from keras.utils import CustomObjectScope
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from unet import build_unet  # Assuming 'unet' contains the model architecture
from metrics import dice_loss, dice_coef  # Assuming these metrics are defined

# Define global parameters (you can customize these)
H = 512
W = 512

app = Flask(__name__)


def load_and_process_image(image_path):
    # Load the model
    with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
        model = tf.keras.models.load_model(r"D:\u-nnet\u-nnet\new_model.h5")  # Change to your model file path

    # Load and preprocess the image
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (W, H))
    x = image / 255.0
    x = np.expand_dims(x, axis=0)

    # Make a prediction
    y_pred = model.predict(x, verbose=0)[0]
    y_pred = np.squeeze(y_pred, axis=-1)
    y_pred = (y_pred >= 0.5).astype(np.uint8)  # Apply a threshold to obtain a binary mask

    return image, y_pred


@app.route('/upload_folder', methods=['POST'])
def upload_folder():
    if 'folder' not in request.files:
        return jsonify({'error': 'No folder part'})

    folder = request.files.getlist('folder')
    folder_path = secure_filename(folder[0].filename)
    os.makedirs(folder_path, exist_ok=True)

    if folder:
        for file in folder:
            filename = secure_filename(file.filename)
            file_path = os.path.join(folder_path, filename)
            file.save(file_path)

            image, y_pred = load_and_process_image(file_path)

            # Save the result
            y_pred_image = np.stack((y_pred, y_pred, y_pred), axis=-1) * 255
            cat_images = np.concatenate([image, np.ones((H, 10, 3)) * 255, y_pred_image], axis=1)

            # Save the processed image temporarily (you may want to adjust this based on your needs)
            save_path = os.path.join(folder_path, f"processed_{filename}")
            cv2.imwrite(save_path, cat_images)

        return jsonify({'result_folder': folder_path})  # Return the folder path
    return jsonify({'error': 'Upload failed'})


if __name__ == "__main__":
    app.run(debug=True)  # Run the Flask app
