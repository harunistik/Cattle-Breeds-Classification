import tensorflow as tf
import numpy as np
import cv2
from preprocess import class_names_train

model = tf.keras.models.load_model("models/mobilenetv2_model.h5")

image_path = "data/test/Jersey cattle/Jerseycattle122.jpg"

def load_test_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print("Görüntü yüklenemedi, dosya yolunu kontrol et!")
        return None

    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    return img

image = load_test_image(image_path)

if image is not None:
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)

    print(f"Modelin Tahmini: {class_names_train[predicted_class]}")

