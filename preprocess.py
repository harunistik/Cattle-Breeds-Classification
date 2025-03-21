import os
import cv2
import numpy as np
from keras.utils import to_categorical

TRAIN_PATH = "data/train"
TEST_PATH = "data/test"
IMG_SIZE = 224

def load_data(dataset_path):
    images = []
    labels = []
    class_names = sorted(os.listdir(dataset_path))

    for label, class_name in enumerate(class_names):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue

        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(label)

    images = np.array(images) / 255.0
    labels = to_categorical(labels, num_classes=len(class_names))

    return images, labels, class_names

(X_train, y_train, class_names_train) = load_data(TRAIN_PATH)
(X_test, y_test, class_names_test) = load_data(TEST_PATH)

if __name__ == "__main__":
    X_train, y_train, class_names_train = load_data(TRAIN_PATH)
    X_test, y_test, class_names_test = load_data(TEST_PATH)

    print(f"Veri seti yüklendi! Eğitim: {len(X_train)} görüntü, Test: {len(X_test)} görüntü")
    print(f"Sınıflar: {class_names_train}")