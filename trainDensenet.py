from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from preprocess import load_data, TRAIN_PATH, TEST_PATH
import matplotlib.pyplot as plt

X_train, y_train, class_names_train = load_data(TRAIN_PATH)
X_test, y_test, class_names_test = load_data(TEST_PATH)

base_model = DenseNet121(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation="relu")(x)
x = Dense(len(class_names_train), activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=x)

model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                    epochs=15, batch_size=32)

model.save("models/densenet121_model.h5")

def plot_loss(history):
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp (Loss)')
    plt.title('DenseNet121 - Eğitim ve Doğrulama Kaybı')
    plt.legend()
    plt.show()

def plot_accuracy(history):
    plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk (Accuracy)')
    plt.title('DenseNet121 - Eğitim ve Doğrulama Doğruluğu')
    plt.legend()
    plt.show()

plot_loss(history)
plot_accuracy(history)

plt.show(block= True)