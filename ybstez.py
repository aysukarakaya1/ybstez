import cv2
import numpy as np
import os
import random
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model


def load_images(folder, label, img_size=(240, 240), sample_size=None):
    images, labels = [], []
    all_files = os.listdir(folder)
    if sample_size:
        random.shuffle(all_files)
        all_files = all_files[:sample_size]
    for filename in all_files:
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            images.append(img)
            labels.append(label)
    return images, labels

def preprocess_images(images):
    return np.array(images) / 255.0

folders = {
    'Early_Renaissance': ('/kaggle/input/wikiart30k/Early_Renaissance', 0),
    'Minimalism': ('/kaggle/input/wikiart30k/Minimalism', 1),
    'Pop_Art': ('/kaggle/input/wikiart30k/Pop_Art', 2),
    'Realism': ('/kaggle/input/wikiart30k/Realism', 3),
    'Color_Field_Painting': ('/kaggle/input/wikiart30k/Color_Field_Painting', 4),
    'Baroque': ('/kaggle/input/wikiart30k/Baroque', 5)
}

images, labels = [], []
for style, (path, label) in folders.items():
    img_list, lbl_list = load_images(path, label, sample_size=510 if style == 'Realism' else None)
    processed_imgs = preprocess_images(img_list)
    images.append(processed_imgs)
    labels.extend(lbl_list)

images, labels = [], []
for style, (path, label) in folders.items():
    img_list, lbl_list = load_images(path, label, sample_size=510 if style == 'Baroque' else None)
    processed_imgs = preprocess_images(img_list)
    images.append(processed_imgs)
    labels.extend(lbl_list)

images = np.concatenate(images)
labels = to_categorical(labels, num_classes=6)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

datagen = ImageDataGenerator(rotation_range=15, width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
datagen.fit(X_train)

model = Sequential([
    Input(shape=(240, 240, 3)),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'), Dropout(0.3),
    Dense(6, activation='softmax')
])


model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

model.fit(datagen.flow(X_train, y_train, batch_size=64),
          epochs=50, validation_data=(X_test, y_test), callbacks=[early_stopping])

test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Doğruluğu: {test_acc * 100:.2f}%")

def predict_art_movement(image_path, model, label_map, img_size=(240, 240)):
    img = cv2.imread(image_path)
    if img is not None:
        img = cv2.resize(img, img_size)
        processed_img = np.expand_dims(img / 255.0, axis=0)
        prediction = model.predict(processed_img)
        return label_map[np.argmax(prediction)]
    return "Görsel yüklenemedi"

label_map = {i: name for name, (_, i) in folders.items()}
image_path = '/kaggle/input/wikiart30k/Color_Field_Painting/anne-appleby_jasmine-2000.jpg'
print(f"Sanat Akımı: {predict_art_movement(image_path, model, label_map)}")