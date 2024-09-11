import tensorflow as tf
from keras import layers, models, optimizers, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os

# Configuration des chemins vers les répertoires de données
train_dir = 'C:/data/train'
test_dir = 'C:/data/test'

# Paramètres de configuration
BATCH_SIZE = 32
IMG_SIZE = (256, 256)
AUTOTUNE = tf.data.AUTOTUNE

# Chargement des images avec tf.data et augmentation des données
def preprocess_image(image, label):
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0  # Normalisation des pixels entre 0 et 1
    return image, label

# Chargement des datasets avec les noms de classes
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

class_names = train_dataset.class_names

# Prétraitement des données avec des augmentations de données
augmentation_layer = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

def augment(image, label):
    image = augmentation_layer(image)
    return image, label

# Appliquer les augmentations de données au jeu d'entraînement
train_dataset = train_dataset.map(augment, num_parallel_calls=AUTOTUNE)
train_dataset = train_dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
validation_dataset = validation_dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)
test_dataset = test_dataset.map(preprocess_image, num_parallel_calls=AUTOTUNE)

# Précharger les données en mémoire pour des performances optimales
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# Définition du modèle CNN amélioré
model = models.Sequential([
    layers.Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001), input_shape=(256, 256, 3)),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.MaxPooling2D((2, 2)),
    
    layers.Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001)),
    layers.BatchNormalization(),
    layers.LeakyReLU(),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])

# Afficher le résumé du modèle
model.summary()

# Compilation du modèle avec un scheduler de taux d'apprentissage
optimizer = optimizers.Adam(learning_rate=1e-4)
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizer,
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint(
    filepath='best_model.weights.h5',
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)

lr_scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)

# Entraîner le modèle
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=100,
    callbacks=[early_stopping, model_checkpoint, lr_scheduler]
)

# Charger les meilleurs poids du modèle
model.load_weights('best_model.weights.h5')

# Évaluation du modèle sur le jeu de test
test_loss, test_acc = model.evaluate(test_dataset)
print(f'Précision sur le test : {test_acc:.4f}')

# Prédictions sur le jeu de test
predictions = model.predict(test_dataset)
predicted_classes = np.where(predictions > 0.5, 1, 0)

# Récupération des labels réels
true_classes = np.concatenate([y for x, y in test_dataset], axis=0)
true_classes = np.where(true_classes > 0.5, 1, 0)

# Rapport de classification
print(classification_report(true_classes, predicted_classes, target_names=class_names))

# Matrice de confusion
cm = confusion_matrix(true_classes, predicted_classes)
print(cm)

# Affichage de la matrice de confusion
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matrice de Confusion')
plt.colorbar()
plt.xticks(np.arange(len(class_names)), class_names, rotation=45)
plt.yticks(np.arange(len(class_names)), class_names)
plt.tight_layout()
plt.ylabel('Vérité terrain')
plt.xlabel('Prédiction')
plt.show()

# Sauvegarde finale du modèle
model.save('friche_cnn_final_model_complexe.keras')
