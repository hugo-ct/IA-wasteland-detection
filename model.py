import tensorflow as tf
import keras
from keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import os

# Configuration des chemins vers les répertoires de données
train_dir = 'C:/data'  # Utilisez des slashs (/) au lieu de backslash (\) pour éviter les erreurs
test_dir = 'C:/data'

# Générateur de données avec augmentation pour l'entraînement
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% des données pour la validation
)

# Générateur pour l'entraînement
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    subset='training'
)

# Générateur pour la validation
validation_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary',
    subset='validation'
)

# Générateur pour les tests
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(256, 256),
    batch_size=32,
    class_mode='binary'
)

# Définition du modèle CNN
model = models.Sequential()

# Première couche convolutive
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(256, 256, 3)))
model.add(layers.MaxPooling2D((2, 2)))

# Deuxième couche convolutive
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Troisième couche convolutive
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Quatrième couche convolutive
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

# Couché aplatie
model.add(layers.Flatten())

# Couches entièrement connectées (denses)
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dropout(0.5))  # Pour éviter le surapprentissage
model.add(layers.Dense(1, activation='sigmoid'))  # Pour une classification binaire

# Afficher le résumé du modèle
model.summary()

# Compilation du modèle
model.compile(
    loss='binary_crossentropy',
    optimizer=optimizers.Adam(learning_rate=1e-4),
    metrics=['accuracy']
)

# Callback pour arrêter l'entraînement si la validation n'améliore pas
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)



# Callback pour sauvegarder les meilleurs poids du modèle
model_checkpoint = ModelCheckpoint(
    filepath='best_model.weights.h5',
    save_weights_only=True,
    save_best_only=True,
    monitor='val_loss',
    mode='min'
)


# Entraîner le modèle
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=50,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    callbacks=[early_stopping, model_checkpoint]
)

# Charger les meilleurs poids du modèle
model.load_weights('best_model.weights.h5')

# Évaluation du modèle sur le jeu de test
test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
print(f'Précision sur le test : {test_acc:.4f}')

# Prédictions sur le jeu de test
test_generator.reset()
predictions = model.predict(test_generator, steps=test_generator.samples // test_generator.batch_size)
predicted_classes = np.where(predictions > 0.5, 1, 0)

# Rapport de classification
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

print(classification_report(true_classes, predicted_classes, target_names=class_labels))

# Matrice de confusion
cm = confusion_matrix(true_classes, predicted_classes)
print(cm)

# Affichage de la matrice de confusion
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matrice de Confusion')
plt.colorbar()
plt.xticks(np.arange(len(class_labels)), class_labels, rotation=45)
plt.yticks(np.arange(len(class_labels)), class_labels)
plt.tight_layout()
plt.ylabel('Vérité terrain')
plt.xlabel('Prédiction')
plt.show()

# Visualisation des courbes de précision et de perte
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs, acc, 'bo', label='Précision d\'entraînement')
plt.plot(epochs, val_acc, 'b', label='Précision de validation')
plt.title('Précision d\'entraînement et de validation')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, loss, 'bo', label='Perte d\'entraînement')
plt.plot(epochs, val_loss, 'b', label='Perte de validation')
plt.title('Perte d\'entraînement et de validation')
plt.legend()

plt.show()

# Sauvegarde finale du modèle
model.save('friche_cnn_final_model.keras')
