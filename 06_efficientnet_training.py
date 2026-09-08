import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load Data
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

train_dataset = tf.keras.utils.image_dataset_from_directory(
    "./data/processed/images/train", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary')
val_dataset = tf.keras.utils.image_dataset_from_directory(
    "./data/processed/images/validation", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary')
test_dataset = tf.keras.utils.image_dataset_from_directory(
    "./data/processed/images/test", image_size=IMG_SIZE, batch_size=BATCH_SIZE, label_mode='binary', shuffle=False)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# 2. Data Augmentation (Stop it from memorizing images)
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2),
])

# 3. Load EfficientNetB0 (Note: EfficientNet handles its own pixel scaling automatically!)
print("\nDownloading EfficientNetB0 backbone...")
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model for Phase 1
base_model.trainable = False

# 4. Build the Model
inputs = layers.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

# ---------------------------------------------------------
# PHASE 1: Warm-up Training (Train only the top layer)
# ---------------------------------------------------------
print("\n--- PHASE 1: Warming up the new classifier head ---")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_dataset, validation_data=val_dataset, epochs=3)

# ---------------------------------------------------------
# PHASE 2: Deep Fine-Tuning (Unfreeze the top layers)
# ---------------------------------------------------------
print("\n--- PHASE 2: Deep Fine-Tuning the top layers ---")
# Unfreeze the base model
base_model.trainable = True

# Freeze everything EXCEPT the last 20 layers so we don't wreck the pre-trained weights
for layer in base_model.layers[:-20]:
    layer.trainable = False

# Recompile with a much smaller learning rate so it makes tiny, careful adjustments
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.fit(train_dataset, validation_data=val_dataset, epochs=4)

# 5. Final Evaluation
print("\nEvaluating EfficientNetB0 on unseen test data...")
test_loss, test_acc = model.evaluate(test_dataset)
print(f"\n✅ EfficientNet Final Test Accuracy: {test_acc * 100:.2f}%")

model.save("./data/processed/coral_efficientnet_model.keras")
print("Model saved to ./data/processed/coral_efficientnet_model.keras")