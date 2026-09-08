import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Load the processed dataset splits
train_dir = "./data/processed/images/train"
val_dir = "./data/processed/images/validation"
test_dir = "./data/processed/images/test"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print("Loading data splits...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary',
    shuffle=False
)

# Optimize memory loading during training
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

# 2. Load the Pre-Trained MobileNetV2 Backbone
print("\nDownloading MobileNetV2 backbone...")
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,     # Exclude the original ImageNet classifier head
    weights='imagenet'
)

# Freeze the base backbone so its existing visual weights aren't destroyed
base_model.trainable = False

# 3. Build the Coral Classification Head
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

inputs = layers.Input(shape=(224, 224, 3))
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x) # Convert feature map into a 1D vector
x = layers.Dropout(0.2)(x)            # Prevent overfitting
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

# 4. Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 5. Train the Classification Head
print("\nTraining the new classifier on coral data...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5
)

# 6. Evaluate on Unseen Test Data
print("\nEvaluating MobileNetV2 on test data...")
test_loss, test_acc = model.evaluate(test_dataset)
print(f"\n✅ MobileNetV2 Test Accuracy: {test_acc * 100:.2f}%")

# 7. Save Model
model.save("./data/processed/coral_mobilenetv2_model.keras")
print("Model saved to ./data/processed/coral_mobilenetv2_model.keras")