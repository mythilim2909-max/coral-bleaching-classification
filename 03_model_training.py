import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# 1. Point the AI to your cleaned image folders
train_dir = "./data/processed/images/train"
val_dir = "./data/processed/images/validation"

print("Loading training data...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary' # Because we only have 2 options: healthy or bleached
)

print("Loading validation data...")
val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary'
)

# 2. Build the Convolutional Neural Network (The AI Brain)
print("\nBuilding the AI Model...")
model = models.Sequential([
    # Rescale pixel values from 0-255 down to 0-1 so the math is easier for the AI
    layers.Rescaling(1./255, input_shape=(224, 224, 3)),
    
    # "Eyes" of the AI: Look for basic patterns (edges, colors)
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)), # Shrink the image to focus on important parts
    
    # Look for complex patterns (shapes, textures)
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    
    # Flatten the 2D image data into a 1D list of numbers
    layers.Flatten(),
    
    # The decision-making layer
    layers.Dense(64, activation='relu'),
    
    # Final output layer: 1 neuron that spits out a probability (0 = healthy, 1 = bleached)
    layers.Dense(1, activation='sigmoid')
])

# 3. Compile the model (give it a learning strategy)
model.compile(
    optimizer='adam',
    loss='binary_crossentropy', # Used for Yes/No, A/B classification
    metrics=['accuracy']
)

# 4. Train the model (Show it the flashcards)
print("\nStarting Training! This will take a few minutes...")
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=5 # It will look through the entire dataset 5 times
)

# 5. Save the trained brain so we don't have to train it again later
model.save("./data/processed/coral_baseline_model.keras")
print("\n✅ Training Complete! Model saved to ./data/processed/coral_baseline_model.keras")