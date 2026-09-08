import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# 1. Load the unseen Test Data
test_dir = "./data/processed/images/test"

print("Loading test data...")
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    image_size=(224, 224),
    batch_size=32,
    label_mode='binary',
    shuffle=False # Keep in order so we can grade the answers perfectly
)

# 2. Load your newly trained AI Brain
print("Waking up the trained model...")
model = tf.keras.models.load_model("./data/processed/coral_baseline_model.keras")

# 3. Give it the Final Exam
print("\nTaking the final exam on the test data...")
loss, accuracy = model.evaluate(test_dataset)
print(f"\n✅ Final Real-World Accuracy: {accuracy * 100:.2f}%")

# 4. Generate the Report Card (Confusion Matrix)
print("\nGrading the answers...")
predictions = model.predict(test_dataset)
# Convert percentage confidence into a hard 0 (Healthy) or 1 (Bleached)
predicted_labels = (predictions > 0.5).astype(int).flatten()

# Get the true answers from the answer key
true_labels = []
for images, labels in test_dataset.unbatch():
    true_labels.append(labels.numpy()[0])
true_labels = np.array(true_labels)

# 5. Draw the chart
cm = confusion_matrix(true_labels, predicted_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Healthy (0)', 'Bleached (1)'], 
            yticklabels=['Healthy (0)', 'Bleached (1)'])
plt.title('AI Report Card: What did it get wrong?')
plt.xlabel('What the AI Guessed')
plt.ylabel('The Actual Truth')
plt.show()