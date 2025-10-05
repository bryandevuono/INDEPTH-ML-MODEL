from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow.keras.models import load_model
import os

MODEL_PATH = "../skin_condition_model.h5"

if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model = load_model(MODEL_PATH)
    
# Create test generator (no augmentation, just rescale)
test_datagen = ImageDataGenerator(rescale=1./255)

# Load only the Ehlers-Danlos class
ed_test_dataset = test_datagen.flow_from_directory(
    "../testdata",                # parent folder
    target_size=(224, 224),
    batch_size=32,
    class_mode="sparse",
    shuffle=False
)

# Predictions
y_pred = model.predict(ed_test_dataset)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = ed_test_dataset.classes

# Accuracy (since only ED images are present, accuracy = correct predictions / total)
ed_accuracy = accuracy_score(y_true, y_pred_classes)
print(f"Ehlers-Danlos accuracy: {ed_accuracy:.2f} ({len(y_true)} samples tested)")
