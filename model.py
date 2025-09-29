import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" #TODO: warning weghalen

MODEL_PATH = 'skin_condition_model.h5'

# TODO: make this more user friendly
# Load the image here:
img = cv2.imread("./testdata/images.jpg")

# Resize for model
img_resized = cv2.resize(img, (224, 224)) / 255.0

# Add batch dimension
input_data = np.expand_dims(img_resized, axis=0)

if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model = load_model(MODEL_PATH)
else:
    print("Training new model...")
    # Base model (pretrained on ImageNet)
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))

    # summarize
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(5, activation="softmax")(x)  # 5 classes, adjust as needed

    model = Model(inputs=base_model.input, outputs=predictions)

    # Freeze base model
    for layer in base_model.layers:
        layer.trainable = False

    # Compile
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # split the dataset for validation
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # 20% validation

    train_dataset = datagen.flow_from_directory(
        "datasets",
        target_size=(224, 224),
        batch_size=32,
        class_mode="sparse",  # sparse integer labels
        subset="training"
    )

    # to protect against memory bias of the model, validation data
    val_dataset = datagen.flow_from_directory(
        "datasets",
        target_size=(224, 224),
        batch_size=32,
        class_mode="categorical",
        subset="validation"
    )

    # training with the dataset
    # NOTE: epoch means amount of training rounds
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=10)

    # Save model so we don't retrain next time
    model.save(MODEL_PATH)

print("processing...")
pred = model.predict(input_data)
class_index = np.argmax(pred)

skin_problems = {
    0 : "eczeem",
    1 : "Ehler Danlos",
    2 : "healthy",
    3 : "mailgnant",
    4 : "psoriasis"
}

print("Predicted skin condition:", skin_problems[class_index])
