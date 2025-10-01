import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.utils.class_weight import compute_class_weight
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0" #TODO: warning weghalen

MODEL_PATH = 'skin_condition_model.h5'
# Enable GPU 
    
# TODO: make this more user friendly
# Load the image here:
img = cv2.imread("./datasets/archive/train/Ehler Danlos/eds2.jpg", cv2.IMREAD_COLOR)
img_resized = cv2.resize(img, (224, 224))
img_resized = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
img_resized = preprocess_input(img_resized)

# Add batch dimension
input_data = np.expand_dims(img_resized, axis=0)

if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model = load_model(MODEL_PATH)
else:
    print("Training new model...")
    # Base model (pretrained)
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))

    # summarize
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)   # helps generalize
    predictions = Dense(5, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers[-30:]:  # last 30 layers
        layer.trainable = True

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # split the dataset for validation
    datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    train_dataset = datagen.flow_from_directory(
        "datasets",
        target_size=(224, 224),
        batch_size=32,
        class_mode="sparse", 
        subset="training"
    )

    # to protect against memory bias of the model, validation data
    val_dataset = datagen.flow_from_directory(
        "datasets",
        target_size=(224, 224),
        batch_size=32,
        class_mode="sparse",
        subset="validation"
    )

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(train_dataset.classes),
        y=train_dataset.classes
    )
    class_weights = dict(enumerate(class_weights))
    # training with the dataset
    # NOTE: epoch means amount of training rounds
    history = model.fit(train_dataset, validation_data=val_dataset, epochs=5, class_weight=class_weights)

    # Save model so we don't retrain next time
    model.save(MODEL_PATH)

print("processing...")

pred = model.predict(input_data)
print(pred)
class_index = np.argmax(pred)

skin_problems = {
    0 : "eczeem",
    1 : "Ehler Danlos",
    2 : "healthy",
    3 : "mailgnant",
    4 : "psoriasis"
}

if pred[0][1] > 0.5:
    print("Ehler Danlos is likely")
else:
    print("Ehler Danlos not likely")
