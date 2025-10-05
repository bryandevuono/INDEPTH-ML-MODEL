import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dropout
from sklearn.utils.class_weight import compute_class_weight
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

MODEL_PATH = 'skin_condition_model.h5'
    
# TODO: make this more user friendly

img = cv2.imread("./testdata/Ehler Danlos/Ehlers-Danlos-syndroom2.jpg", cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_resized = cv2.resize(img, (224,224))
img_resized = img_resized / 255.0  # match ImageDataGenerator's rescale

input_data = np.expand_dims(img_resized, axis=0)

if os.path.exists(MODEL_PATH):
    print("Loading saved model...")
    model = load_model(MODEL_PATH)
else:
    print("Training new model...")                  
    # Base model (pretrained)
    base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    predictions = Dense(2, activation="softmax", kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    for layer in base_model.layers:
        layer.trainable = False
    for layer in base_model.layers[-30:]:
        layer.trainable = True


    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    # split the dataset
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
        "datasets/Train",
        target_size=(224, 224),
        batch_size=32,
        class_mode="sparse", 
        subset="training"
    )

    val_dataset = datagen.flow_from_directory(
        "datasets/Train",
        target_size=(224, 224),
        batch_size=32,
        class_mode="sparse",
        subset="validation"
    )

    y_train = train_dataset.classes
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )

    class_weights = dict(enumerate(class_weights))
    # training with the dataset
    # NOTE: epoch means amount of training rounds
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=20,
        class_weight=class_weights,
    )

    # Save model so we don't retrain next time
    model.save(MODEL_PATH)

print("processing...")

pred = model.predict(input_data)
print(pred)
class_index = np.argmax(pred)

skin_problems = {
    0 : "Ehler Danlos",
    1 : "Healthy"
}


print(skin_problems[class_index])
