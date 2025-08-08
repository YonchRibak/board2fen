
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
from .config import DATASET_DIR, BATCH_SIZE, IMG_SIZE, SEED

AUTOTUNE = tf.data.AUTOTUNE

def preprocess(image, label):
    image = tf.cast(image, tf.float32) / 255.0  # Normalize
    return image, label

def load_datasets():
    train_ds_raw = image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="training",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    val_ds_raw = image_dataset_from_directory(
        DATASET_DIR,
        validation_split=0.2,
        subset="validation",
        seed=SEED,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
    )

    class_names = train_ds_raw.class_names

    train_ds = train_ds_raw.map(preprocess).shuffle(100).prefetch(AUTOTUNE)
    val_ds = val_ds_raw.map(preprocess).prefetch(AUTOTUNE)

    return train_ds, val_ds, class_names
