# piece_classifier/train.py
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable aggressive native CPU optimizations

from tensorflow.keras import layers, models
from piece_classifier.config import IMG_SIZE, NUM_CLASSES, OUTPUT_DIR, LEARNING_RATE
from piece_classifier.dataset_utils import load_datasets


import tensorflow as tf
from utils.plotting import plot_training_history, plot_confusion_matrix, print_training_summary, evaluate_model

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)


def build_model():
    base_model = tf.keras.applications.ResNet50(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model



def main():
    # Create output directory if it doesn't exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    train_ds, val_ds, class_names = load_datasets()
    print("Detected classes:", class_names)
    print(f"Number of classes: {len(class_names)}")

    # Build and compile model
    model = build_model()
    model.summary()

    # Setup callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        OUTPUT_DIR / "best_model.keras",
        save_best_only=True,
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        monitor='val_accuracy',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    reduce_lr_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )

    # Train the model
    print("\nStarting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=20,
        callbacks=[checkpoint_cb, early_stopping_cb, reduce_lr_cb],
        verbose=1
    )

    # Plot training history
    plot_training_history(history, save_path=OUTPUT_DIR / "training_history.png")

    # Load best model for evaluation
    print("\nLoading best model for evaluation...")
    best_model = tf.keras.models.load_model(OUTPUT_DIR / "best_model.keras")

    # Comprehensive evaluation
    y_true, y_pred, final_accuracy = evaluate_model(best_model, val_ds, class_names)

    # Print training summary
    print_training_summary(history, final_accuracy)

    # Save training history
    import json
    history_dict = {
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']],
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']]
    }

    with open(OUTPUT_DIR / "training_history.json", 'w') as f:
        json.dump(history_dict, f, indent=2)

    print(f"\nAll outputs saved to: {OUTPUT_DIR}")
    print("✅ Training completed successfully!")


if __name__ == "__main__":
    main()