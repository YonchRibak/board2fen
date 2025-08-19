# second_attempt_train.py - Revised training with better configurations

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import tensorflow as tf
from tensorflow.keras import layers, models, applications
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

from legacy.piece_classifier.config import IMG_SIZE, NUM_CLASSES, OUTPUT_DIR
from legacy.piece_classifier.dataset_utils import load_datasets


def build_improved_model():
    """Build model with better training characteristics"""

    # Use ResNet50 - reliable, well-tested, no weight loading issues
    base_model = applications.ResNet50(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet"
    )

    # Start with frozen backbone
    base_model.trainable = False

    model = models.Sequential([
        base_model,

        # Improved head for better learning
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),

        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),

        layers.Dense(128, activation='relu'),
        layers.Dropout(0.2),

        layers.Dense(NUM_CLASSES, activation='softmax')
    ])

    return model


def progressive_training_strategy(model, train_ds, val_ds, class_names):
    """Two-phase progressive training for better results"""

    print("\n🚀 PROGRESSIVE TRAINING STRATEGY")
    print("=" * 50)

    # ====================================================================
    # PHASE 1: Train classifier head with higher learning rate
    # ====================================================================

    print("\n📚 PHASE 1: Training classification head")
    print("Backbone: FROZEN (transfer learning)")
    print("Learning rate: 1e-3 (high for fast learning)")

    # Compile with higher learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3, beta_1=0.9, beta_2=0.999),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
    )

    # Callbacks for Phase 1
    callbacks_phase1 = [
        tf.keras.callbacks.ModelCheckpoint(
            OUTPUT_DIR / "phase1_best.keras",
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1
        )
    ]

    # Train Phase 1
    print("\nStarting Phase 1 training...")
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks_phase1,
        verbose=1
    )

    # Check Phase 1 results
    phase1_acc = max(history1.history['val_accuracy'])
    print(f"\n📊 Phase 1 Results: Best validation accuracy = {phase1_acc:.4f} ({phase1_acc * 100:.1f}%)")

    if phase1_acc < 0.5:
        print("❌ Phase 1 failed to reach 50% accuracy. Stopping training.")
        print("   This suggests data loading or model architecture issues.")
        return [history1], False

    # ====================================================================
    # PHASE 2: Fine-tune entire model with lower learning rate
    # ====================================================================

    print("\n🔓 PHASE 2: Fine-tuning entire model")
    print("Backbone: UNFROZEN (full training)")
    print("Learning rate: 1e-4 (lower for stability)")

    # Unfreeze the backbone
    base_model = model.layers[0]  # Base model is at index 0 now
    base_model.trainable = True

    # Compile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.9, beta_2=0.999),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy', 'sparse_top_k_categorical_accuracy']
    )

    # Callbacks for Phase 2
    callbacks_phase2 = [
        tf.keras.callbacks.ModelCheckpoint(
            OUTPUT_DIR / "phase2_best.keras",
            save_best_only=True,
            monitor='val_accuracy',
            mode='max',
            verbose=1
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=7,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.3,
            patience=3,
            min_lr=1e-7,
            verbose=1
        )
    ]

    # Train Phase 2
    print("\nStarting Phase 2 training...")
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=15,
        callbacks=callbacks_phase2,
        verbose=1
    )

    # Final results
    phase2_acc = max(history2.history['val_accuracy'])
    print(f"\n📊 Phase 2 Results: Best validation accuracy = {phase2_acc:.4f} ({phase2_acc * 100:.1f}%)")

    return [history1, history2], True


def plot_training_progress(histories, save_path=None):
    """Plot training progress for both phases"""

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # Combine histories
    all_loss = []
    all_val_loss = []
    all_acc = []
    all_val_acc = []

    phase_boundaries = [0]

    for i, history in enumerate(histories):
        all_loss.extend(history.history['loss'])
        all_val_loss.extend(history.history['val_loss'])
        all_acc.extend(history.history['accuracy'])
        all_val_acc.extend(history.history['val_accuracy'])
        phase_boundaries.append(len(all_loss))

    epochs = range(len(all_loss))

    # Plot 1: Loss
    ax1.plot(epochs, all_loss, 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, all_val_loss, 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontsize=14)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add phase boundaries
    for boundary in phase_boundaries[1:-1]:
        ax1.axvline(x=boundary - 0.5, color='gray', linestyle='--', alpha=0.7)

    # Plot 2: Accuracy
    ax2.plot(epochs, all_acc, 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, all_val_acc, 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontsize=14)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Add phase boundaries
    for boundary in phase_boundaries[1:-1]:
        ax2.axvline(x=boundary - 0.5, color='gray', linestyle='--', alpha=0.7)

    # Plot 3: Loss zoom (last 10 epochs)
    last_epochs = max(10, len(all_loss))
    ax3.plot(epochs[-last_epochs:], all_loss[-last_epochs:], 'b-', label='Training Loss')
    ax3.plot(epochs[-last_epochs:], all_val_loss[-last_epochs:], 'r-', label='Validation Loss')
    ax3.set_title('Loss (Final Epochs)', fontsize=14)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Loss')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Accuracy zoom (last 10 epochs)
    ax4.plot(epochs[-last_epochs:], all_acc[-last_epochs:], 'b-', label='Training Accuracy')
    ax4.plot(epochs[-last_epochs:], all_val_acc[-last_epochs:], 'r-', label='Validation Accuracy')
    ax4.set_title('Accuracy (Final Epochs)', fontsize=14)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training plots saved to: {save_path}")

    plt.show()


def evaluate_final_model(model, val_ds, class_names):
    """Comprehensive evaluation of final model"""

    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION")
    print("=" * 60)

    # Overall evaluation
    val_loss, val_accuracy, val_top_k = model.evaluate(val_ds, verbose=1)

    print(f"\nFINAL RESULTS:")
    print(f"Validation Loss:      {val_loss:.4f}")
    print(f"Validation Accuracy:  {val_accuracy:.4f} ({val_accuracy * 100:.2f}%)")
    print(f"Top-2 Accuracy:       {val_top_k:.4f} ({val_top_k * 100:.2f}%)")

    # Detailed predictions
    print("\nGenerating detailed predictions...")
    y_true = []
    y_pred = []

    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Per-class accuracy
    print("\nPER-CLASS ACCURACY:")
    print("-" * 40)
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_pred[class_mask] == y_true[class_mask])
            count = np.sum(class_mask)
            print(f"{class_name:15}: {class_acc:.4f} ({class_acc * 100:.1f}%) - {count} samples")

    # Classification report
    print("\nDETAILED CLASSIFICATION REPORT:")
    print("-" * 50)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Final Model')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks(rotation=45)
    plt.yticks(rotation=0)
    plt.tight_layout()

    conf_matrix_path = OUTPUT_DIR / "final_confusion_matrix.png"
    plt.savefig(conf_matrix_path, dpi=300, bbox_inches='tight')
    plt.show()

    # Performance assessment
    print("\nPERFORMANCE ASSESSMENT:")
    print("-" * 30)
    if val_accuracy >= 0.95:
        print("🎉 EXCELLENT! Model ready for production use.")
    elif val_accuracy >= 0.90:
        print("✅ VERY GOOD! Model performs well for most use cases.")
    elif val_accuracy >= 0.85:
        print("✅ GOOD! Model is usable but could benefit from improvements.")
    elif val_accuracy >= 0.80:
        print("⚠️  ACCEPTABLE! Consider more training or data improvements.")
    else:
        print("❌ POOR! Model needs significant improvements.")

    return val_accuracy


def main():
    """Main training function with improved configuration"""

    print("♟️  SECOND ATTEMPT CHESS PIECE CLASSIFIER")
    print("=" * 60)
    print("Improvements: Better model, progressive training, higher learning rates")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load datasets
    print("\nLoading datasets...")
    train_ds, val_ds, class_names = load_datasets()
    print(f"Classes: {class_names}")
    print(f"Number of classes: {len(class_names)}")

    # Build improved model
    print("\nBuilding improved model...")
    model = build_improved_model()
    model.summary()

    # Progressive training
    histories, success = progressive_training_strategy(model, train_ds, val_ds, class_names)

    if not success:
        print("❌ Training failed. Please check data loading and model configuration.")
        return

    # Plot training progress
    plot_training_progress(histories, save_path=OUTPUT_DIR / "training_progress.png")

    # Load best model for final evaluation
    best_model_path = OUTPUT_DIR / "phase2_best.keras"
    if best_model_path.exists():
        print(f"\nLoading best model from: {best_model_path}")
        final_model = tf.keras.models.load_model(best_model_path)
    else:
        final_model = model

    # Final evaluation
    final_accuracy = evaluate_final_model(final_model, val_ds, class_names)

    # Save final model
    final_model.save(OUTPUT_DIR / "final_chess_piece_classifier.keras")

    print(f"\n🎯 TRAINING COMPLETE!")
    print(f"Final accuracy: {final_accuracy:.4f} ({final_accuracy * 100:.1f}%)")
    print(f"Model saved to: {OUTPUT_DIR / 'final_chess_piece_classifier.keras'}")
    print(f"All outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()