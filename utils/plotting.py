import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

from legacy.piece_classifier.config import OUTPUT_DIR


def plot_training_history(history, save_path=None):
    """Plot training and validation loss and accuracy over epochs"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot training & validation loss
    ax1.plot(history.history['loss'], label='Training Loss', color='blue')
    ax1.plot(history.history['val_loss'], label='Validation Loss', color='red')
    ax1.set_title('Model Loss Over Epochs')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot training & validation accuracy
    ax2.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
    ax2.plot(history.history['val_accuracy'], label='Validation Accuracy', color='red')
    ax2.set_title('Model Accuracy Over Epochs')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")

    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names, save_path=None):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to: {save_path}")

    plt.show()

    return cm


def evaluate_model(model, val_ds, class_names):
    """Comprehensive model evaluation"""
    print("\n" + "=" * 50)
    print("MODEL EVALUATION")
    print("=" * 50)

    # Get predictions
    y_true = []
    y_pred = []

    print("Generating predictions...")
    for images, labels in val_ds:
        predictions = model.predict(images, verbose=0)
        y_pred.extend(np.argmax(predictions, axis=1))
        y_true.extend(labels.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Overall accuracy
    accuracy = np.mean(y_true == y_pred)
    print(f"\nOverall Validation Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    # Per-class accuracy
    print("\nPer-class Accuracy:")
    print("-" * 30)
    for i, class_name in enumerate(class_names):
        class_mask = y_true == i
        if np.sum(class_mask) > 0:
            class_acc = np.mean(y_pred[class_mask] == y_true[class_mask])
            print(f"{class_name:15}: {class_acc:.4f} ({class_acc * 100:.2f}%)")

    # Classification report
    print("\nDetailed Classification Report:")
    print("-" * 40)
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Plot confusion matrix
    cm = plot_confusion_matrix(y_true, y_pred, class_names,
                               save_path=OUTPUT_DIR / "confusion_matrix.png")

    return y_true, y_pred, accuracy


def print_training_summary(history, final_accuracy):
    """Print summary of training results"""
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)

    final_train_loss = history.history['loss'][-1]
    final_val_loss = history.history['val_loss'][-1]
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]

    print(f"Final Training Loss:     {final_train_loss:.4f}")
    print(f"Final Validation Loss:   {final_val_loss:.4f}")
    print(f"Final Training Accuracy: {final_train_acc:.4f} ({final_train_acc * 100:.2f}%)")
    print(f"Final Validation Accuracy: {final_val_acc:.4f} ({final_val_acc * 100:.2f}%)")
    print(f"Test Accuracy:           {final_accuracy:.4f} ({final_accuracy * 100:.2f}%)")

    # Check for overfitting
    loss_diff = final_train_loss - final_val_loss
    acc_diff = final_train_acc - final_val_acc

    print(f"\nOverfitting Analysis:")
    print(f"Loss difference (train - val): {loss_diff:.4f}")
    print(f"Accuracy difference (train - val): {acc_diff:.4f}")

    if loss_diff < -0.1 and acc_diff > 0.05:
        print("⚠️  Model shows signs of overfitting")
    elif loss_diff > 0.1:
        print("📈 Model might benefit from more training")
    else:
        print("✅ Model training looks balanced")
