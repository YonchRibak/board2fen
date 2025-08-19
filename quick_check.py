import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from legacy.piece_classifier.dataset_utils import load_datasets
from collections import Counter

# Load the datasets
train_ds, val_ds, class_names = load_datasets()
print("Classes:", class_names)

# Check validation set balance
val_labels = []
for images, labels in val_ds.take(10):  # Just first 10 batches
    val_labels.extend(labels.numpy())

counter = Counter(val_labels)
for i, name in enumerate(class_names):
    print(f"{name}: {counter.get(i, 0)} samples")