"""
Multi-class Insect Bite Classifier with ResNet50 and LIME Explanations
- Uses validation split from training data
- Filters test data to match training classes
- Saves all outputs to 'newresult' folder
"""

import numpy as np
import pandas as pd
import os
from PIL import Image
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.applications import ResNet50
import plotly.graph_objects as go
from lime import lime_image

# ------------------------------
# Configuration
# ------------------------------
IMAGE_SIZE = 128
BATCH_SIZE = 10
EPOCHS = 1000
DATA_DIR = "../bug/input/data"                # Folder containing 'train' and 'test' subfolders
RESULTS_DIR = "newresult"         # All outputs go here
VALIDATION_SPLIT = 0.2            # Use 20% of training data for validation

os.makedirs(RESULTS_DIR, exist_ok=True)

# ------------------------------
# Data generators with validation split
# ------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=90,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=VALIDATION_SPLIT      # This creates a validation split
)

# No augmentation for validation/test
val_test_datagen = ImageDataGenerator(rescale=1./255)

# Training generator (subset='training')
train_generator = train_datagen.flow_from_directory(
    directory=os.path.join(DATA_DIR, 'train'),
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    color_mode='rgb',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True,
    seed=42
)

# Validation generator (subset='validation')
val_generator = train_datagen.flow_from_directory(
    directory=os.path.join(DATA_DIR, 'train'),
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    color_mode='rgb',
    batch_size=BATCH_SIZE,        # Can use larger batch for validation
    class_mode='categorical',
    subset='validation',
    shuffle=True,
    seed=42
)

# Test generator – restrict to the classes found in training
class_names = list(train_generator.class_indices.keys())
test_generator = val_test_datagen.flow_from_directory(
    directory=os.path.join(DATA_DIR, 'test'),
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    color_mode='rgb',
    batch_size=1,
    class_mode='categorical',
    classes=class_names,           # Only keep these classes
    shuffle=False,                 # Keep order for later file mapping
    seed=42
)

# Number of steps
STEP_SIZE_TRAIN = train_generator.samples // train_generator.batch_size
STEP_SIZE_VALID = val_generator.samples // val_generator.batch_size
STEP_SIZE_TEST = test_generator.samples // test_generator.batch_size

print(f"Train samples: {train_generator.samples}, Validation samples: {val_generator.samples}")
print(f"Test samples (after filtering): {test_generator.samples}")
print(f"Classes: {class_names}")

# ------------------------------
# Build the model
# ------------------------------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(len(class_names), activation='softmax')(x)   # Use number of classes
model = Model(inputs=base_model.input, outputs=predictions)

# Learning rate schedule
decay_steps = EPOCHS * STEP_SIZE_TRAIN
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=decay_steps,
    alpha=0.0
)

model.compile(
    optimizer=SGD(learning_rate=lr_schedule, momentum=0.9),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callback to save best model
checkpoint = ModelCheckpoint(
    filepath=os.path.join(RESULTS_DIR, 'weights_best.h5'),
    monitor='val_loss',
    save_best_only=True,
    mode='min',
    verbose=1
)

# ------------------------------
# Training
# ------------------------------
history = model.fit(
    train_generator,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=val_generator,
    validation_steps=STEP_SIZE_VALID,
    epochs=EPOCHS,
    callbacks=[checkpoint],
    verbose=1
)

# ------------------------------
# Save the final model
# ------------------------------
model.save(os.path.join(RESULTS_DIR, 'insect_bite_classifier.h5'))
print(f"Model saved as {os.path.join(RESULTS_DIR, 'insect_bite_classifier.h5')}")

# ------------------------------
# Evaluation on validation set
# ------------------------------
val_loss, val_acc = model.evaluate(val_generator, steps=STEP_SIZE_VALID, verbose=0)
print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

# ------------------------------
# Predict on test set and save results
# ------------------------------
test_generator.reset()
pred = model.predict(test_generator, steps=STEP_SIZE_TEST, verbose=1)
predicted_class_indices = np.argmax(pred, axis=1)

# Map indices to class labels
labels = {v: k for k, v in train_generator.class_indices.items()}   # same as class_names order
predictions = [labels[idx] for idx in predicted_class_indices]

filenames = test_generator.filenames
results = pd.DataFrame({"Filename": filenames, "Predictions": predictions})
results.to_csv(os.path.join(RESULTS_DIR, "results.csv"), index=False)
print(f"Test predictions saved to {os.path.join(RESULTS_DIR, 'results.csv')}")
print(results)

# ------------------------------
# Plot training curves (plotly HTML)
# ------------------------------
def display_training_curves(training, validation, yaxis, filename):
    if yaxis == "loss":
        ylabel = "Loss"
        title = "Loss vs. Epochs"
    else:
        ylabel = "Accuracy"
        title = "Accuracy vs. Epochs"

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=np.arange(1, EPOCHS+1), y=training, mode='lines+markers',
                             name='Train', marker=dict(color='dodgerblue')))
    fig.add_trace(go.Scatter(x=np.arange(1, EPOCHS+1), y=validation, mode='lines+markers',
                             name='Validation', marker=dict(color='darkorange')))
    fig.update_layout(title=title, xaxis_title='Epochs', yaxis_title=ylabel, template='plotly_white')
    fig.write_html(os.path.join(RESULTS_DIR, filename))
    print(f"Plot saved as {os.path.join(RESULTS_DIR, filename)}")

display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 
                        'Accuracy', 'accuracy_curve.html')
display_training_curves(history.history['loss'], history.history['val_loss'], 
                        'loss', 'loss_curve.html')

# ------------------------------
# LIME Explanations on first few test images
# ------------------------------
test_generator.reset()
X_exp = []
for i in range(min(4, test_generator.samples)):
    img_batch, _ = next(test_generator)
    X_exp.append(img_batch[0])
X_exp = np.array(X_exp)

explainer = lime_image.LimeImageExplainer()

for i in range(len(X_exp)):
    print(f"\n--- Explaining image {i+1} ---")
    prob = model.predict(X_exp[i].reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3))[0]
    top_idx = np.argmax(prob)
    print(f"Predicted: {labels[top_idx]} (prob: {prob[top_idx]:.4f})")

    # Save original image
    plt.figure()
    plt.imshow(X_exp[i])
    plt.title(f"Original Image {i+1}")
    plt.axis('off')
    plt.savefig(os.path.join(RESULTS_DIR, f"image{i+1}.png"))
    plt.close()

    # Generate explanation
    explanation = explainer.explain_instance(
        X_exp[i],
        model.predict,
        top_labels=len(class_names),
        hide_color=0,
        num_samples=1000
    )

    # Get explanation for the predicted class
    temp, mask = explanation.get_image_and_mask(
        top_idx,
        positive_only=True,
        num_features=5,
        hide_rest=True
    )
    # Visualize
    plt.figure()
    plt.imshow(mark_boundaries(temp / 2 + 0.5, mask))
    plt.title(f"LIME Explanation (class: {labels[top_idx]})")
    plt.axis('off')
    plt.savefig(os.path.join(RESULTS_DIR, f"image{i+1}_exp.png"))
    plt.close()
    print(f"Explanation saved as image{i+1}_exp.png")

print("\nAll done! Results saved in 'newresult' folder.")