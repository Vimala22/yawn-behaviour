# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Basic utilities
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from glob import glob
import random
from PIL import Image
import plotly.express as px

# Deep learning: TensorFlow and Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization)
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Metrics and Evaluation
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Define base path
base_path = "/kaggle/input/yawn-eye-dataset-new/dataset_new" 
splits = ['train', 'test']
# Dataset path (train folder)
base_path = "/kaggle/input/yawn-eye-dataset-new/dataset_new/train"
# Collect pixel values from all images (scaled 0-1)
all_pixels = []
# Limit number of images per class to avoid memory issues (optional)
max_images_per_class = 100  # adjust if you want

print("Loading images and extracting pixel values...")

for cls in os.listdir(base_path):
    cls_path = os.path.join(base_path, cls)
    if os.path.isdir(cls_path):
        image_files = os.listdir(cls_path)[:max_images_per_class]
        for img_file in image_files:
            img_path = os.path.join(cls_path, img_file)
            try:
                img = Image.open(img_path).convert('RGB')  # ensure RGB
                img_arr = np.array(img) / 255.0  # scale pixels to [0,1]
                all_pixels.extend(img_arr.flatten())
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

all_pixels = np.array(all_pixels)
print(f"Total pixels collected: {len(all_pixels)}")
# Downsample pixels for plotting
sample_size = 100_000
plot_pixels = np.random.choice(all_pixels, size=min(len(all_pixels), sample_size), replace=False)

# Compute stats
mean, std = all_pixels.mean(), all_pixels.std()

# Plot
fig = px.histogram(
    x=plot_pixels,
    nbins=100,
    opacity=0.8,
    color_discrete_sequence=['#add8e6'],
    title=f"Pixel Value Distribution (Mean={mean:.4f}, Std={std:.4f})"
)

fig.add_vline(x=mean, line_width=2, line_dash="dash", line_color="red", annotation_text=f"Mean={mean:.3f}")

fig.update_layout(
    xaxis_title='Pixel value (scaled 0–1)',
    yaxis_title='Frequency',
    bargap=0.05,
    template='simple_white'
)

fig.show()
# Define Yawn-Eye Dataset Path
yawn_eye_base = "/kaggle/input/yawn-eye-dataset-new/dataset_new/train"
# Print available class names
print("Available class folders:", os.listdir(yawn_eye_base))

# Class Mapping for Yawn-Eye (adjusted based on actual folder names)
# 1 = Drowsy (yawning or eyes closed), 0 = Awake (not yawning and eyes open)
yawn_eye_map = {
    "Closed": 1,
    "yawn": 1,
    "Open": 0,
    "no_yawn": 0
}
# Function to safely load an image and catch errors
def is_image_valid(path):
    try:
        img = Image.open(path)
        img.verify()  # check if image is corrupted
        return True
    except:
        return False

# Iterate over both train and test
for split in splits:
    print(f"\n Exploring '{split.upper()}' set")
    split_path = os.path.join(base_path, split)
    class_counts = {}
    img_shapes = []
    corrupted_images = []
split_paths = ["/kaggle/input/yawn-eye-dataset-new/dataset_new/train",
               "/kaggle/input/yawn-eye-dataset-new/dataset_new/test"]

class_counts, corrupted_images, img_shapes = {}, [], []

for split_path in split_paths:
    print(f"\nCounting images in: {split_path}")
    for cls in os.listdir(split_path):
        class_dir = os.path.join(split_path, cls)
        if os.path.isdir(class_dir):
            files = os.listdir(class_dir)
            valid_images = []

            for f in files:
                file_path = os.path.join(class_dir, f)
                if is_image_valid(file_path):
                    valid_images.append(f)

                    # Task 3: save shape of first 10 valid images
                    if len(img_shapes) < 10:
                        img = cv2.imread(file_path)
                        if img is not None:
                            img_shapes.append(img.shape)
                else:
                    corrupted_images.append(file_path)

            # Save count with split name (train/test) to avoid clash
            key = f"{os.path.basename(split_path)}/{cls}"
            class_counts[key] = len(valid_images)
            print(f" {cls}: {len(valid_images)} valid images")
dataset_root = "/kaggle/input/yawn-eye-dataset-new/dataset_new"

plt.figure(figsize=(20, 7))
for idx, cls in enumerate(class_counts.keys()):
    cls_path = os.path.join(dataset_root, cls)  # directly join with dataset root
    if not os.path.exists(cls_path):
        print(f"Skipping missing folder: {cls_path}")
        continue

    img_files = [f for f in os.listdir(cls_path) if is_image_valid(os.path.join(cls_path, f))][:2]

    for i, img_file in enumerate(img_files):
        img_path = os.path.join(cls_path, img_file)
        img = Image.open(img_path)
        plt.subplot(len(class_counts), 2, idx * 2 + i + 1)
        plt.imshow(img)
        plt.title(f"{cls}")
        plt.axis('off')

plt.tight_layout()
plt.suptitle("Sample Images from Dataset", fontsize=14, y=1.02)
plt.show()
# Task 5: Plot class distribution
plt.figure(figsize=(7, 4))
sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()))
plt.title(f"Class Distribution - {split.upper()} Set")
plt.ylabel("Number of Images")
plt.xlabel("Class")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
# Evaluation Metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
# Load dataset
yawn_eye_data = []
for cls in os.listdir(yawn_eye_base):
    cls_path = os.path.join(yawn_eye_base, cls)
    label = yawn_eye_map.get(cls, -1)
    if label == -1 or not os.path.isdir(cls_path):
        continue
    for img_file in os.listdir(cls_path):
        try:
            img_path = os.path.join(cls_path, img_file)
            Image.open(img_path).verify()
            yawn_eye_data.append([img_path, label])
        except: pass

df = pd.DataFrame(yawn_eye_data, columns=["image_path", "label"])
print(f"Loaded {len(df)} valid images.")

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Plot distribution
sns.countplot(data=df, x="label")
plt.title("Class Distribution")
plt.xticks([0,1], ["Awake (0)", "Drowsy (1)"])
plt.show()

# Show sample images
for i in range(min(6, len(df))):
    img = Image.open(df.iloc[i]['image_path'])
    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.title(f"Label: {df.iloc[i]['label']}")
    plt.axis('off')
plt.tight_layout()
plt.show()

# Preprocess images
def preprocess(path, size=(96,96)):
    try:
        return np.array(Image.open(path).convert("RGB").resize(size)) / 255.0
    except: return None

X, y = [], []
for _, row in df.iterrows():
    img = preprocess(row['image_path'])
    if img is not None:
        X.append(img)
        y.append(row['label'])

X, y = np.array(X), np.array(y)
print("Preprocessing done. Shape:", X.shape)

# Split data (70/10/20)
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.125, stratify=y_temp, random_state=42)

print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
from tensorflow.keras.applications import VGG16, VGG19, InceptionV3, EfficientNetB0
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix

# Dataset paths
train_dir = "/kaggle/input/yawn-eye-dataset-new/dataset_new/train"
test_dir  = "/kaggle/input/yawn-eye-dataset-new/dataset_new/test"

# Class mapping
yawn_eye_map = {"Closed": 1, "yawn": 1, "Open": 0, "no_yawn": 0}

# Function: load images & labels from directory
def load_data(base_dir):
    data = []
    for cls in os.listdir(base_dir):
        path = os.path.join(base_dir, cls)
        label = yawn_eye_map.get(cls, -1)
        if label == -1 or not os.path.isdir(path): continue
        for img in os.listdir(path):
            try:
                Image.open(os.path.join(path, img)).verify()
                data.append([os.path.join(path, img), label])
            except: pass
    return pd.DataFrame(data, columns=["image_path", "label"])

# Function: preprocess image
def preprocess(path, size=(96,96)):
    try:
        return np.array(Image.open(path).convert("RGB").resize(size)) / 255.0
    except: return None

# Load and preprocess train data
df_train = load_data(train_dir)
X_train, y_train = zip(*[(img, lbl) for path,lbl in df_train.values if (img:=preprocess(path)) is not None])
X_train, y_train = np.array(X_train), np.array(y_train)

# Load and preprocess test data
df_test = load_data(test_dir)
X_test, y_test = zip(*[(img, lbl) for path,lbl in df_test.values if (img:=preprocess(path)) is not None])
X_test, y_test = np.array(X_test), np.array(y_test)

print(f"Train: {len(X_train)} samples | Test: {len(X_test)} samples")

# Function: build model
def build_model(type="vgg16"):
    input_tensor = Input(shape=(96,96,3))
    if type in ["vgg16","vgg19","inception","efficientnet"]:
        base_cls = {"vgg16": VGG16, "vgg19": VGG19, "inception": InceptionV3, "efficientnet": EfficientNetB0}[type]
        base = base_cls(include_top=False, weights='imagenet', input_tensor=input_tensor)
        for l in base.layers: l.trainable = False
        x = GlobalAveragePooling2D()(base.output)
    else:
        x = Conv2D(32,3,activation='relu')(input_tensor)
        x = MaxPooling2D()(x)
        x = Conv2D(64,3,activation='relu')(x)
        x = GlobalAveragePooling2D()(x)
    x = Dense(64 if type!="shufflenet" else 32, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=input_tensor, outputs=out)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Build and train models
models = {name: build_model(name) for name in ["vgg16","vgg19","inception","efficientnet","squeezenet","shufflenet"]}
history = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    hist = model.fit(X_train, y_train, epochs=15, batch_size=32, verbose=1)
    history[name] = hist

print("\nAll models trained.")

# Evaluate each model on test set
for name, model in models.items():
    print(f"\nEvaluation - {name}:")
    preds = (model.predict(X_test) > 0.5).astype(int)
    print(classification_report(y_test, preds))
    cm = confusion_matrix(y_test, preds)
    print("Confusion matrix:\n", cm)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# =====================================================
# TRAINING + EVALUATION + PLOTS FOR ALL MODELS
# =====================================================

results = []
history = {}
y_preds = {}

for name, model in models.items():

    print("\n" + "="*60)
    print(f"Training and Evaluating: {name}")
    print("="*60)

    # -------------------------------------------------
    # Train Model
    # -------------------------------------------------
    hist = model.fit(
        X_train,
        y_train,
        epochs=15,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )

    history[name] = hist

    # -------------------------------------------------
    # Evaluate Model
    # -------------------------------------------------
    train_loss, train_acc = model.evaluate(
        X_train, y_train, verbose=0
    )

    test_loss, test_acc = model.evaluate(
        X_test, y_test, verbose=0
    )

    # -------------------------------------------------
    # Predictions
    # -------------------------------------------------
    y_prob = model.predict(X_test, verbose=0)
    y_pred = (y_prob > 0.5).astype(int)

    y_preds[name] = y_pred

    # -------------------------------------------------
    # Performance Metrics
    # -------------------------------------------------
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    results.append([
        name,
        train_loss,
        test_loss,
        train_acc,
        test_acc,
        acc,
        prec,
        rec,
        f1
    ])

    # -------------------------------------------------
    # Print Metrics
    # -------------------------------------------------
    print(f"\nTrain Loss     : {train_loss:.4f}")
    print(f"Test Loss      : {test_loss:.4f}")
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test Accuracy  : {test_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # -------------------------------------------------
    # Confusion Matrix
    # -------------------------------------------------
    plt.figure(figsize=(5,4))
    sns.heatmap(
        confusion_matrix(y_test, y_pred),
        annot=True,
        fmt='d',
        cmap='Blues'
    )
    plt.title(f'Confusion Matrix - {name}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # -------------------------------------------------
    # Loss & Accuracy Curves
    # -------------------------------------------------
    train_loss_curve = hist.history['loss']
    val_loss_curve = hist.history['val_loss']

    train_acc_curve = hist.history['accuracy']
    val_acc_curve = hist.history['val_accuracy']

    epochs = range(1, len(train_loss_curve) + 1)

    plt.figure(figsize=(14,5))

    # ---------------- Loss Graph ----------------
    plt.subplot(1,2,1)
    plt.plot(
        epochs,
        train_loss_curve,
        marker='o',
        linewidth=2,
        label='Train Loss'
    )
    plt.plot(
        epochs,
        val_loss_curve,
        marker='s',
        linewidth=2,
        label='Test Loss'
    )

    plt.title(f'Loss over Epochs ({name})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # ---------------- Accuracy Graph ----------------
    plt.subplot(1,2,2)
    plt.plot(
        epochs,
        train_acc_curve,
        marker='o',
        linewidth=2,
        label='Train Accuracy'
    )
    plt.plot(
        epochs,
        val_acc_curve,
        marker='s',
        linewidth=2,
        label='Test Accuracy'
    )

    plt.title(f'Accuracy over Epochs ({name})')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# =====================================================
# ENSEMBLE VOTING
# =====================================================

print("\nApplying Ensemble Voting...")

ensemble_prob = np.mean(
    np.array(list(y_preds.values())),
    axis=0
)

ensemble_pred = (ensemble_prob > 0.5).astype(int)

ensemble_acc = accuracy_score(y_test, ensemble_pred)
ensemble_prec = precision_score(y_test, ensemble_pred)
ensemble_rec = recall_score(y_test, ensemble_pred)
ensemble_f1 = f1_score(y_test, ensemble_pred)

print("\nEnsemble Classification Report:")
print(classification_report(y_test, ensemble_pred))

# Ensemble Confusion Matrix
plt.figure(figsize=(5,4))
sns.heatmap(
    confusion_matrix(y_test, ensemble_pred),
    annot=True,
    fmt='d',
    cmap='Greens'
)

plt.title("Confusion Matrix - Ensemble Voting")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# -------------------------------------------------
# Add Ensemble Results
# -------------------------------------------------
results.append([
    "Ensemble Voting",
    np.nan,
    np.nan,
    np.nan,
    np.nan,
    ensemble_acc,
    ensemble_prec,
    ensemble_rec,
    ensemble_f1
])

# =====================================================
# RESULTS TABLE
# =====================================================

results_df = pd.DataFrame(
    results,
    columns=[
        "Model",
        "Train Loss",
        "Test Loss",
        "Train Accuracy",
        "Test Accuracy",
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]
)

print("\nFinal Results Summary:")
print(results_df)

# Save Results
results_df.to_csv("Model_Comparison_Results.csv", index=False)

print("\nResults saved as 'Model_Comparison_Results.csv'")
# Create folder if not exists
save_dir = "/kaggle/working/model"
os.makedirs(save_dir, exist_ok=True)

# Save model in that folder
model_path = os.path.join(save_dir, "inception_best.h5")
best_model = models['inception']
best_model.save(model_path)

print(f"Inception model saved successfully at: {model_path}")
import matplotlib.pyplot as plt
import numpy as np
import random

# Label mapping
def label_text(label):
    return "Drowsy" if label == 1 else "Awake"

# Prediction function
def predict_label(img):
    img_input = np.expand_dims(img, axis=0)
    pred_prob = best_model.predict(img_input, verbose=0)[0][0]
    return 1 if pred_prob > 0.5 else 0


# -------------------------------
# 1. RANDOM SAMPLE PREDICTIONS
# -------------------------------
plt.figure(figsize=(12,6))
plt.suptitle("Random Sample Predictions - Inception Model", fontsize=14)

indices = random.sample(range(len(X_test)), 6)

for i, idx in enumerate(indices):
    img = X_test[idx]
    pred = predict_label(img)
    actual = y_test[idx]

    plt.subplot(2,3,i+1)
    plt.imshow(img)
    plt.title(f"Pred: {label_text(pred)}\nActual: {label_text(actual)}")
    plt.axis('off')

plt.tight_layout()
plt.show()


# -------------------------------
# 2. RANDOM CORRECT PREDICTIONS
# -------------------------------
plt.figure(figsize=(12,6))
plt.suptitle("Random Correct Predictions", fontsize=14)

correct_indices = [i for i in range(len(X_test)) if predict_label(X_test[i]) == y_test[i]]
selected = random.sample(correct_indices, min(6, len(correct_indices)))

for i, idx in enumerate(selected):
    plt.subplot(2,3,i+1)
    plt.imshow(X_test[idx])
    plt.title(f"Correct: {label_text(y_test[idx])}")
    plt.axis('off')

plt.tight_layout()
plt.show()


# -------------------------------
# 3. RANDOM MISCLASSIFIED SAMPLES
# -------------------------------
plt.figure(figsize=(12,6))
plt.suptitle("Random Misclassified Samples", fontsize=14)

wrong_indices = [i for i in range(len(X_test)) if predict_label(X_test[i]) != y_test[i]]
selected = random.sample(wrong_indices, min(6, len(wrong_indices)))

for i, idx in enumerate(selected):
    pred = predict_label(X_test[idx])
    
    plt.subplot(2,3,i+1)
    plt.imshow(X_test[idx])
    plt.title(f"Pred: {label_text(pred)}\nActual: {label_text(y_test[idx])}")
    plt.axis('off')

plt.tight_layout()
plt.show()
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Binary prediction function (Awake / Drowsy)
def lime_predict(images):

    preds = best_model.predict(
        images,
        verbose=0
    )

    # Convert sigmoid output to 2-class probabilities
    if preds.shape[1] == 1:

        preds = np.concatenate(
            [1-preds, preds],
            axis=1
        )

    return preds


# Number of images
num_images = 5

plt.figure(figsize=(15, num_images*4))

explainer = lime_image.LimeImageExplainer()

for sample in range(num_images):

    image = X_test[sample]

    explanation = explainer.explain_instance(

        image.astype('double'),

        lime_predict,

        top_labels=2,

        hide_color=0,

        num_samples=1000

    )

    pred = best_model.predict(

        np.expand_dims(
            image,
            axis=0
        ),

        verbose=0

    )[0][0]

    pred_class = int(pred > 0.5)

    temp, mask = explanation.get_image_and_mask(

        pred_class,

        positive_only=True,

        num_features=10,

        hide_rest=False

    )

    # Original Image
    plt.subplot(
        num_images,
        3,
        sample*3+1
    )

    plt.imshow(image)

    plt.title(

        f"Original {sample+1}"

    )

    plt.axis("off")

    # LIME Mask
    plt.subplot(
        num_images,
        3,
        sample*3+2
    )

    plt.imshow(
        mask,
        cmap='jet'
    )

    plt.title(
        "LIME Mask"
    )

    plt.axis("off")

    # Overlay
    plt.subplot(
        num_images,
        3,
        sample*3+3
    )

    plt.imshow(

        mark_boundaries(
            temp/255.0,
            mask
        )

    )

    plt.title(
        "LIME Overlay"
    )

    plt.axis("off")

plt.tight_layout()

plt.show()
import shap
import numpy as np
import matplotlib.pyplot as plt

# Number of images
num_images = 10

# Select images from Yawn dataset test set
test_images = X_test[:num_images]

# SHAP Masker
masker = shap.maskers.Image(
    "inpaint_telea",
    test_images[0].shape
)

# Build explainer
explainer = shap.Explainer(
    best_model,
    masker
)

# Compute explanations
shap_values = explainer(
    test_images,
    max_evals=500
)

values = shap_values.values

# Binary classification handling
if values.shape[-1] == 1:
    values = values[...,0]

# Label mapping
label_names = {
    0:"Awake",
    1:"Drowsy"
}

# Model predictions
preds = best_model.predict(
    test_images,
    verbose=0
)

pred_classes = (
    preds>0.5
).astype(int).flatten()

plt.figure(
    figsize=(15,num_images*4)
)

for sample in range(num_images):

    heatmap = values[
        sample
    ].mean(axis=-1)

    # Original Image
    plt.subplot(
        num_images,
        3,
        sample*3+1
    )

    plt.imshow(
        test_images[sample]
    )

    plt.title(
        f"Original\nPred:{label_names[pred_classes[sample]]}"
    )

    plt.axis("off")

    # SHAP Heatmap
    plt.subplot(
        num_images,
        3,
        sample*3+2
    )

    plt.imshow(
        heatmap,
        cmap='jet'
    )

    plt.title(
        "SHAP Heatmap"
    )

    plt.axis("off")

    # Overlay
    plt.subplot(
        num_images,
        3,
        sample*3+3
    )

    plt.imshow(
        test_images[sample]
    )

    plt.imshow(
        heatmap,
        cmap='jet',
        alpha=0.5
    )

    plt.title(
        "Overlay"
    )

    plt.axis("off")

plt.tight_layout()

plt.show()
sample = 0

plt.figure(figsize=(12,4))

# Original image
plt.subplot(1,3,1)
plt.imshow(test_images[sample])
plt.title("Original Image")
plt.axis("off")

# SHAP heatmap
plt.subplot(1,3,2)
plt.imshow(values[sample].mean(axis=-1), cmap='jet')
plt.title("SHAP Heatmap")
plt.axis("off")

# Overlay
plt.subplot(1,3,3)
plt.imshow(test_images[sample])
plt.imshow(
    values[sample].mean(axis=-1),
    cmap='jet',
    alpha=0.5
)
plt.title("Overlay Visualization")
plt.axis("off")

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Number of images to visualize
num_images = 10

plt.figure(figsize=(15, num_images*4))

for sample in range(num_images):

    # Original Image
    plt.subplot(num_images,3,sample*3+1)

    plt.imshow(test_images[sample])
    plt.title(f"Original {sample+1}")
    plt.axis("off")

    # SHAP Heatmap
    plt.subplot(num_images,3,sample*3+2)

    heatmap = values[sample].mean(axis=-1)

    plt.imshow(
        heatmap,
        cmap='jet'
    )

    plt.title(f"SHAP {sample+1}")
    plt.axis("off")

    # Overlay Visualization
    plt.subplot(num_images,3,sample*3+3)

    plt.imshow(test_images[sample])

    plt.imshow(
        heatmap,
        cmap='jet',
        alpha=0.5
    )

    plt.title(f"Overlay {sample+1}")
    plt.axis("off")

plt.tight_layout()
plt.show()