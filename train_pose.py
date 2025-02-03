#!/usr/bin/env python3
import os
import logging
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# === Step 1: Load Dataset ===
csv_file = "pose_data.csv"  # update if your file name/path is different
logging.info("Loading dataset from CSV...")
try:
    data = pd.read_csv(csv_file)
except Exception as e:
    logging.error("Failed to load CSV file: " + str(e))
    exit(1)
logging.info(f"Dataset loaded with shape: {data.shape}")

# === Step 2: Preprocess Dataset ===
# Expecting columns: x1, y1, z1, visibility1, x2, y2, z2, visibility2, ..., x33, y33, z33, visibility33, class
if "class" not in data.columns:
    logging.error("Dataset must contain a 'class' column for labels!")
    exit(1)

logging.info("Preprocessing data...")

# Separate features (all columns except 'class') and labels
X = data.drop("class", axis=1).values.astype(np.float32)
y = data["class"].values
logging.info(f"Features shape: {X.shape}, Labels shape: {y.shape}")

# Encode class labels to integers and then one-hot encode them
logging.info("Encoding labels...")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(np.unique(y_encoded))
logging.info(f"Classes found: {label_encoder.classes_} (encoded as {np.unique(y_encoded)})")
y_onehot = tf.keras.utils.to_categorical(y_encoded, num_classes=num_classes)

# === Step 3: Split Dataset ===
logging.info("Splitting dataset into training and test sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.2, random_state=42)
logging.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")

# === Step 4: Build the Model ===
logging.info("Building the model...")
model = Sequential([
    Dense(128, activation='relu', input_shape=(X.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary(print_fn=lambda x: logging.info(x))

# === Step 5: Train the Model ===
logging.info("Training the model...")
early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = model.fit(X_train, y_train, 
                    validation_split=0.2, 
                    epochs=100, 
                    batch_size=32, 
                    callbacks=[early_stop])
logging.info("Training completed.")

# === Step 6: Evaluate the Model ===
logging.info("Evaluating the model on test set...")
loss, accuracy = model.evaluate(X_test, y_test)
logging.info(f"Test Loss: {loss:.4f}, Test Accuracy: {accuracy:.4f}")

# === Step 7: Save the Model for TFJS Conversion ===
import tensorflow as tf
import os

model_save_path = "saved_model/pose_model"
logging.info(f"Saving the model to {model_save_path} ...")

# Ensure directory exists
os.makedirs(model_save_path, exist_ok=True)

# Save model in TensorFlow's SavedModel format (required for TFJS conversion)
tf.saved_model.save(model, "saved_model/pose_model")


# convert the model to graph
# !tensorflowjs_converter \
#     --input_format=tf_saved_model \
#     --output_format=tfjs_graph_model \
#     --signature_name=serving_default \
#     --saved_model_tags=serve \
#     saved_model/pose_model \
#     tfjs_model
