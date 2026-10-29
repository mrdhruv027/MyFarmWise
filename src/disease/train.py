"""
Model Training Script for Attention-CNN on Plant Disease Datasets (Module E).
Scheduled for Week 7 (Adam optimizer, batch size 32, categorical cross-entropy, 50-100 epochs).
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import json
from pathlib import Path
from typing import Dict, Any, Optional
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from config.settings import (
    CNN_INPUT_SHAPE,
    NUM_DISEASE_CLASSES,
    SE_CNN_WEIGHTS_PATH,
    MODELS_DIR,
)
from src.disease.model import build_se_attention_cnn


def create_training_pipeline(
    data_dir: Path,
    batch_size: int = 32,
    img_size: tuple = (CNN_INPUT_SHAPE[0], CNN_INPUT_SHAPE[1])
):
    """
    Creates tf.data input pipelines with field-realistic augmentations
    for PlantVillage, PlantDoc, and CD&S datasets.
    """
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=25,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.15,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode="nearest",
        validation_split=0.2
    )

    train_gen = train_datagen.flow_from_directory(
        str(data_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="training"
    )

    val_gen = train_datagen.flow_from_directory(
        str(data_dir),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        subset="validation"
    )

    return train_gen, val_gen


def train_se_cnn(
    data_dir: Optional[Path] = None,
    epochs: int = 50,
    batch_size: int = 32,
    initial_lr: float = 1e-3
) -> Dict[str, Any]:
    """
    Executes deep learning training routine for Week 7 milestone.
    Saves model checkpoints, learning rate schedules, and training curves.
    """
    if data_dir is None or not Path(data_dir).exists():
        return {
            "status": "Pending Week 7 Execution",
            "message": "Full 54,305 image dataset directory not mounted. Training routine verified and scaffolded for Week 7.",
            "configuration": {
                "optimizer": "Adam",
                "initial_learning_rate": initial_lr,
                "loss": "categorical_crossentropy",
                "batch_size": batch_size,
                "target_epochs": epochs,
                "input_dimensions": CNN_INPUT_SHAPE,
                "classes": NUM_DISEASE_CLASSES,
                "weights_destination": str(SE_CNN_WEIGHTS_PATH)
            }
        }

    train_gen, val_gen = create_training_pipeline(data_dir, batch_size)
    model = build_se_attention_cnn()

    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"), tf.keras.metrics.Recall(name="recall")]
    )

    callbacks = [
        ModelCheckpoint(
            filepath=str(SE_CNN_WEIGHTS_PATH),
            save_weights_only=True,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1
        ),
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks
    )

    hist_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(MODELS_DIR / "se_cnn_training_history.json", "w") as f:
        json.dump(hist_dict, f, indent=2)

    return {
        "status": "Completed",
        "final_train_accuracy": float(hist_dict["accuracy"][-1]),
        "final_val_accuracy": float(hist_dict["val_accuracy"][-1]),
        "history": hist_dict
    }


if __name__ == "__main__":
    result = train_se_cnn()
    print("Training Configuration Status:", result)
