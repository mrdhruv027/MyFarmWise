"""
Attention-Based CNN Architecture with Squeeze-and-Excitation (SE) Identity Blocks.
Directly implements Figure 4.1 and Section 4.4 of the MyFarmWise Project Synopsis.
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import tensorflow as tf
from tensorflow.keras import layers, models, Model
from typing import Tuple, Dict, Any

from config.settings import CNN_INPUT_SHAPE, SE_REDUCTION_RATIO, NUM_DISEASE_CLASSES


def squeeze_and_excitation_block(
    input_tensor: tf.Tensor,
    filters: int,
    reduction_ratio: int = SE_REDUCTION_RATIO,
    block_name: str = "se_block"
) -> tf.Tensor:
    """
    Squeeze-and-Excitation (SE) Channel-Wise Feature Recalibration block.
    1. Squeeze: Global Average Pooling compresses spatial dimensions HxW into 1x1xC channel descriptor.
    2. Excitation: Two Dense layers with ReLU then Sigmoid capture non-linear channel interdependencies.
    3. Scale: Element-wise multiply channel excitation weights with original feature map.
    4. Residual: Add input tensor back to the calibrated features.
    """
    # 1. Squeeze
    squeeze = layers.GlobalAveragePooling2D(name=f"{block_name}_squeeze_gap")(input_tensor)
    squeeze = layers.Reshape((1, 1, filters), name=f"{block_name}_reshape")(squeeze)

    # 2. Excitation
    reduced_filters = max(filters // reduction_ratio, 4)
    excitation = layers.Dense(
        reduced_filters,
        activation="relu",
        use_bias=False,
        name=f"{block_name}_excite_relu"
    )(squeeze)
    excitation = layers.Dense(
        filters,
        activation="sigmoid",
        use_bias=False,
        name=f"{block_name}_excite_sigmoid"
    )(excitation)

    # 3. Channel-wise Scale
    scaled = layers.Multiply(name=f"{block_name}_scale")([input_tensor, excitation])

    # 4. Identity / Residual Connection
    output = layers.Add(name=f"{block_name}_residual_add")([input_tensor, scaled])
    return output


def build_se_attention_cnn(
    input_shape: Tuple[int, int, int] = CNN_INPUT_SHAPE,
    num_classes: int = NUM_DISEASE_CLASSES,
    reduction_ratio: int = SE_REDUCTION_RATIO
) -> Model:
    """
    Constructs the 38-class Attention-CNN specified in Figure 4.1:
    - 224x224x3 Input
    - Conv Stem (32 filters 3x3, BatchNorm, ReLU, MaxPool 2x2)
    - SE Identity Block 1 (64 filters 3x3, SE attention, MaxPool 2x2)
    - SE Identity Block 2 (128 filters 3x3, SE attention, MaxPool 2x2)
    - SE Identity Block 3 (256 filters 3x3, SE attention -> Grad-CAM target layer)
    - Global Average Pooling (256-d feature vector)
    - Dense Head (128 units, ReLU, Dropout 0.5, 38-way Softmax output)
    """
    inputs = layers.Input(shape=input_shape, name="leaf_image_input")

    # 1. Conv Stem: 32 filters (3x3), ReLU + MaxPool
    x = layers.Conv2D(32, (3, 3), padding="same", name="stem_conv")(inputs)
    x = layers.BatchNormalization(name="stem_bn")(x)
    x = layers.ReLU(name="stem_relu")(x)
    x = layers.MaxPooling2D((2, 2), name="stem_maxpool")(x)

    # 2. SE Identity Block 1: 64 filters
    x = layers.Conv2D(64, (3, 3), padding="same", name="block1_conv")(x)
    x = layers.BatchNormalization(name="block1_bn")(x)
    x = layers.ReLU(name="block1_relu")(x)
    x = squeeze_and_excitation_block(x, 64, reduction_ratio, block_name="se_block_1")
    x = layers.MaxPooling2D((2, 2), name="block1_maxpool")(x)

    # 3. SE Identity Block 2: 128 filters
    x = layers.Conv2D(128, (3, 3), padding="same", name="block2_conv")(x)
    x = layers.BatchNormalization(name="block2_bn")(x)
    x = layers.ReLU(name="block2_relu")(x)
    x = squeeze_and_excitation_block(x, 128, reduction_ratio, block_name="se_block_2")
    x = layers.MaxPooling2D((2, 2), name="block2_maxpool")(x)

    # 4. SE Identity Block 3: 256 filters (Target Layer for Grad-CAM)
    x = layers.Conv2D(256, (3, 3), padding="same", name="last_se_conv")(x)
    x = layers.BatchNormalization(name="block3_bn")(x)
    x = layers.ReLU(name="block3_relu")(x)
    x = squeeze_and_excitation_block(x, 256, reduction_ratio, block_name="last_se_block")

    # 5. Global Average Pooling (GAP): 256-dimensional feature representation
    gap = layers.GlobalAveragePooling2D(name="global_avg_pooling")(x)

    # 6. Dense Classification Head
    dense = layers.Dense(128, activation="relu", name="head_dense_128")(gap)
    dropout = layers.Dropout(0.5, name="head_dropout")(dense)
    outputs = layers.Dense(num_classes, activation="softmax", name="disease_softmax_output")(dropout)

    model = Model(inputs=inputs, outputs=outputs, name="MyFarmWise_SE_Attention_CNN")
    return model


def get_model_architecture_summary() -> Dict[str, Any]:
    """Generates a structured dictionary detailing the SE-CNN architecture for reporting."""
    model = build_se_attention_cnn()
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    non_trainable_params = total_params - trainable_params

    return {
        "model_name": model.name,
        "input_shape": CNN_INPUT_SHAPE,
        "num_classes": NUM_DISEASE_CLASSES,
        "total_parameters": int(total_params),
        "trainable_parameters": int(trainable_params),
        "non_trainable_parameters": int(non_trainable_params),
        "gradcam_target_layer": "last_se_conv",
        "se_blocks_count": 3,
        "stages": [
            {"stage": "Input", "output_dim": "224 x 224 x 3", "details": "RGB leaf image"},
            {"stage": "Stem", "output_dim": "112 x 112 x 32", "details": "Conv2D (32, 3x3) + BN + ReLU + MaxPool"},
            {"stage": "SE Block 1", "output_dim": "56 x 56 x 64", "details": "Conv2D (64, 3x3) + SE Attention (r=16) + MaxPool"},
            {"stage": "SE Block 2", "output_dim": "28 x 28 x 128", "details": "Conv2D (128, 3x3) + SE Attention (r=16) + MaxPool"},
            {"stage": "SE Block 3", "output_dim": "28 x 28 x 256", "details": "Conv2D (256, 3x3) + SE Attention (r=16) [Grad-CAM Layer]"},
            {"stage": "GAP", "output_dim": "256", "details": "Global Average Pooling feature vector"},
            {"stage": "Head", "output_dim": "38", "details": "Dense (128) + Dropout (0.5) + Softmax (38 classes)"}
        ]
    }


if __name__ == "__main__":
    summary = get_model_architecture_summary()
    print("=== SE-ATTENTION CNN ARCHITECTURE SPECIFICATION ===")
    print(f"Model: {summary['model_name']}")
    print(f"Input: {summary['input_shape']}")
    print(f"Classes: {summary['num_classes']}")
    print(f"Total Parameters: {summary['total_parameters']:,}")
    print(f"Trainable Parameters: {summary['trainable_parameters']:,}")
    print(f"Grad-CAM Target Layer: {summary['gradcam_target_layer']}")
