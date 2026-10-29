"""
Explainable AI Grad-CAM Subsystem for Plant Pathology (Module F).
Generates Gradient-Weighted Class Activation Maps overlaying leaf lesions.
Targets the last Squeeze-and-Excitation convolution block (Figure 4.1).
"""

import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from typing import Dict, Any, Optional, Tuple, Union
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf

from config.settings import CNN_INPUT_SHAPE, SE_CNN_WEIGHTS_PATH
from src.disease.preprocessing import load_and_preprocess_image
from src.disease.model import build_se_attention_cnn


def make_gradcam_heatmap(
    img_array: np.ndarray,
    model: tf.keras.Model,
    last_conv_layer_name: str = "last_se_conv",
    pred_index: Optional[int] = None
) -> np.ndarray:
    """
    Computes Gradient-weighted Class Activation Map (Grad-CAM):
    1. Obtains feature activation maps A^k from the target conv layer.
    2. Computes gradient of top class score y^c with respect to A^k.
    3. Global average pools gradients to calculate neuron importance weights alpha_k^c.
    4. Computes ReLU(sum_k alpha_k^c * A^k) to produce 2D attention heatmap.
    """
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy()


def overlay_gradcam(
    original_pil_img: Image.Image,
    heatmap: np.ndarray,
    alpha: float = 0.45,
    colormap: int = cv2.COLORMAP_JET
) -> Image.Image:
    """Overlays colorized Grad-CAM heatmap onto the original leaf image."""
    w, h = original_pil_img.size
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    color_heatmap_rgb = cv2.cvtColor(color_heatmap, cv2.COLOR_BGR2RGB)

    orig_np = np.array(original_pil_img.convert("RGB"))
    blended = np.uint8(orig_np * (1 - alpha) + color_heatmap_rgb * alpha)
    return Image.fromarray(blended)


def generate_gradcam_explanation(
    image_input: Union[str, bytes, Image.Image],
    target_class_idx: Optional[int] = None
) -> Dict[str, Any]:
    """
    Main interface for Grad-CAM generation.
    Checks model readiness and returns heatmap overlay or honest milestone status.
    """
    prep = load_and_preprocess_image(image_input)
    if not prep["valid"]:
        return {
            "available": False,
            "error": prep["error"],
            "status": "Preprocessing Error"
        }

    # If SE-CNN weights exist:
    if SE_CNN_WEIGHTS_PATH.exists():
        try:
            model = build_se_attention_cnn()
            model.load_weights(str(SE_CNN_WEIGHTS_PATH))

            heatmap = make_gradcam_heatmap(
                prep["batch_array"],
                model,
                last_conv_layer_name="last_se_conv",
                pred_index=target_class_idx
            )
            overlay_img = overlay_gradcam(prep["display_image"], heatmap)

            return {
                "available": True,
                "status": "Generated from Trained Attention-CNN",
                "target_layer": "last_se_conv (Block 3 SE Attention)",
                "overlay_image": overlay_img,
                "raw_heatmap": heatmap,
                "math_formulation": "L_{Grad-CAM}^c = ReLU(sum_k alpha_k^c * A^k)"
            }
        except Exception as e:
            return {
                "available": False,
                "error": str(e),
                "status": "Grad-CAM Computation Exception"
            }

    # Current Week 6 milestone status:
    return {
        "available": False,
        "status": "Pending Week 7-9 Milestone Integration",
        "message": (
            "Grad-CAM engine architecture is connected to layer 'last_se_conv' of Figure 4.1. "
            "Full activation heatmaps will be generated once Week 7 model training completes."
        ),
        "target_layer": "last_se_conv (Squeeze-and-Excitation Block 3)",
        "mathematical_foundation": {
            "gradient_formula": "alpha_k^c = (1/Z) * sum_i sum_j (d y^c / d A_ij^k)",
            "activation_formula": "L_{Grad-CAM}^c = ReLU( sum_k alpha_k^c * A^k )",
            "clinical_utility": "Validates whether the CNN focused on diseased necrotic spots rather than background soil/sky."
        }
    }
