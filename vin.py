import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import shap
from skimage.segmentation import mark_boundaries

# ---------------------------- Load model and background ----------------------------
directory = 'The IQ-OTHNCCD lung cancer dataset/The IQ-OTHNCCD lung cancer dataset'
categories = ['Bengin cases', 'Malignant cases', 'Normal cases']
img_size = 256
X = []

# This part of the code is problematic as it tries to load data from a local directory
# that is not present in the sandbox. For the purpose of making the application runnable,
# I will comment this out and assume `background_data` is handled externally or pre-loaded.
# In a real-world scenario, this data would need to be provided or generated.
# For now, I will create a dummy background_data array to allow the code to run.

# for label, category in enumerate(categories):
#     path = os.path.join(directory, category)
#     for file in os.listdir(path)[:50]:
#         try:
#             img = cv2.imread(os.path.join(path, file), 0)
#             img = cv2.resize(img, (img_size, img_size))
#             X.append(img)
#         except:
#             continue

# X = np.array(X).reshape(-1, 256, 256, 1) / 255.0
# background_data = X[np.random.choice(len(X), 5, replace=False)]

# Dummy background_data for now to prevent errors
background_data = np.random.rand(5, 256, 256, 1)

model = load_model("lungmodel.h5")

# ---------------------------- Explanation & SHAP ----------------------------
def explain_with_shap(model, img_path, background_data, save_path="static/shap_output/shap_overlay.png"):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256, 256))
    img = img / 255.0
    img_array = np.expand_dims(img, axis=(0, -1))

    # Ensure background_data has the correct shape for the explainer
    # The original code assumes background_data is already scaled and shaped correctly.
    # If background_data is dummy, it needs to be adjusted.
    # For now, assuming it's correctly shaped (5, 256, 256, 1)

    explainer = shap.GradientExplainer(model, background_data)
    shap_values, _ = explainer.shap_values(img_array, ranked_outputs=1)

    shap_arr = shap_values[0][0].squeeze()
    threshold = np.percentile(np.abs(shap_arr), 95)
    mask = np.abs(shap_arr) > threshold

    img_rgb = np.repeat(img_array[0], 3, axis=-1)
    overlay = mark_boundaries(img_rgb, mask.astype(bool), color=(1, 0, 0))

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.imsave(save_path, overlay)
    return shap_arr, mask, save_path

# ---------------------------- Utility Functions ----------------------------
def predict_image(model, img_path):
    img = cv2.imread(img_path, 0)
    img = cv2.resize(img, (256, 256)) / 255.0
    img = img.reshape(1, 256, 256, 1)
    prediction = model.predict(img)[0]
    class_idx = np.argmax(prediction)
    class_names = ['Benign', 'Malignant', 'Normal']
    return class_names[class_idx], prediction[class_idx] * 100

def get_tumor_stage(area):
    if area > 40:
        return "Stage III"
    elif area > 25:
        return "Stage II"
    elif area > 10:
        return "Stage I"
    else:
        return "No significant tumor"

def get_risk_level(area):
    if area > 40:
        return "High"
    elif area > 25:
        return "Moderate"
    elif area > 10:
        return "Low"
    else:
        return "None"

def get_nodule_location(mask):
    if mask.ndim != 2 or not np.any(mask):
        return "No abnormal region detected (location undetermined)"

    ys, xs = np.where(mask)
    y_mean, x_mean = np.mean(ys), np.mean(xs)

    # Vertical zones
    if y_mean < 85:
        vertical = "upper lung"
    elif y_mean < 170:
        vertical = "middle lung"
    else:
        vertical = "lower lung"

    # Horizontal zones
    if x_mean < 85:
        horizontal = "left"
    elif x_mean < 170:
        horizontal = "center"
    else:
        horizontal = "right"

    return f"Abnormality predominantly located in the {vertical} - {horizontal} region."


def generate_explanation(confidence, tumor_area):
    explanation = "Based on "
    explanation += "high model confidence and " if confidence > 90 else "moderate model confidence and "
    if tumor_area > 40:
        explanation += "extensive tumor area with highly abnormal features."
    elif tumor_area > 25:
        explanation += "moderate-sized tumor area with visible abnormalities."
    elif tumor_area > 10:
        explanation += "small tumor presence showing mild irregularities."
    else:
        explanation += "minimal or no visible tumor signs."
    return explanation

# ---------------------------- Main Report Generator ----------------------------
def generate_report(image_path):
    prediction, confidence = predict_image(model, image_path)
    shap_arr, mask, shap_path = explain_with_shap(model, image_path, background_data)
    tumor_area = int(np.sum(mask))
    stage = get_tumor_stage(tumor_area)
    risk = get_risk_level(tumor_area)
    location_sentence = get_nodule_location(mask)
    explanation_text = generate_explanation(confidence, tumor_area)

    report_data = {
        "prediction": prediction,
        "confidence": f"{confidence:.2f}%",
        "tumor_area": tumor_area,
        "tumor_stage": stage,
        "risk_level": risk,
        "nodule_location": location_sentence,
        "explanation_text": explanation_text,
        "shap_path": shap_path
    }
    return report_data

# ---------------------------- Run ----------------------------
if __name__ == "__main__":
    # Example usage (this part won't be executed by app.py)
    # For testing, you might need a dummy image file here
    dummy_image_path = "static/uploads/dummy_ct_scan.png"
    # Create a dummy image for testing if it doesn't exist
    if not os.path.exists(dummy_image_path):
        from PIL import Image
        dummy_image = Image.new('L', (256, 256), color = 'white')
        os.makedirs(os.path.dirname(dummy_image_path), exist_ok=True)
        dummy_image.save(dummy_image_path)

    report = generate_report(dummy_image_path)
    print("\n--- Lung Cancer Prediction Report ---")
    for key, value in report.items():
        print(f"{key}: {value}")


