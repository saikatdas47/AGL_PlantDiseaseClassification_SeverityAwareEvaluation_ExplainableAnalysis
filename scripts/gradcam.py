# =========================
# run_final_analysis_t4_optimized.py
# =========================

from google.colab import drive
drive.mount('/content/drive')

# =========================
# IMPORTS
# =========================
import os
import json
import time
import random
import shutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess
from sklearn.metrics import accuracy_score, f1_score

# =========================
# GPU SETUP
# =========================
print("TensorFlow version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices("GPU"))

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU memory growth enabled.")
    except RuntimeError as e:
        print("Memory growth setup warning:", e)

print("\n=== NVIDIA GPU INFO ===")
!nvidia-smi

# =========================
# CONFIG
# =========================
SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

DATA_ROOT = "/content/drive/MyDrive/CV/dataset"
TRAIN_DIR = os.path.join(DATA_ROOT, "train")
VAL_DIR   = os.path.join(DATA_ROOT, "val")
TEST_DIR  = os.path.join(DATA_ROOT, "test")

RESULTS_DIR = "/content/drive/MyDrive/CV/plant_results"
MODELS_DIR = os.path.join(RESULTS_DIR, "saved_models")
PLOTS_DIR  = os.path.join(RESULTS_DIR, "plots")
CSV_DIR    = os.path.join(RESULTS_DIR, "csv")
META_DIR   = os.path.join(RESULTS_DIR, "metadata")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(META_DIR, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
AUTOTUNE = tf.data.AUTOTUNE

# =========================
# CUSTOM LAYERS FOR CBAM MODELS
# =========================
class ChannelAvgPool(keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_mean(inputs, axis=-1, keepdims=True)

    def get_config(self):
        return super().get_config()


class ChannelMaxPool(keras.layers.Layer):
    def call(self, inputs):
        return tf.reduce_max(inputs, axis=-1, keepdims=True)

    def get_config(self):
        return super().get_config()


def load_saved_model(model_path):
    return keras.models.load_model(
        model_path,
        compile=False,
        safe_mode=False,
        custom_objects={
            "ChannelAvgPool": ChannelAvgPool,
            "ChannelMaxPool": ChannelMaxPool
        }
    )

# =========================
# PREPROCESS HELPERS
# =========================
def get_preprocess_fn(model_name):
    if model_name.startswith("MobileNetV3Small"):
        return mobilenet_preprocess
    elif model_name.startswith("EfficientNetB0"):
        return efficientnet_preprocess
    elif model_name.startswith("ResNet50"):
        return resnet_preprocess
    else:
        raise ValueError(f"Unknown model family in {model_name}")


def load_raw_image_array(img_path, target_size=IMG_SIZE):
    img = keras.utils.load_img(img_path, target_size=target_size)
    arr = keras.utils.img_to_array(img).astype(np.float32)
    return arr


def preprocess_single_image(img_path, model_name):
    arr = load_raw_image_array(img_path, target_size=IMG_SIZE)
    arr = np.expand_dims(arr, axis=0)
    arr = get_preprocess_fn(model_name)(arr)
    return arr


def preprocess_batch_image_paths(image_paths, model_name, batch_size=BATCH_SIZE):
    preprocess_fn = get_preprocess_fn(model_name)

    def gen():
        for p in image_paths:
            arr = load_raw_image_array(p, target_size=IMG_SIZE)
            yield arr

    ds = tf.data.Dataset.from_generator(
        gen,
        output_signature=tf.TensorSpec(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32)
    )

    ds = ds.batch(batch_size).map(lambda x: preprocess_fn(x), num_parallel_calls=AUTOTUNE)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def batched_predict_from_paths(model, model_name, image_paths, batch_size=BATCH_SIZE):
    ds = preprocess_batch_image_paths(image_paths, model_name, batch_size=batch_size)
    probs = model.predict(ds, verbose=0)
    pred_idx = np.argmax(probs, axis=1)
    return pred_idx, probs

# =========================
# LOAD METADATA
# =========================
class_names_path = os.path.join(META_DIR, "class_names.json")
test_index_path = os.path.join(CSV_DIR, "test_index.csv")

if not os.path.exists(class_names_path):
    raise FileNotFoundError(f"Missing: {class_names_path}")
if not os.path.exists(test_index_path):
    raise FileNotFoundError(f"Missing: {test_index_path}")

with open(class_names_path, "r") as f:
    class_names = json.load(f)

test_index_df = pd.read_csv(test_index_path)

# =========================
# LOAD SUMMARY CSVs
# =========================
summary_files = [
    os.path.join(CSV_DIR, "mobilenetv3small_summary.csv"),
    os.path.join(CSV_DIR, "efficientnetb0_summary.csv"),
    os.path.join(CSV_DIR, "resnet50_summary.csv"),
]

summary_parts = []
for fpath in summary_files:
    if os.path.exists(fpath):
        summary_parts.append(pd.read_csv(fpath))

if len(summary_parts) == 0:
    raise FileNotFoundError("No model summary CSV found. Run training scripts first.")

summary_df = pd.concat(summary_parts, ignore_index=True)
summary_df = summary_df.sort_values("F1_macro", ascending=False).reset_index(drop=True)
summary_df.to_csv(os.path.join(CSV_DIR, "all_models_summary.csv"), index=False)

print("\n=== Combined Summary ===")
print(summary_df)

# =========================
# EXPERIMENT GROUP 1 + 2
# =========================
baseline_models = ["MobileNetV3Small_baseline", "EfficientNetB0_baseline", "ResNet50_baseline"]
group1_df = summary_df[summary_df["Model"].isin(baseline_models)].copy()
group1_df.to_csv(os.path.join(CSV_DIR, "experiment_group1_baselines.csv"), index=False)

attention_models = ["MobileNetV3Small_SE", "MobileNetV3Small_CBAM", "EfficientNetB0_CBAM"]
group2_df = summary_df[summary_df["Model"].isin(attention_models)].copy()
group2_df.to_csv(os.path.join(CSV_DIR, "experiment_group2_attention.csv"), index=False)

# =========================
# EXPERIMENT GROUP 3
# SEVERITY-AWARE ANALYSIS
# =========================
def estimate_severity_from_image(img_bgr):
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    lower_green = np.array([25, 20, 20])
    upper_green = np.array([95, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    lower_yellow = np.array([10, 30, 30])
    upper_yellow = np.array([35, 255, 255])
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

    lower_brown = np.array([5, 20, 20])
    upper_brown = np.array([25, 255, 180])
    brown_mask = cv2.inRange(hsv, lower_brown, upper_brown)

    disease_mask = cv2.bitwise_or(yellow_mask, brown_mask)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, leaf_mask2 = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    leaf_mask = cv2.bitwise_or(green_mask, leaf_mask2)

    leaf_area = np.sum(leaf_mask > 0)
    disease_area = np.sum((disease_mask > 0) & (leaf_mask > 0))
    lesion_ratio = 0.0 if leaf_area == 0 else disease_area / leaf_area

    if lesion_ratio < 0.10:
        severity = "mild"
    elif lesion_ratio < 0.25:
        severity = "moderate"
    else:
        severity = "severe"

    return lesion_ratio, severity


severity_rows = []
for _, row in test_index_df.iterrows():
    img_path = row["image_path"]
    cls_name = row["class_name"]
    img_bgr = cv2.imread(img_path)

    if img_bgr is None:
        continue

    if cls_name.lower() == "healthy":
        lesion_ratio = 0.0
        severity = "healthy"
    else:
        lesion_ratio, severity = estimate_severity_from_image(img_bgr)

    severity_rows.append({
        "image_path": img_path,
        "true_class": cls_name,
        "true_idx": int(row["label_idx"]),
        "lesion_ratio": lesion_ratio,
        "severity": severity
    })

severity_df = pd.DataFrame(severity_rows)
severity_proxy_path = os.path.join(CSV_DIR, "test_severity_proxy.csv")
severity_df.to_csv(severity_proxy_path, index=False)

# Use best model from combined summary
best_model_name = summary_df.iloc[0]["Model"]
best_model_path = summary_df.iloc[0]["Model_Path"]

print("\nBest model for severity analysis:", best_model_name)
best_model = load_saved_model(best_model_path)

preds, probs = batched_predict_from_paths(
    best_model,
    best_model_name,
    severity_df["image_path"].tolist(),
    batch_size=BATCH_SIZE
)

severity_df["pred_idx"] = preds
severity_df["pred_class"] = [class_names[i] for i in preds]
severity_df["correct"] = (severity_df["true_idx"] == severity_df["pred_idx"]).astype(int)
severity_df.to_csv(os.path.join(CSV_DIR, f"{best_model_name}_severity_predictions.csv"), index=False)

sev_rows = []
for sev in ["mild", "moderate", "severe"]:
    sub = severity_df[severity_df["severity"] == sev].copy()
    if len(sub) == 0:
        continue

    y_true = sub["true_idx"].values
    y_pred = sub["pred_idx"].values

    sev_rows.append({
        "Severity": sev,
        "Count": len(sub),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Macro_F1": float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    })

severity_result_df = pd.DataFrame(sev_rows)
severity_result_df.to_csv(os.path.join(CSV_DIR, "experiment_group3_severity_results.csv"), index=False)

print("\n=== Severity Results ===")
print(severity_result_df)

# =========================
# EXPERIMENT GROUP 4
# GRAD-CAM FOR ALL MODELS
# =========================
def find_last_conv_layer(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("No Conv2D layer found.")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array, training=False)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def save_gradcam_overlay(img_path, model, model_name, out_path,
                         true_class=None, severity=None, alpha=0.4):
    img = keras.utils.load_img(img_path, target_size=IMG_SIZE)
    img_array = keras.utils.img_to_array(img)

    x = np.expand_dims(img_array.copy(), axis=0).astype(np.float32)
    x = get_preprocess_fn(model_name)(x)

    last_conv_layer_name = find_last_conv_layer(model)
    heatmap = make_gradcam_heatmap(x, model, last_conv_layer_name)

    img_uint8 = np.uint8(img_array)
    heatmap_resized = cv2.resize(heatmap, (img_uint8.shape[1], img_uint8.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

    base_bgr = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR)
    superimposed = cv2.addWeighted(base_bgr, 1 - alpha, heatmap_color, alpha, 0)

    pred = model.predict(x, verbose=0)[0]
    pred_idx = int(np.argmax(pred))
    pred_class = class_names[pred_idx]

    status = ""
    if true_class is not None:
        status = "Correct" if true_class == pred_class else "Incorrect"

    title_text = f"{model_name}\nTrue: {true_class} | Pred: {pred_class}"
    if status:
        title_text += f" | {status}"
    if severity is not None:
        title_text += f"\nSeverity: {severity}"

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.imshow(img_uint8.astype(np.uint8))
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB))
    plt.title(title_text)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()


gradcam_dir = os.path.join(PLOTS_DIR, "gradcam_samples")
os.makedirs(gradcam_dir, exist_ok=True)

# Optional clear old Grad-CAM files
for item in os.listdir(gradcam_dir):
    item_path = os.path.join(gradcam_dir, item)
    if os.path.isfile(item_path):
        os.remove(item_path)
    elif os.path.isdir(item_path):
        shutil.rmtree(item_path)

# Select Grad-CAM samples from severity groups only
selected_parts = []
severity_order = ["mild", "moderate", "severe"]
TOTAL_TARGET = 20
PER_SEVERITY_TARGET = max(1, TOTAL_TARGET // len(severity_order))

for sev in severity_order:
    sev_df = severity_df[severity_df["severity"] == sev].copy()
    if len(sev_df) == 0:
        continue

    chosen_df = sev_df.sample(
        n=min(PER_SEVERITY_TARGET, len(sev_df)),
        random_state=SEED
    )
    selected_parts.append(chosen_df)

sample_df = pd.concat(selected_parts, ignore_index=True) if len(selected_parts) > 0 else pd.DataFrame()

if len(sample_df) < TOTAL_TARGET:
    remaining_pool = severity_df[severity_df["severity"].isin(severity_order)].drop_duplicates(subset=["image_path"])
    if len(sample_df) > 0:
        remaining_pool = remaining_pool[~remaining_pool["image_path"].isin(sample_df["image_path"])]

    if len(remaining_pool) > 0:
        extra_n = min(TOTAL_TARGET - len(sample_df), len(remaining_pool))
        extra_df = remaining_pool.sample(n=extra_n, random_state=SEED)
        sample_df = pd.concat([sample_df, extra_df], ignore_index=True)

sample_df = sample_df.head(TOTAL_TARGET).copy()

print(f"\nTotal Grad-CAM base samples selected: {len(sample_df)}")
if len(sample_df) > 0:
    print(sample_df["severity"].value_counts(dropna=False))

gradcam_rows = []

# Keep all models
gradcam_model_list = summary_df["Model"].tolist()

for _, mrow in summary_df.iterrows():
    model_name = mrow["Model"]
    if model_name not in gradcam_model_list:
        continue

    model_path = mrow["Model_Path"]
    print(f"\nGenerating Grad-CAM for: {model_name}")
    model = load_saved_model(model_path)

    model_gradcam_dir = os.path.join(gradcam_dir, model_name)
    os.makedirs(model_gradcam_dir, exist_ok=True)

    # Batched predictions first for chosen sample set
    sample_image_paths = sample_df["image_path"].tolist()
    pred_idx_arr, _ = batched_predict_from_paths(
        model, model_name, sample_image_paths, batch_size=BATCH_SIZE
    )

    model_pred_rows = []
    for idx_row, (_, row) in enumerate(sample_df.iterrows()):
        pred_idx = int(pred_idx_arr[idx_row])
        pred_class = class_names[pred_idx]
        is_correct = int(pred_idx == int(row["true_idx"]))

        model_pred_rows.append({
            "image_path": row["image_path"],
            "true_class": row["true_class"],
            "true_idx": int(row["true_idx"]),
            "pred_idx": pred_idx,
            "pred_class": pred_class,
            "severity": row["severity"],
            "correct": is_correct
        })

    model_pred_df = pd.DataFrame(model_pred_rows)
    model_pred_df.to_csv(
        os.path.join(CSV_DIR, f"{model_name}_gradcam_sample_predictions.csv"),
        index=False
    )

    for i, prow in model_pred_df.iterrows():
        img_path = prow["image_path"]
        fname = os.path.basename(img_path)
        correctness_tag = "correct" if int(prow["correct"]) == 1 else "wrong"

        out_path = os.path.join(
            model_gradcam_dir,
            f"{model_name}_{prow['severity']}_{correctness_tag}_{i}_{fname}.png"
        )

        save_gradcam_overlay(
            img_path=img_path,
            model=model,
            model_name=model_name,
            out_path=out_path,
            true_class=prow["true_class"],
            severity=prow["severity"]
        )

        gradcam_rows.append({
            "model_name": model_name,
            "image_path": img_path,
            "output_path": out_path,
            "true_class": prow["true_class"],
            "pred_class": prow["pred_class"],
            "severity": prow["severity"],
            "correct": int(prow["correct"])
        })

gradcam_df = pd.DataFrame(gradcam_rows)
gradcam_df.to_csv(os.path.join(CSV_DIR, "experiment_group4_gradcam_files.csv"), index=False)

print("\nSaved Grad-CAM file summary:")
if len(gradcam_df) > 0:
    print(gradcam_df[["model_name", "true_class", "pred_class", "severity", "correct", "output_path"]].head(50))
else:
    print("No Grad-CAM files saved.")

# =========================
# EXPERIMENT GROUP 5
# EFFICIENCY
# =========================
def measure_inference_time(model, model_name, sample_paths, warmup_batches=2, batch_size=BATCH_SIZE):
    if len(sample_paths) == 0:
        return np.nan

    ds = preprocess_batch_image_paths(sample_paths, model_name, batch_size=batch_size)

    # Warmup
    for i, batch in enumerate(ds.take(warmup_batches)):
        _ = model.predict(batch, verbose=0)

    # Actual timing
    n_images = 0
    start = time.time()
    for batch in ds:
        preds = model.predict(batch, verbose=0)
        n_images += preds.shape[0]
    end = time.time()

    return (end - start) / max(n_images, 1)


eff_rows = []

# Use subset for fair timing if test set is large
timing_paths = test_index_df["image_path"].tolist()
timing_paths = timing_paths[:min(len(timing_paths), 256)]

for _, row in summary_df.iterrows():
    model_name = row["Model"]
    model_path = row["Model_Path"]
    model = load_saved_model(model_path)

    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    avg_inf_time = measure_inference_time(
        model,
        model_name,
        timing_paths,
        warmup_batches=2,
        batch_size=BATCH_SIZE
    )

    eff_rows.append({
        "Model": model_name,
        "Parameters_M": float(model.count_params() / 1e6),
        "Model_Size_MB": float(model_size_mb),
        "Avg_Inference_Time_sec_per_image": float(avg_inf_time)
    })

eff_df = pd.DataFrame(eff_rows).sort_values("Avg_Inference_Time_sec_per_image")
eff_df.to_csv(os.path.join(CSV_DIR, "experiment_group5_efficiency.csv"), index=False)

print("\n=== Efficiency ===")
print(eff_df)

# =========================
# FINAL MERGE
# =========================
final_df = summary_df.merge(eff_df, on="Model", how="left")
final_df.to_csv(os.path.join(CSV_DIR, "final_experiment_summary.csv"), index=False)

print("\n=== Final Summary ===")
print(final_df)
print("\nAll important files saved in:", RESULTS_DIR)