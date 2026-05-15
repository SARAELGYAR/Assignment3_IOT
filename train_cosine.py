"""
SWAPD 453 - Assignment 3: TinyML Cosine Wave Predictor
Train a small NN to approximate y = cos(x), quantize to int8, export for ESP32.
"""

import os
import subprocess
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras

SEED = 42
NUM_SAMPLES = 1000
NOISE_STD = 0.1
EPOCHS = 200
BATCH_SIZE = 32
REP_DATASET_SIZE = 100

PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)


def set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)


def generate_dataset(n: int = NUM_SAMPLES):
    """x ~ U[0, 2pi], y = cos(x) + N(0, 0.1)."""
    x = np.random.uniform(0.0, 2.0 * np.pi, size=(n, 1)).astype(np.float32)
    noise = np.random.normal(0.0, NOISE_STD, size=(n, 1)).astype(np.float32)
    y = np.cos(x) + noise
    return x, y


def split_dataset(x, y, train_frac=0.6, val_frac=0.2):
    n = len(x)
    indices = np.random.permutation(n)
    x, y = x[indices], y[indices]

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    x_train, y_train = x[:n_train], y[:n_train]
    x_val, y_val = x[n_train : n_train + n_val], y[n_train : n_train + n_val]
    x_test, y_test = x[n_train + n_val :], y[n_train + n_val :]
    return (x_train, y_train), (x_val, y_val), (x_test, y_test)


def build_model() -> keras.Model:
    return keras.Sequential(
        [
            keras.layers.Dense(16, activation="relu", input_shape=(1,)),
            keras.layers.Dense(16, activation="relu"),
            keras.layers.Dense(1),
        ]
    )


def evaluate_mse_mae(model_or_fn, x_test, y_test):
    if callable(model_or_fn):
        pred = model_or_fn(x_test)
    else:
        pred = model_or_fn.predict(x_test, verbose=0)
    pred = np.asarray(pred, dtype=np.float32)
    mse = float(np.mean((pred - y_test) ** 2))
    mae = float(np.mean(np.abs(pred - y_test)))
    return mse, mae, pred


def plot_loss(history):
    plt.figure(figsize=(8, 5))
    plt.plot(history.history["loss"], label="Training loss")
    plt.plot(history.history["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "loss_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def plot_float_predictions(x_dense, y_true, y_pred):
    plt.figure(figsize=(10, 5))
    plt.plot(x_dense, y_true, label="Ground truth cos(x)", linewidth=2)
    plt.plot(x_dense, y_pred, label="Float model prediction", linestyle="--")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Float model vs ground truth cosine")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "float_prediction.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def export_float_tflite(model, path="cosine_float.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved {path} ({len(tflite_model)} bytes)")
    return tflite_model


def representative_dataset_gen(x_train):
    def generator():
        for i in range(min(REP_DATASET_SIZE, len(x_train))):
            yield [x_train[i : i + 1].astype(np.float32)]

    return generator


def export_int8_tflite(model, x_train, path="cosine_int8.tflite"):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen(x_train)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tflite_model = converter.convert()
    with open(path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved {path} ({len(tflite_model)} bytes)")
    return tflite_model


def run_tflite_int8(tflite_path, x_test):
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_scale, in_zp = input_details[0]["quantization"]
    out_scale, out_zp = output_details[0]["quantization"]

    preds = []
    for i in range(len(x_test)):
        x_val = float(x_test[i, 0])
        x_q = np.clip(np.round(x_val / in_scale + in_zp), -128, 127).astype(np.int8)
        interpreter.set_tensor(input_details[0]["index"], np.array([[x_q]], dtype=np.int8))
        interpreter.invoke()
        out_q = interpreter.get_tensor(output_details[0]["index"])[0, 0]
        preds.append((float(out_q) - out_zp) * out_scale)

    return np.array(preds, dtype=np.float32).reshape(-1, 1), (in_scale, in_zp), (
        out_scale,
        out_zp,
    )


def plot_three_curves(x_dense, y_true, y_float, y_int8):
    plt.figure(figsize=(10, 5))
    plt.plot(x_dense, y_true, label="Ground truth cos(x)", linewidth=2)
    plt.plot(x_dense, y_float, label="Float TFLite prediction", linestyle="--")
    plt.plot(x_dense, y_int8, label="Int8 quantized prediction", linestyle=":")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Ground truth vs float vs int8 predictions")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "int8_vs_float.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved {path}")


def tflite_to_model_h(tflite_path="cosine_int8.tflite", header_path="model.h"):
    """Convert .tflite to C header with g_model[] (xxd if available, else Python)."""
    with open(tflite_path, "rb") as f:
        data = f.read()

    array_lines = None
    try:
        result = subprocess.run(
            ["xxd", "-i", tflite_path],
            check=True,
            capture_output=True,
            text=True,
        )
        array_lines = [ln for ln in result.stdout.splitlines() if "0x" in ln]
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

    if array_lines:
        header = [
            "#ifndef MODEL_H",
            "#define MODEL_H",
            "",
            "#include <stdint.h>",
            "",
            "alignas(8) const unsigned char g_model[] = {",
        ]
        for ln in array_lines:
            header.append("  " + ln.strip().rstrip(","))
        header[-1] = header[-1] + ","
        header[-1] = header[-1].rstrip(",")
        header.extend(["};", "", f"const unsigned int g_model_len = {len(data)};", "", "#endif"])
    else:
        bytes_per_line = 12
        lines = []
        for i in range(0, len(data), bytes_per_line):
            chunk = data[i : i + bytes_per_line]
            lines.append("  " + ", ".join(f"0x{b:02x}" for b in chunk))
        body = ",\n".join(lines)
        header_text = f"""#ifndef MODEL_H
#define MODEL_H

#include <stdint.h>

alignas(8) const unsigned char g_model[] = {{
{body}
}};

const unsigned int g_model_len = {len(data)};

#endif  // MODEL_H
"""
        with open(header_path, "w", encoding="utf-8") as f:
            f.write(header_text)
        print(f"Saved {header_path} ({len(data)} bytes in g_model)")
        return

    with open(header_path, "w", encoding="utf-8") as f:
        f.write("\n".join(header) + "\n")
    print(f"Saved {header_path} ({len(data)} bytes in g_model)")


def write_results_summary(
    model,
    test_mse,
    test_mae,
    y_clean_mae,
    int8_mse,
    int8_mae,
    float_size,
    int8_size,
    in_q,
    out_q,
):
    """Write metrics for the PDF report (results.txt)."""
    trainable = int(np.sum([np.prod(v.shape) for v in model.trainable_weights]))
    lines = [
        "SWAPD 453 - Assignment 3 Results",
        "================================",
        f"TensorFlow version: {tf.__version__}",
        f"Random seed: {SEED}",
        f"Samples: {NUM_SAMPLES} (60/20/20 split), noise sigma: {NOISE_STD}",
        f"Epochs: {EPOCHS}, batch size: {BATCH_SIZE}",
        f"Trainable parameters: {trainable}",
        "",
        "Float Keras (test set, noisy labels):",
        f"  MSE = {test_mse:.6f}",
        f"  MAE = {test_mae:.6f}",
        f"Float Keras vs clean cos(x) on test x:",
        f"  MAE = {y_clean_mae:.6f}",
        "",
        "TFLite file sizes:",
        f"  cosine_float.tflite = {float_size} bytes",
        f"  cosine_int8.tflite  = {int8_size} bytes",
        f"  Reduction = {100.0 * (1.0 - int8_size / float_size):.1f}%",
        "",
        "Int8 TFLite (test set, noisy labels):",
        f"  MSE = {int8_mse:.6f}",
        f"  MAE = {int8_mae:.6f}",
        "",
        "Quantization (use in ESP32 sketch / report):",
        f"  Input  scale = {in_q[0]}, zero_point = {in_q[1]}",
        f"  Output scale = {out_q[0]}, zero_point = {out_q[1]}",
        "",
        "Plots: plots/loss_curves.png, float_prediction.png, int8_vs_float.png",
    ]
    text = "\n".join(lines) + "\n"
    with open("results.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(text)


def export_artifacts(model, x_train, x_test, y_test, history=None):
    """TFLite conversion, plots, and model.h (after training)."""
    if history is not None:
        plot_loss(history)

    test_mse, test_mae, y_pred_test = evaluate_mse_mae(model, x_test, y_test)
    y_clean = np.cos(x_test)
    y_clean_mae = float(np.mean(np.abs(y_pred_test - y_clean)))
    print(f"Float Keras test MSE: {test_mse:.6f}, MAE: {test_mae:.6f}")
    print(f"Float Keras vs clean cos(x) on test x: MAE = {y_clean_mae:.6f}")

    x_dense = np.linspace(0, 2 * np.pi, 500, dtype=np.float32).reshape(-1, 1)
    y_true_dense = np.cos(x_dense)
    _, _, y_pred_dense = evaluate_mse_mae(model, x_dense, y_true_dense)
    plot_float_predictions(
        x_dense.flatten(), y_true_dense.flatten(), y_pred_dense.flatten()
    )

    float_bytes = export_float_tflite(model)
    int8_bytes = export_int8_tflite(model, x_train)
    reduction = 100.0 * (1.0 - len(int8_bytes) / len(float_bytes))
    print(
        f"Size reduction (float -> int8): {len(float_bytes)} -> {len(int8_bytes)} bytes "
        f"({reduction:.1f}% smaller)"
    )

    y_int8_test, in_q, out_q = run_tflite_int8("cosine_int8.tflite", x_test)
    int8_mse, int8_mae = (
        float(np.mean((y_int8_test - y_test) ** 2)),
        float(np.mean(np.abs(y_int8_test - y_test))),
    )
    print(f"Int8 TFLite test MSE: {int8_mse:.6f}, MAE: {int8_mae:.6f}")
    print(f"Input quant: scale={in_q[0]}, zero_point={in_q[1]}")
    print(f"Output quant: scale={out_q[0]}, zero_point={out_q[1]}")

    interpreter = tf.lite.Interpreter(model_content=int8_bytes)
    interpreter.allocate_tensors()
    in_details = interpreter.get_input_details()
    out_details = interpreter.get_output_details()
    in_scale, in_zp = in_details[0]["quantization"]
    out_scale, out_zp = out_details[0]["quantization"]

    y_int8_dense = []
    for i in range(len(x_dense)):
        x_val = float(x_dense[i, 0])
        x_q = int(np.clip(np.round(x_val / in_scale + in_zp), -128, 127))
        interpreter.set_tensor(in_details[0]["index"], np.array([[x_q]], dtype=np.int8))
        interpreter.invoke()
        out_q_val = interpreter.get_tensor(out_details[0]["index"])[0, 0]
        y_int8_dense.append((float(out_q_val) - out_zp) * out_scale)

    float_interpreter = tf.lite.Interpreter(model_content=float_bytes)
    float_interpreter.allocate_tensors()
    fin = float_interpreter.get_input_details()
    fout = float_interpreter.get_output_details()
    y_float_dense = []
    for i in range(len(x_dense)):
        float_interpreter.set_tensor(fin[0]["index"], x_dense[i : i + 1])
        float_interpreter.invoke()
        y_float_dense.append(float_interpreter.get_tensor(fout[0]["index"])[0, 0])

    plot_three_curves(
        x_dense.flatten(),
        y_true_dense.flatten(),
        np.array(y_float_dense),
        np.array(y_int8_dense),
    )

    tflite_to_model_h()
    sketch_header = os.path.join("cosine_predictor", "model.h")
    with open("model.h", "r", encoding="utf-8") as src:
        with open(sketch_header, "w", encoding="utf-8") as dst:
            dst.write(src.read())
    print(f"Copied model.h -> {sketch_header}")

    write_results_summary(
        model,
        test_mse,
        test_mae,
        y_clean_mae,
        int8_mse,
        int8_mae,
        len(float_bytes),
        len(int8_bytes),
        in_q,
        out_q,
    )


def main():
    set_seeds()
    print(f"TensorFlow {tf.__version__}")

    x, y = generate_dataset()
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = split_dataset(x, y)
    print(f"Train: {len(x_train)}, Val: {len(x_val)}, Test: {len(x_test)}")

    model = build_model()
    model.compile(optimizer="adam", loss="mse")
    model.summary()

    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
    )

    model.save_weights("cosine_weights.keras")
    export_artifacts(model, x_train, x_test, y_test, history=history)
    print("\nDone. Flash cosine_predictor.ino to the ESP32.")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--export-only":
        set_seeds()
        x, y = generate_dataset()
        (x_train, y_train), (_, y_val), (x_test, y_test) = split_dataset(x, y)
        model = build_model()
        model.compile(optimizer="adam", loss="mse")
        model.load_weights("cosine_weights.keras")
        export_artifacts(model, x_train, x_test, y_test)
        print("\nExport complete.")
    else:
        main()
