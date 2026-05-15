/*
 * SWAPD 453 - Assignment 3: TinyML Cosine Wave Predictor (ESP32)
 *
 * Library: Chirale_TensorFlowLite (Library Manager) OR TensorFlowLite_ESP32
 * Board: ESP32 Dev Module, 115200 baud Serial Plotter
 *
 * Place model.h in this folder (same directory as the .ino sketch).
 */

#include <Arduino.h>
#include <math.h>

#include "model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

constexpr int kTensorArenaSize = 4096;
constexpr int kStepsPerPeriod = 80;
constexpr float kTwoPi = 6.283185307f;

uint8_t tensor_arena[kTensorArenaSize];

float x_value = 0.0f;
const float kStep = kTwoPi / static_cast<float>(kStepsPerPeriod);

float input_scale = 1.0f;
int32_t input_zero_point = 0;
float output_scale = 1.0f;
int32_t output_zero_point = 0;

}  // namespace

tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input_tensor = nullptr;
TfLiteTensor* output_tensor = nullptr;

void setup() {
  Serial.begin(115200);
  delay(500);

  static tflite::MicroMutableOpResolver<2> resolver;
  if (resolver.AddFullyConnected() != kTfLiteOk ||
      resolver.AddRelu() != kTfLiteOk) {
    Serial.println("Failed to register ops");
    while (true) {
      delay(1000);
    }
  }

  const tflite::Model* model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema version mismatch");
    while (true) {
      delay(1000);
    }
  }

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    Serial.println("AllocateTensors() failed");
    while (true) {
      delay(1000);
    }
  }

  input_tensor = interpreter->input(0);
  output_tensor = interpreter->output(0);

  input_scale = input_tensor->params.scale;
  input_zero_point = input_tensor->params.zero_point;
  output_scale = output_tensor->params.scale;
  output_zero_point = output_tensor->params.zero_point;

  Serial.println("=== TinyML Cosine Predictor ===");
  Serial.print("Arena used (bytes): ");
  Serial.println(interpreter->arena_used_bytes());
  Serial.print("Arena size (bytes): ");
  Serial.println(kTensorArenaSize);
  Serial.print("g_model size (bytes): ");
  Serial.println(static_cast<unsigned>(g_model_len));
  Serial.print("Input scale: ");
  Serial.println(input_scale, 8);
  Serial.print("Input zero_point: ");
  Serial.println(input_zero_point);
  Serial.print("Output scale: ");
  Serial.println(output_scale, 8);
  Serial.print("Output zero_point: ");
  Serial.println(output_zero_point);
  Serial.println("Format: predicted,actual");
  delay(2000);
}

void loop() {
  int32_t x_quantized =
      static_cast<int32_t>(lroundf(x_value / input_scale) + input_zero_point);
  if (x_quantized < -128) {
    x_quantized = -128;
  }
  if (x_quantized > 127) {
    x_quantized = 127;
  }
  input_tensor->data.int8[0] = static_cast<int8_t>(x_quantized);

  if (interpreter->Invoke() != kTfLiteOk) {
    Serial.println("Invoke failed");
    return;
  }

  const int8_t out_q = output_tensor->data.int8[0];
  const float predicted =
      (static_cast<float>(out_q) - static_cast<float>(output_zero_point)) *
      output_scale;
  const float actual = cosf(x_value);

  Serial.print(predicted, 4);
  Serial.print(",");
  Serial.println(actual, 4);

  x_value += kStep;
  if (x_value > kTwoPi) {
    x_value = 0.0f;
  }

  delay(40);
}
