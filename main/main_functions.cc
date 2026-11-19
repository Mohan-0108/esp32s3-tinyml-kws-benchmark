/* Copyright 2020-2023 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 */
#include "esp_timer.h"
#include "esp_log.h"
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iterator>

#include "main_functions.h"

#include "audio_provider.h"
#include "command_responder.h"
#include "feature_provider.h"
#include "micro_model_settings.h"
#include "model.h"
#include "recognize_commands.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* model_input = nullptr;
FeatureProvider* feature_provider = nullptr;
RecognizeCommands* recognizer = nullptr;
int32_t previous_time = 0;

constexpr int kTensorArenaSize = 30 * 1024;
uint8_t tensor_arena[kTensorArenaSize];
int8_t feature_buffer[kFeatureElementCount];
int8_t* model_input_buffer = nullptr;

// Accumulators for 1-second (50-frame) telemetry summaries
int64_t g_feat_sum_us = 0;
int64_t g_inf_sum_us = 0;
int64_t g_e2e_sum_us = 0;
int g_steady_frames = 0;
float g_peak_scores[kCategoryCount] = {0.0f};
bool g_first_inference = true;
}  // namespace

void setup() {
  model = tflite::GetModel(g_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    MicroPrintf("Model provided is schema version %d not equal to supported "
                "version %d.", model->version(), TFLITE_SCHEMA_VERSION);
    return;
  }

  static tflite::MicroMutableOpResolver<4> micro_op_resolver;
  if (micro_op_resolver.AddDepthwiseConv2D() != kTfLiteOk) return;
  if (micro_op_resolver.AddFullyConnected() != kTfLiteOk) return;
  if (micro_op_resolver.AddSoftmax() != kTfLiteOk) return;
  if (micro_op_resolver.AddReshape() != kTfLiteOk) return;

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("AllocateTensors() failed");
    return;
  }
  ESP_LOGW("THESIS_MEM", "Peak Tensor Arena Used: %u / %d bytes",
           (unsigned int)interpreter->arena_used_bytes(), kTensorArenaSize);

  model_input = interpreter->input(0);
  if ((model_input->dims->size != 2) || (model_input->dims->data[0] != 1) ||
      (model_input->dims->data[1] != (kFeatureCount * kFeatureSize)) ||
      (model_input->type != kTfLiteInt8)) {
    MicroPrintf("Bad input tensor parameters in model");
    return;
  }
  model_input_buffer = tflite::GetTensorData<int8_t>(model_input);

  static FeatureProvider static_feature_provider(kFeatureElementCount,
                                                 feature_buffer);
  feature_provider = &static_feature_provider;

  // 300 ms window (~15 frames < 50 max queue), 0.60 threshold, 1000 ms cooldown
static RecognizeCommands static_recognizer(300, 0.72f, 1200, 3);
  recognizer = &static_recognizer;

  previous_time = 0;
}

void loop() {
  const int32_t current_time = LatestAudioTimestamp();
  int how_many_new_slices = 0;

  int64_t feat_start = esp_timer_get_time();
  TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
      previous_time, current_time, &how_many_new_slices);
  int64_t feat_end = esp_timer_get_time();

  if (feature_status != kTfLiteOk) {
    MicroPrintf("Feature generation failed");
    return;
  }
  previous_time = current_time;
  if (how_many_new_slices == 0) {
    return;
  }

  for (int i = 0; i < kFeatureElementCount; i++) {
    model_input_buffer[i] = feature_buffer[i];
  }

  int64_t inf_start = esp_timer_get_time();
  TfLiteStatus invoke_status = interpreter->Invoke();
  int64_t inf_end = esp_timer_get_time();

  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Invoke failed");
    return;
  }

  TfLiteTensor* output = interpreter->output(0);
  const char* found_command = nullptr;
  float score = 0.0f;
  bool is_new_command = false;
  TfLiteStatus process_status = recognizer->ProcessLatestResults(
      output, current_time, &found_command, &score, &is_new_command);
  int64_t post_end = esp_timer_get_time();

  if (process_status != kTfLiteOk) {
    MicroPrintf("RecognizeCommands::ProcessLatestResults() failed");
    return;
  }

  // Track peak class probabilities outside the E2E timer for diagnostics
  float output_scale = output->params.scale;
  int output_zero_point = output->params.zero_point;
  const int8_t* raw_scores = tflite::GetTensorData<int8_t>(output);
  for (int i = 0; i < kCategoryCount; i++) {
    float prob = (raw_scores[i] - output_zero_point) * output_scale;
    if (prob > g_peak_scores[i]) {
      g_peak_scores[i] = prob;
    }
  }

  int64_t feat_us = feat_end - feat_start;
  int64_t inf_us = inf_end - inf_start;
  int64_t e2e_us = post_end - feat_start;

  if (g_first_inference) {
    ESP_LOGW("THESIS_COLD", "Cold Start -> Feat(%d slices): %lld us | Inf: %lld us | E2E: %lld us",
             how_many_new_slices, (long long)feat_us, (long long)inf_us, (long long)e2e_us);
    g_first_inference = false;
  } else if (how_many_new_slices == 1) {
    g_feat_sum_us += feat_us;
    g_inf_sum_us += inf_us;
    g_e2e_sum_us += e2e_us;
    g_steady_frames++;
    if (g_steady_frames == 50) {
      ESP_LOGW("THESIS_LIVE",
               "Avg(50f) -> Feat: %lld us | Inf: %lld us | E2E: %lld us | Peak[sil:%.2f unk:%.2f yes:%.2f no:%.2f]",
               (long long)(g_feat_sum_us / 50),
               (long long)(g_inf_sum_us / 50),
               (long long)(g_e2e_sum_us / 50),
               static_cast<double>(g_peak_scores[0]),
               static_cast<double>(g_peak_scores[1]),
               static_cast<double>(g_peak_scores[2]),
               static_cast<double>(g_peak_scores[3]));
      g_feat_sum_us = 0;
      g_inf_sum_us = 0;
      g_e2e_sum_us = 0;
      g_steady_frames = 0;
      for (int i = 0; i < kCategoryCount; i++) {
        g_peak_scores[i] = 0.0f;
      }
    }
  }

  if (is_new_command && found_command != nullptr &&
      (std::strcmp(found_command, "yes") == 0 ||
       std::strcmp(found_command, "no") == 0)) {
    ESP_LOGW("THESIS_KWS", ">>> DETECTED: %s (score: %.2f) at %ld ms",
             found_command, static_cast<double>(score), (long)current_time);
  }
}