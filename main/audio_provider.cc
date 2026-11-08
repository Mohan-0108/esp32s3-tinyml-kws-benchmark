#include "audio_provider.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "micro_model_settings.h"
#include <cstring>

namespace {
    int16_t g_dummy_audio_buffer[kMaxAudioSampleSize * 32];
    int32_t g_latest_audio_timestamp = 0;
}

TfLiteStatus GetAudioSamples1(int* audio_samples_size, int16_t** audio_samples) {
    memset(g_dummy_audio_buffer, 0, sizeof(g_dummy_audio_buffer));
    *audio_samples_size = 16000;
    *audio_samples = g_dummy_audio_buffer;
    return kTfLiteOk;
}

TfLiteStatus GetAudioSamples(int start_ms, int duration_ms, int* audio_samples_size, int16_t** audio_samples) {
    memset(g_dummy_audio_buffer, 0, sizeof(g_dummy_audio_buffer));
    *audio_samples_size = kMaxAudioSampleSize;
    *audio_samples = g_dummy_audio_buffer;
    return kTfLiteOk;
}

int32_t LatestAudioTimestamp() {
    // 1. Advance the timestamp by the exact 20ms slice the model expects
    g_latest_audio_timestamp += 20;
    
    // 2. MUST yield to FreeRTOS. This prevents the CPU from locking up and crashing the USB.
    vTaskDelay(pdMS_TO_TICKS(20)); 
    
    return g_latest_audio_timestamp;
}