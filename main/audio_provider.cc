#include "audio_provider.h"

#include <cstdint>
#include <cstring>

#include "driver/gpio.h"
#include "driver/i2s_std.h"
#include "esp_err.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "micro_model_settings.h"

namespace {

constexpr char kTag[] = "INMP441_AUDIO";
constexpr gpio_num_t kI2sWsPin = GPIO_NUM_17;
constexpr gpio_num_t kI2sSckPin = GPIO_NUM_7;
constexpr gpio_num_t kI2sSdPin = GPIO_NUM_6;

// 20 ms chunk at 16 kHz = 320 samples
constexpr int kChunkSamples = (kAudioSampleFrequency * 20) / 1000;
// Power-of-2 ring buffer: 32,768 samples (~2.048 seconds of 16 kHz audio)
constexpr int kHistoryBufferSize = 32768;
constexpr int kHistoryBufferMask = kHistoryBufferSize - 1;

// INMP441 outputs 24-bit audio in a 32-bit slot.
// Shift by 15 gives ~2x digital gain (reduces room-noise masking).
constexpr int kAudioGainShift = 15;

int16_t g_history_buffer[kHistoryBufferSize] = {0};
int16_t g_audio_output_buffer[kMaxAudioSampleSize] = {0};

portMUX_TYPE g_audio_mux = portMUX_INITIALIZER_UNLOCKED;
volatile int64_t g_total_samples_captured = 0;
volatile int32_t g_latest_audio_timestamp = 0;
int32_t g_last_reported_timestamp = 0;
bool g_is_audio_initialized = false;
i2s_chan_handle_t g_rx_handle = nullptr;

void CaptureSamplesTask(void* arg) {
  (void)arg;
  int32_t raw_buf[kChunkSamples];
  size_t bytes_read = 0;

  while (true) {
    esp_err_t ret = i2s_channel_read(g_rx_handle, raw_buf, sizeof(raw_buf),
                                     &bytes_read, portMAX_DELAY);
    if (ret == ESP_OK && bytes_read > 0) {
      int count = static_cast<int>(bytes_read / sizeof(int32_t));
      int64_t write_pos = g_total_samples_captured;

      for (int i = 0; i < count; ++i) {
        int32_t scaled = raw_buf[i] >> kAudioGainShift;
        if (scaled > 32767) scaled = 32767;
        if (scaled < -32768) scaled = -32768;
        g_history_buffer[(write_pos + i) & kHistoryBufferMask] =
            static_cast<int16_t>(scaled);
      }

      portENTER_CRITICAL(&g_audio_mux);
      g_total_samples_captured = write_pos + count;
      int64_t safe_samples = g_total_samples_captured - kMaxAudioSampleSize;
      if (safe_samples > 0) {
        g_latest_audio_timestamp = static_cast<int32_t>(
            (safe_samples * 1000LL) / kAudioSampleFrequency);
      }
      portEXIT_CRITICAL(&g_audio_mux);
    }
  }
}

void InitAudioRecording() {
  if (g_is_audio_initialized) {
    return;
  }

  i2s_chan_config_t chan_cfg =
      I2S_CHANNEL_DEFAULT_CONFIG(I2S_NUM_0, I2S_ROLE_MASTER);
  ESP_ERROR_CHECK(i2s_new_channel(&chan_cfg, nullptr, &g_rx_handle));

  i2s_std_config_t std_cfg = {};
  std_cfg.clk_cfg = I2S_STD_CLK_DEFAULT_CONFIG(kAudioSampleFrequency);
  std_cfg.slot_cfg = I2S_STD_PHILIPS_SLOT_DEFAULT_CONFIG(
      I2S_DATA_BIT_WIDTH_32BIT, I2S_SLOT_MODE_MONO);
  std_cfg.slot_cfg.slot_mask = I2S_STD_SLOT_LEFT;
  std_cfg.gpio_cfg.mclk = I2S_GPIO_UNUSED;
  std_cfg.gpio_cfg.bclk = kI2sSckPin;
  std_cfg.gpio_cfg.ws = kI2sWsPin;
  std_cfg.gpio_cfg.dout = I2S_GPIO_UNUSED;
  std_cfg.gpio_cfg.din = kI2sSdPin;
  std_cfg.gpio_cfg.invert_flags.mclk_inv = false;
  std_cfg.gpio_cfg.invert_flags.bclk_inv = false;
  std_cfg.gpio_cfg.invert_flags.ws_inv = false;

  ESP_ERROR_CHECK(i2s_channel_init_std_mode(g_rx_handle, &std_cfg));
  ESP_ERROR_CHECK(i2s_channel_enable(g_rx_handle));

  xTaskCreatePinnedToCore(CaptureSamplesTask, "inmp441_capture", 4096, nullptr,
                          5, nullptr, 1);

  g_is_audio_initialized = true;
  ESP_LOGW(kTag, "INMP441 live audio initialized on Core 1 (16 kHz mono)");
}

}  // namespace

TfLiteStatus GetAudioSamples1(int* audio_samples_size,
                              int16_t** audio_samples) {
  if (!g_is_audio_initialized) {
    InitAudioRecording();
  }
  int64_t total_captured = 0;
  portENTER_CRITICAL(&g_audio_mux);
  total_captured = g_total_samples_captured;
  portEXIT_CRITICAL(&g_audio_mux);

  int64_t start_sample = total_captured - kMaxAudioSampleSize;
  for (int i = 0; i < kMaxAudioSampleSize; ++i) {
    int64_t idx = start_sample + i;
    if (idx < 0 || idx >= total_captured ||
        (total_captured - idx) >= kHistoryBufferSize) {
      g_audio_output_buffer[i] = 0;
    } else {
      g_audio_output_buffer[i] = g_history_buffer[idx & kHistoryBufferMask];
    }
  }
  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_audio_output_buffer;
  return kTfLiteOk;
}

TfLiteStatus GetAudioSamples(int start_ms, int duration_ms,
                             int* audio_samples_size,
                             int16_t** audio_samples) {
  (void)duration_ms;
  if (!g_is_audio_initialized) {
    InitAudioRecording();
  }

  int64_t start_sample =
      (static_cast<int64_t>(start_ms) * kAudioSampleFrequency) / 1000;

  int64_t total_captured = 0;
  portENTER_CRITICAL(&g_audio_mux);
  total_captured = g_total_samples_captured;
  portEXIT_CRITICAL(&g_audio_mux);

  for (int i = 0; i < kMaxAudioSampleSize; ++i) {
    int64_t idx = start_sample + i;
    if (idx < 0 || idx >= total_captured ||
        (total_captured - idx) >= kHistoryBufferSize) {
      g_audio_output_buffer[i] = 0;
    } else {
      g_audio_output_buffer[i] = g_history_buffer[idx & kHistoryBufferMask];
    }
  }

  *audio_samples_size = kMaxAudioSampleSize;
  *audio_samples = g_audio_output_buffer;
  return kTfLiteOk;
}

int32_t LatestAudioTimestamp() {
  if (!g_is_audio_initialized) {
    InitAudioRecording();
  }

  int32_t current = g_latest_audio_timestamp;
  while ((current - g_last_reported_timestamp) < 20) {
    vTaskDelay(pdMS_TO_TICKS(5));
    current = g_latest_audio_timestamp;
  }
  g_last_reported_timestamp = current;
  return current;
}