#include "command_responder.h"
#include "esp_log.h"

void RespondToCommand(int32_t current_time, const char* found_command,
                      uint8_t score, bool is_new_command) {
    // Force the AI to print every single inference result, ignoring the silence filter
    ESP_LOGI("AI_RESULT", "Heard %s (Score: %d)", found_command, score);
}