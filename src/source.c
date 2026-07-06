#include "source.h"

#include <string.h>

bool source_get_line(const char *source, size_t line_number,
                     char *buffer, size_t buffer_size) {
    const char *start = source;
    const char *end;
    size_t current_line = 1;
    size_t length;
    if (source == NULL || line_number == 0 || buffer_size == 0) return false;
    while (current_line < line_number && *start != '\0') {
        if (*start++ == '\n') ++current_line;
    }
    if (current_line != line_number) return false;
    end = start;
    while (*end != '\0' && *end != '\n' && *end != '\r') ++end;
    length = (size_t)(end - start);
    if (length >= buffer_size) length = buffer_size - 1;
    memcpy(buffer, start, length);
    buffer[length] = '\0';
    return true;
}
