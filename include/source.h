#ifndef FLANG_SOURCE_H
#define FLANG_SOURCE_H

#include <stdbool.h>
#include <stddef.h>

bool source_get_line(const char *source, size_t line_number,
                     char *buffer, size_t buffer_size);

#endif
