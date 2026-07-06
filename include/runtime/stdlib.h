#ifndef FLANG_RUNTIME_STDLIB_H
#define FLANG_RUNTIME_STDLIB_H

#include <stdbool.h>

bool stdlib_is_native_module(const char *name);
char *stdlib_module_path(const char *name);

#endif
