#include "runtime/stdlib.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

bool stdlib_is_native_module(const char *name) {
    return strcmp(name, "tensor") == 0;
}

char *stdlib_module_path(const char *name) {
    size_t length = strlen(name) + sizeof("stdlib/.fl");
    char *path = malloc(length);
    if (path != NULL) snprintf(path, length, "stdlib/%s.fl", name);
    return path;
}
