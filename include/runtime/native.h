#ifndef FLANG_RUNTIME_NATIVE_H
#define FLANG_RUNTIME_NATIVE_H

#include "value.h"

#include <stdbool.h>
#include <stddef.h>

typedef enum {
    NATIVE_TENSOR_CREATE,
    NATIVE_TENSOR_ZEROS,
    NATIVE_TENSOR_ONES,
    NATIVE_TENSOR_ARANGE
} NativeId;

typedef bool (*NativeFn)(Value *args, size_t arg_count, Value *result,
                         char *error, size_t error_size);

typedef struct {
    const char *module;
    const char *name;
    NativeId id;
    NativeFn fn;
} NativeSymbol;

const NativeSymbol *native_find(const char *module, const char *name);
const NativeSymbol *native_symbol_at(size_t index);
size_t native_symbol_count(void);
bool native_call(NativeId id, Value *args, size_t arg_count, Value *result,
                 char *error, size_t error_size);

#endif
