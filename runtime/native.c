#include "runtime/native.h"

#include "runtime/tensor.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    INFER_OK,
    INFER_RAGGED,
    INFER_TYPE,
    INFER_MEMORY
} InferResult;

static InferResult infer_shape(const Value *value, int64_t **shape, int *rank) {
    int64_t *child_shape = NULL;
    int child_rank = 0;
    size_t index;
    if (value->kind == VALUE_INT) {
        *shape = NULL;
        *rank = 0;
        return INFER_OK;
    }
    if (value->kind != VALUE_ARRAY) return INFER_TYPE;
    if (value->as.array.count > 0) {
        InferResult status = infer_shape(&value->as.array.items[0],
                                         &child_shape, &child_rank);
        if (status != INFER_OK) return status;
        for (index = 1; index < value->as.array.count; ++index) {
            int64_t *other_shape = NULL;
            int other_rank = 0;
            int dimension;
            status = infer_shape(&value->as.array.items[index], &other_shape,
                                 &other_rank);
            if (status != INFER_OK) {
                free(child_shape);
                return status;
            }
            if (other_rank != child_rank) {
                free(other_shape);
                free(child_shape);
                return INFER_RAGGED;
            }
            for (dimension = 0; dimension < child_rank; ++dimension) {
                if (other_shape[dimension] != child_shape[dimension]) {
                    free(other_shape);
                    free(child_shape);
                    return INFER_RAGGED;
                }
            }
            free(other_shape);
        }
    }
    *shape = malloc((size_t)(child_rank + 1) * sizeof(**shape));
    if (*shape == NULL) {
        free(child_shape);
        return INFER_MEMORY;
    }
    (*shape)[0] = (int64_t)value->as.array.count;
    if (child_rank > 0)
        memcpy(*shape + 1, child_shape,
               (size_t)child_rank * sizeof(**shape));
    free(child_shape);
    *rank = child_rank + 1;
    return INFER_OK;
}

static void flatten(const Value *value, double *data, int64_t *offset) {
    size_t index;
    if (value->kind == VALUE_INT) {
        data[(*offset)++] = (double)value->as.integer;
        return;
    }
    for (index = 0; index < value->as.array.count; ++index)
        flatten(&value->as.array.items[index], data, offset);
}

static bool native_tensor(Value *args, size_t count, Value *result,
                          char *error, size_t error_size) {
    int64_t *shape = NULL;
    int rank = 0;
    int64_t offset = 0;
    Tensor *tensor;
    InferResult inferred;
    bool requires_grad = false;
    if (count != 1 && count != 2) {
        snprintf(error, error_size,
                 "tensor() expects 1 or 2 arguments, got %zu", count);
        return false;
    }
    if (args[0].kind != VALUE_ARRAY) {
        snprintf(error, error_size, "tensor() expects a list");
        return false;
    }
    if (count == 2) {
        if (args[1].kind != VALUE_BOOL) {
            snprintf(error, error_size,
                     "tensor() requires requires_grad to be bool");
            return false;
        }
        requires_grad = args[1].as.boolean;
    }
    inferred = infer_shape(&args[0], &shape, &rank);
    if (inferred != INFER_OK) {
        snprintf(error, error_size, "%s",
                 inferred == INFER_RAGGED ? "Ragged tensor literal"
                 : inferred == INFER_TYPE ? "Tensor elements must be int"
                                          : "Out of memory");
        return false;
    }
    tensor = tensor_new(shape, rank, requires_grad);
    free(shape);
    if (tensor == NULL) {
        snprintf(error, error_size, "Out of memory");
        return false;
    }
    flatten(&args[0], tensor->data, &offset);
    *result = value_tensor(tensor);
    return true;
}

static bool shape_from_value(const Value *value, int64_t **shape, int *rank,
                             char *error, size_t error_size) {
    size_t index;
    if (value->kind != VALUE_ARRAY || value->as.array.count == 0) {
        snprintf(error, error_size,
                 "Tensor shape must be a non-empty list");
        return false;
    }
    if (value->as.array.count > (size_t)INT_MAX) {
        snprintf(error, error_size, "Tensor rank is too large");
        return false;
    }
    *shape = malloc(value->as.array.count * sizeof(**shape));
    if (*shape == NULL) {
        snprintf(error, error_size, "Out of memory");
        return false;
    }
    *rank = (int)value->as.array.count;
    for (index = 0; index < value->as.array.count; ++index) {
        const Value *dimension = &value->as.array.items[index];
        if (dimension->kind != VALUE_INT || dimension->as.integer < 0) {
            free(*shape);
            *shape = NULL;
            snprintf(error, error_size,
                     "Tensor shape dimensions must be non-negative ints");
            return false;
        }
        (*shape)[index] = dimension->as.integer;
    }
    return true;
}

static bool native_fill(Value *args, size_t count, Value *result,
                        char *error, size_t error_size, double fill) {
    int64_t *shape = NULL;
    int rank = 0;
    Tensor *tensor;
    if (count != 1) {
        snprintf(error, error_size, "Tensor factory expects 1 argument");
        return false;
    }
    if (!shape_from_value(&args[0], &shape, &rank, error, error_size))
        return false;
    tensor = tensor_filled(shape, rank, fill, false, error, error_size);
    free(shape);
    if (tensor == NULL) return false;
    *result = value_tensor(tensor);
    return true;
}

static bool native_zeros(Value *args, size_t count, Value *result,
                         char *error, size_t error_size) {
    return native_fill(args, count, result, error, error_size, 0.0);
}

static bool native_ones(Value *args, size_t count, Value *result,
                        char *error, size_t error_size) {
    return native_fill(args, count, result, error, error_size, 1.0);
}

static bool native_arange(Value *args, size_t count, Value *result,
                          char *error, size_t error_size) {
    Tensor *tensor;
    if (count != 2) {
        snprintf(error, error_size, "arange() expects 2 arguments, got %zu",
                 count);
        return false;
    }
    if (args[0].kind != VALUE_INT || args[1].kind != VALUE_INT) {
        snprintf(error, error_size, "arange() expects int and int");
        return false;
    }
    tensor = tensor_arange(args[0].as.integer, args[1].as.integer,
                           error, error_size);
    if (tensor == NULL) return false;
    *result = value_tensor(tensor);
    return true;
}

static const NativeSymbol symbols[] = {
    {"tensor", "tensor", NATIVE_TENSOR_CREATE, native_tensor},
    {"tensor", "zeros", NATIVE_TENSOR_ZEROS, native_zeros},
    {"tensor", "ones", NATIVE_TENSOR_ONES, native_ones},
    {"tensor", "arange", NATIVE_TENSOR_ARANGE, native_arange}
};

size_t native_symbol_count(void) {
    return sizeof(symbols) / sizeof(symbols[0]);
}

const NativeSymbol *native_symbol_at(size_t index) {
    return index < native_symbol_count() ? &symbols[index] : NULL;
}

const NativeSymbol *native_find(const char *module, const char *name) {
    size_t index;
    for (index = 0; index < native_symbol_count(); ++index)
        if (strcmp(symbols[index].module, module) == 0 &&
            strcmp(symbols[index].name, name) == 0)
            return &symbols[index];
    return NULL;
}

bool native_call(NativeId id, Value *args, size_t arg_count, Value *result,
                 char *error, size_t error_size) {
    size_t index;
    for (index = 0; index < native_symbol_count(); ++index)
        if (symbols[index].id == id)
            return symbols[index].fn(args, arg_count, result, error, error_size);
    snprintf(error, error_size, "Unknown native function");
    return false;
}
