#include "runtime/value.h"

#include "runtime/tensor.h"

#include <stdlib.h>
#include <string.h>

Value value_int(int64_t value) {
    Value result;
    memset(&result, 0, sizeof(result));
    result.kind = VALUE_INT;
    result.as.integer = value;
    return result;
}

Value value_bool(bool value) {
    Value result;
    memset(&result, 0, sizeof(result));
    result.kind = VALUE_BOOL;
    result.as.boolean = value;
    return result;
}

Value value_null(void) {
    Value result;
    memset(&result, 0, sizeof(result));
    result.kind = VALUE_NULL;
    return result;
}

Value value_function(FunctionValue *function) {
    Value result;
    memset(&result, 0, sizeof(result));
    result.kind = VALUE_FUNCTION;
    result.as.function = function;
    return result;
}

Value value_module(ModuleValue *module) {
    Value result;
    memset(&result, 0, sizeof(result));
    result.kind = VALUE_MODULE;
    result.as.module = module;
    return result;
}

Value value_native(int native_id) {
    Value result;
    memset(&result, 0, sizeof(result));
    result.kind = VALUE_NATIVE;
    result.as.native_id = native_id;
    return result;
}

Value value_tensor(Tensor *tensor) {
    Value result;
    memset(&result, 0, sizeof(result));
    result.kind = VALUE_TENSOR;
    result.as.tensor = tensor;
    return result;
}

Value value_range(int64_t start, int64_t end) {
    Value result;
    memset(&result, 0, sizeof(result));
    result.kind = VALUE_RANGE;
    result.as.range.start = start;
    result.as.range.end = end;
    return result;
}

bool value_string(const char *data, Value *result) {
    size_t length = strlen(data);
    char *copy = malloc(length + 1);
    if (copy == NULL) return false;
    memcpy(copy, data, length + 1);
    memset(result, 0, sizeof(*result));
    result->kind = VALUE_STRING;
    result->as.string.data = copy;
    result->as.string.length = length;
    return true;
}

bool value_array(Value *result) {
    memset(result, 0, sizeof(*result));
    result->kind = VALUE_ARRAY;
    return true;
}

bool value_array_append(Value *array, Value item) {
    Value *new_items;
    size_t new_capacity;
    if (array->as.array.count == array->as.array.capacity) {
        new_capacity = array->as.array.capacity == 0 ? 4 : array->as.array.capacity * 2;
        new_items = realloc(array->as.array.items,
                            new_capacity * sizeof(*new_items));
        if (new_items == NULL) return false;
        array->as.array.items = new_items;
        array->as.array.capacity = new_capacity;
    }
    array->as.array.items[array->as.array.count++] = item;
    return true;
}

bool value_clone(const Value *source, Value *result) {
    size_t index;
    memset(result, 0, sizeof(*result));
    result->kind = source->kind;
    switch (source->kind) {
        case VALUE_INT:
            result->as.integer = source->as.integer;
            return true;
        case VALUE_BOOL:
            result->as.boolean = source->as.boolean;
            return true;
        case VALUE_NULL:
            return true;
        case VALUE_STRING:
            return value_string(source->as.string.data, result);
        case VALUE_ARRAY:
            value_array(result);
            for (index = 0; index < source->as.array.count; ++index) {
                Value item;
                if (!value_clone(&source->as.array.items[index], &item) ||
                    !value_array_append(result, item)) {
                    if (item.kind == VALUE_STRING || item.kind == VALUE_ARRAY ||
                        item.kind == VALUE_TENSOR)
                        value_free(&item);
                    value_free(result);
                    return false;
                }
            }
            return true;
        case VALUE_TENSOR:
            result->as.tensor = source->as.tensor;
            tensor_retain(result->as.tensor);
            return true;
        case VALUE_RANGE:
            result->as.range = source->as.range;
            return true;
        case VALUE_FUNCTION:
            result->as.function = source->as.function;
            return true;
        case VALUE_MODULE:
            result->as.module = source->as.module;
            return true;
        case VALUE_NATIVE:
            result->as.native_id = source->as.native_id;
            return true;
    }
    return false;
}

void value_free(Value *value) {
    size_t index;
    switch (value->kind) {
        case VALUE_STRING:
            free(value->as.string.data);
            break;
        case VALUE_ARRAY:
            for (index = 0; index < value->as.array.count; ++index)
                value_free(&value->as.array.items[index]);
            free(value->as.array.items);
            break;
        case VALUE_TENSOR:
            tensor_release(value->as.tensor);
            break;
        default:
            break;
    }
    *value = value_null();
}
