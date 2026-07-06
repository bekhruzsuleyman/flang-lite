#ifndef FLANG_VALUE_H
#define FLANG_VALUE_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef struct FunctionValue FunctionValue;
typedef struct ModuleValue ModuleValue;
typedef struct Tensor Tensor;
typedef struct Value Value;

typedef struct {
    char *data;
    size_t length;
} StringValue;

typedef struct {
    Value *items;
    size_t count;
    size_t capacity;
} ArrayValue;

typedef struct {
    int64_t start;
    int64_t end;
} RangeValue;

typedef enum {
    VALUE_INT,
    VALUE_BOOL,
    VALUE_NULL,
    VALUE_STRING,
    VALUE_ARRAY,
    VALUE_TENSOR,
    VALUE_RANGE,
    VALUE_FUNCTION,
    VALUE_MODULE,
    VALUE_NATIVE
} ValueKind;

struct Value {
    ValueKind kind;
    union {
        int64_t integer;
        bool boolean;
        StringValue string;
        ArrayValue array;
        Tensor *tensor;
        RangeValue range;
        FunctionValue *function;
        ModuleValue *module;
        int native_id;
    } as;
};

Value value_int(int64_t value);
Value value_bool(bool value);
Value value_null(void);
Value value_function(FunctionValue *function);
Value value_module(ModuleValue *module);
Value value_native(int native_id);
Value value_tensor(Tensor *tensor);
Value value_range(int64_t start, int64_t end);
bool value_string(const char *data, Value *result);
bool value_array(Value *result);
bool value_array_append(Value *array, Value item);
bool value_clone(const Value *source, Value *result);
void value_free(Value *value);

#endif
