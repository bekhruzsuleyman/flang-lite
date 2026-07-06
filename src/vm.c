#include "vm.h"

#include "ir.h"
#include "runtime/native.h"
#include "runtime/tensor.h"

#include <inttypes.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const BytecodeProgram *program;
    Value *stack;
    size_t count;
    size_t capacity;
    Value *globals;
    FILE *output;
    RuntimeError *error;
    const char *path;
} VM;

static bool vm_fail(VM *vm, const BytecodeInstr *instruction,
                    const char *message) {
    if (!vm->error->has_error) {
        vm->error->has_error = true;
        vm->error->line = instruction->span.line;
        vm->error->column = instruction->span.column;
        snprintf(vm->error->path, sizeof(vm->error->path), "%s", vm->path);
        snprintf(vm->error->message, sizeof(vm->error->message), "%s", message);
        if (strstr(message, "backward() requires scalar") != NULL)
            snprintf(vm->error->hint, sizeof(vm->error->hint),
                     "use tensor.sum().backward() or tensor.mean().backward()");
    }
    return false;
}

static bool push(VM *vm, Value value) {
    Value *grown;
    if (vm->count == vm->capacity) {
        size_t next = vm->capacity == 0 ? 256 : vm->capacity * 2;
        grown = realloc(vm->stack, next * sizeof(*grown));
        if (grown == NULL) return false;
        vm->stack = grown;
        vm->capacity = next;
    }
    vm->stack[vm->count++] = value;
    return true;
}

static Value pop(VM *vm) {
    return vm->count == 0 ? value_null() : vm->stack[--vm->count];
}

static bool print_value(FILE *output, const Value *value) {
    size_t index;
    if (value->kind == VALUE_INT)
        return fprintf(output, "%" PRId64, value->as.integer) >= 0;
    if (value->kind == VALUE_BOOL)
        return fputs(value->as.boolean ? "true" : "false", output) != EOF;
    if (value->kind == VALUE_STRING)
        return fputs(value->as.string.data, output) != EOF;
    if (value->kind == VALUE_TENSOR)
        return tensor_print(output, value->as.tensor);
    if (value->kind == VALUE_ARRAY) {
        if (fputc('[', output) == EOF) return false;
        for (index = 0; index < value->as.array.count; ++index) {
            if (index > 0 && fputs(", ", output) == EOF) return false;
            if (!print_value(output, &value->as.array.items[index])) return false;
        }
        return fputc(']', output) != EOF;
    }
    return false;
}

static bool integer_op(VM *vm, const BytecodeInstr *in, OpCode op,
                       int64_t left, int64_t right, Value *result) {
    int64_t value;
    if (op == OP_ADD_INT) {
        if ((right > 0 && left > INT64_MAX - right) ||
            (right < 0 && left < INT64_MIN - right))
            return vm_fail(vm, in, "Integer overflow");
        value = left + right;
    } else if (op == OP_SUB_INT) {
        if ((right > 0 && left < INT64_MIN + right) ||
            (right < 0 && left > INT64_MAX + right))
            return vm_fail(vm, in, "Integer overflow");
        value = left - right;
    } else if (op == OP_MUL_INT) {
        if (left != 0 && right != 0 &&
            (left == INT64_MIN ? right != 1
             : right == INT64_MIN ? left != 1
             : left > 0 ? (right > 0 ? left > INT64_MAX / right
                                      : right < INT64_MIN / left)
                        : (right > 0 ? left < INT64_MIN / right
                                     : left < INT64_MAX / right)))
            return vm_fail(vm, in, "Integer overflow");
        value = left * right;
    } else {
        if (right == 0) return vm_fail(vm, in, "Division by zero");
        if (left == INT64_MIN && right == -1)
            return vm_fail(vm, in, "Integer overflow");
        value = left / right;
    }
    *result = value_int(value);
    return true;
}

static bool native_call_vm(VM *vm, const BytecodeInstr *in, int id,
                           Value *args, size_t count, Value *result) {
    char error[256] = {0};
    size_t index;
    NativeId native;
    if (id == IR_NATIVE_LEN) {
        if (count != 1 || (args[0].kind != VALUE_ARRAY &&
                           args[0].kind != VALUE_STRING))
            return vm_fail(vm, in, "len() expects a string or list");
        *result = value_int((int64_t)(args[0].kind == VALUE_ARRAY
                                          ? args[0].as.array.count
                                          : args[0].as.string.length));
        return true;
    }
    if (id == IR_NATIVE_RANGE) {
        if (count != 2 || args[0].kind != VALUE_INT || args[1].kind != VALUE_INT)
            return vm_fail(vm, in, "range() expects int and int");
        *result = value_range(args[0].as.integer, args[1].as.integer);
        return true;
    }
    if (id == IR_NATIVE_SHAPE || id == IR_NATIVE_RANK) {
        if (count != 1 || args[0].kind != VALUE_TENSOR)
            return vm_fail(vm, in, "shape()/rank() expects tensor");
        if (id == IR_NATIVE_RANK) {
            *result = value_int(args[0].as.tensor->rank);
            return true;
        }
        value_array(result);
        for (index = 0; index < (size_t)args[0].as.tensor->rank; ++index)
            if (!value_array_append(
                    result, value_int(args[0].as.tensor->shape[index]))) {
                value_free(result);
                return vm_fail(vm, in, "Out of memory");
            }
        return true;
    }
    native = id == IR_NATIVE_TENSOR ? NATIVE_TENSOR_CREATE
             : id == IR_NATIVE_ZEROS ? NATIVE_TENSOR_ZEROS
             : id == IR_NATIVE_ONES ? NATIVE_TENSOR_ONES
                                    : NATIVE_TENSOR_ARANGE;
    if (!native_call(native, args, count, result, error, sizeof(error)))
        return vm_fail(vm, in, error);
    return true;
}

static bool execute_function(VM *vm, int function_index, Value *arguments,
                             size_t argument_count, Value *returned) {
    const BytecodeFunction *function = &vm->program->functions[function_index];
    Value *locals = calloc(function->local_count, sizeof(*locals));
    size_t ip = 0;
    size_t index;
    bool ok = false;
    if (function->local_count > 0 && locals == NULL) return false;
    for (index = 0; index < function->local_count; ++index) locals[index] = value_null();
    for (index = 0; index < argument_count && index < function->parameter_count;
         ++index)
        if (!value_clone(&arguments[index], &locals[index])) goto cleanup;
    while (ip < function->count) {
        const BytecodeInstr *in = &function->code[ip++];
        OpCode op = (OpCode)in->op;
        if (op == OP_NOP) continue;
        if (op == OP_CONST) {
            Value value;
            if (!value_clone(&in->constant, &value) || !push(vm, value))
                goto memory_error;
        } else if (op == OP_LOAD_LOCAL || op == OP_LOAD_GLOBAL) {
            Value value;
            Value *source = op == OP_LOAD_LOCAL ? &locals[in->a]
                                                : &vm->globals[in->a];
            if (!value_clone(source, &value) || !push(vm, value))
                goto memory_error;
        } else if (op == OP_STORE_LOCAL || op == OP_STORE_GLOBAL) {
            Value value = pop(vm);
            Value *target = op == OP_STORE_LOCAL ? &locals[in->a]
                                                 : &vm->globals[in->a];
            value_free(target);
            *target = value;
        } else if (op >= OP_ADD_INT && op <= OP_DIV_INT) {
            Value right = pop(vm), left = pop(vm), result;
            bool success = integer_op(vm, in, op, left.as.integer,
                                      right.as.integer, &result);
            value_free(&left); value_free(&right);
            if (!success || !push(vm, result)) goto cleanup;
        } else if (op == OP_ADD_STR) {
            Value right = pop(vm), left = pop(vm), result = value_null();
            size_t length = left.as.string.length + right.as.string.length;
            result.kind = VALUE_STRING;
            result.as.string.data = malloc(length + 1);
            if (result.as.string.data == NULL) {
                value_free(&left); value_free(&right); goto memory_error;
            }
            memcpy(result.as.string.data, left.as.string.data, left.as.string.length);
            memcpy(result.as.string.data + left.as.string.length,
                   right.as.string.data, right.as.string.length + 1);
            result.as.string.length = length;
            value_free(&left); value_free(&right);
            if (!push(vm, result)) { value_free(&result); goto memory_error; }
        } else if (op == OP_EQ || op == OP_NEQ ||
                   (op >= OP_LT_INT && op <= OP_GTE_INT)) {
            Value right = pop(vm), left = pop(vm);
            bool equal = left.kind == VALUE_INT
                             ? left.as.integer == right.as.integer
                         : left.kind == VALUE_BOOL
                             ? left.as.boolean == right.as.boolean
                         : strcmp(left.as.string.data, right.as.string.data) == 0;
            bool answer = op == OP_EQ ? equal : op == OP_NEQ ? !equal
                : op == OP_LT_INT ? left.as.integer < right.as.integer
                : op == OP_LTE_INT ? left.as.integer <= right.as.integer
                : op == OP_GT_INT ? left.as.integer > right.as.integer
                                  : left.as.integer >= right.as.integer;
            value_free(&left); value_free(&right);
            if (!push(vm, value_bool(answer))) goto memory_error;
        } else if (op == OP_NEG_INT || op == OP_NOT_BOOL) {
            Value value = pop(vm);
            if (op == OP_NEG_INT) value.as.integer = -value.as.integer;
            else value.as.boolean = !value.as.boolean;
            if (!push(vm, value)) goto memory_error;
        } else if (op == OP_AND_BOOL || op == OP_OR_BOOL) {
            Value right = pop(vm), left = pop(vm);
            bool answer = op == OP_AND_BOOL ? left.as.boolean && right.as.boolean
                                            : left.as.boolean || right.as.boolean;
            value_free(&left); value_free(&right);
            if (!push(vm, value_bool(answer))) goto memory_error;
        } else if (op == OP_JUMP) {
            ip = (size_t)in->a;
        } else if (op == OP_JUMP_IF_FALSE) {
            Value condition = pop(vm);
            bool take = !condition.as.boolean;
            value_free(&condition);
            if (take) ip = (size_t)in->a;
        } else if (op == OP_CALL || op == OP_CALL_NATIVE) {
            size_t count = (size_t)in->b;
            Value *args = count == 0 ? NULL : calloc(count, sizeof(*args));
            Value result = value_null();
            bool success;
            if (count > 0 && args == NULL) goto memory_error;
            for (index = count; index-- > 0;) args[index] = pop(vm);
            success = op == OP_CALL
                          ? execute_function(vm, in->a, args, count, &result)
                          : native_call_vm(vm, in, in->a, args, count, &result);
            for (index = 0; index < count; ++index) value_free(&args[index]);
            free(args);
            if (!success || !push(vm, result)) { value_free(&result); goto cleanup; }
        } else if (op == OP_RETURN) {
            *returned = pop(vm);
            ok = true;
            goto cleanup;
        } else if (op == OP_PRINT) {
            Value value = pop(vm);
            if (!print_value(vm->output, &value) || fputc('\n', vm->output) == EOF) {
                value_free(&value);
                vm_fail(vm, in, "Unable to print value");
                goto cleanup;
            }
            value_free(&value);
        } else if (op == OP_POP) {
            Value value = pop(vm); value_free(&value);
        } else if (op == OP_BUILD_ARRAY) {
            size_t count = (size_t)in->a;
            Value *items = count == 0 ? NULL : calloc(count, sizeof(*items));
            Value array;
            if (count > 0 && items == NULL) goto memory_error;
            for (index = count; index-- > 0;) items[index] = pop(vm);
            value_array(&array);
            for (index = 0; index < count; ++index)
                if (!value_array_append(&array, items[index])) goto memory_error;
            free(items);
            if (!push(vm, array)) { value_free(&array); goto memory_error; }
        } else if (op == OP_INDEX) {
            Value subscript = pop(vm), object = pop(vm), result = value_null();
            int64_t position = subscript.as.integer;
            if (object.kind == VALUE_ARRAY && position >= 0 &&
                (size_t)position < object.as.array.count)
                value_clone(&object.as.array.items[position], &result);
            else if (object.kind == VALUE_STRING && position >= 0 &&
                     (size_t)position < object.as.string.length) {
                char text[2] = {object.as.string.data[position], '\0'};
                value_string(text, &result);
            } else if (object.kind == VALUE_TENSOR && position >= 0 &&
                       object.as.tensor->rank > 0 &&
                       position < object.as.tensor->shape[0]) {
                if (object.as.tensor->rank == 1)
                    result = value_int((int64_t)object.as.tensor->data[position]);
                else {
                    char error[256] = {0};
                    Tensor *slice = tensor_copy_slice(object.as.tensor, position,
                                                      error, sizeof(error));
                    result = value_tensor(slice);
                }
            } else {
                const char *message = object.kind == VALUE_ARRAY
                                          ? "Array index out of bounds"
                                      : object.kind == VALUE_STRING
                                          ? "String index out of bounds"
                                          : "Tensor index out of bounds";
                value_free(&object); value_free(&subscript);
                vm_fail(vm, in, message); goto cleanup;
            }
            value_free(&object); value_free(&subscript);
            if (!push(vm, result)) { value_free(&result); goto memory_error; }
        } else if (op == OP_GET_PROPERTY) {
            Value object = pop(vm), result = value_null();
            if (in->a == IR_PROP_LEN)
                result = value_int(object.kind == VALUE_ARRAY
                    ? (int64_t)object.as.array.count
                    : object.kind == VALUE_STRING ? (int64_t)object.as.string.length
                                                  : object.as.tensor->len);
            else if (in->a == IR_PROP_RANK)
                result = value_int(object.as.tensor->rank);
            else if (in->a == IR_PROP_REQUIRES_GRAD)
                result = value_bool(object.as.tensor->requires_grad);
            else if (in->a == IR_PROP_SHAPE) {
                value_array(&result);
                for (index = 0; index < (size_t)object.as.tensor->rank; ++index)
                    value_array_append(&result,
                        value_int(object.as.tensor->shape[index]));
            } else {
                char error[256] = {0};
                Tensor *gradient = tensor_gradient_copy(object.as.tensor, error,
                                                        sizeof(error));
                if (gradient == NULL) {
                    value_free(&object); vm_fail(vm, in, error); goto cleanup;
                }
                result = value_tensor(gradient);
            }
            value_free(&object);
            if (!push(vm, result)) { value_free(&result); goto memory_error; }
        } else if (op == OP_CALL_METHOD) {
            Value object = pop(vm), result = value_null();
            char error[256] = {0};
            Tensor *tensor = object.as.tensor;
            if (in->a == AST_NATIVE_TENSOR_SUM ||
                in->a == AST_NATIVE_TENSOR_MEAN) {
                Tensor *reduced = tensor_reduce(
                    tensor, in->a == AST_NATIVE_TENSOR_MEAN,
                    error, sizeof(error));
                if (reduced != NULL) result = value_tensor(reduced);
            } else if (in->a == AST_NATIVE_TENSOR_BACKWARD) {
                if (tensor_backward(tensor, error, sizeof(error)))
                    result = value_null();
            } else {
                tensor_zero_grad(tensor);
                result = value_null();
            }
            value_free(&object);
            if (error[0] != '\0') { vm_fail(vm, in, error); goto cleanup; }
            if (!push(vm, result)) { value_free(&result); goto memory_error; }
        } else if (op >= OP_TENSOR_ADD && op <= OP_TENSOR_DIV) {
            Value right = pop(vm), left = pop(vm), result;
            char error[256] = {0};
            TensorOp tensor_op = op == OP_TENSOR_ADD ? TENSOR_OP_ADD
                : op == OP_TENSOR_SUB ? TENSOR_OP_SUB
                : op == OP_TENSOR_MUL ? TENSOR_OP_MUL : TENSOR_OP_DIV;
            Tensor *tensor = tensor_binary(
                tensor_op, left.kind == VALUE_TENSOR ? left.as.tensor : NULL,
                left.kind == VALUE_INT ? (double)left.as.integer : 0.0,
                right.kind == VALUE_TENSOR ? right.as.tensor : NULL,
                right.kind == VALUE_INT ? (double)right.as.integer : 0.0,
                error, sizeof(error));
            value_free(&left); value_free(&right);
            if (tensor == NULL) { vm_fail(vm, in, error); goto cleanup; }
            result = value_tensor(tensor);
            if (!push(vm, result)) { value_free(&result); goto memory_error; }
        } else if (op == OP_HALT) {
            *returned = value_null();
            ok = true;
            goto cleanup;
        }
    }
    *returned = value_null();
    ok = true;
    goto cleanup;
memory_error:
    vm_fail(vm, function->count > 0 ? &function->code[ip == 0 ? 0 : ip - 1]
                                    : NULL,
            "Out of memory");
cleanup:
    for (index = 0; index < function->local_count; ++index)
        value_free(&locals[index]);
    free(locals);
    return ok;
}

bool vm_run(const BytecodeProgram *program, const char *entry_path,
            FILE *output, RuntimeError *error) {
    VM vm;
    Value result = value_null();
    size_t index;
    memset(&vm, 0, sizeof(vm));
    memset(error, 0, sizeof(*error));
    vm.program = program;
    vm.output = output;
    vm.error = error;
    vm.path = entry_path;
    vm.globals = calloc(program->global_count, sizeof(*vm.globals));
    if (program->global_count > 0 && vm.globals == NULL) return false;
    for (index = 0; index < program->global_count; ++index)
        vm.globals[index] = value_null();
    if (!execute_function(&vm, 0, NULL, 0, &result)) {
        value_free(&result);
        for (index = 0; index < vm.count; ++index) value_free(&vm.stack[index]);
        for (index = 0; index < program->global_count; ++index)
            value_free(&vm.globals[index]);
        free(vm.globals); free(vm.stack);
        return false;
    }
    value_free(&result);
    for (index = 0; index < vm.count; ++index) value_free(&vm.stack[index]);
    for (index = 0; index < program->global_count; ++index)
        value_free(&vm.globals[index]);
    free(vm.globals); free(vm.stack);
    return true;
}
