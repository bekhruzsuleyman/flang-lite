
#include "interpreter.h"

#include "env.h"
#include "lexer.h"
#include "parser.h"
#include "runtime/native.h"
#include "runtime/stdlib.h"
#include "runtime/tensor.h"
#include "token.h"
#include "value.h"

#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef enum {
    MODULE_LOADING,
    MODULE_LOADED
} ModuleState;

struct ModuleValue {
    char *name;
    char *path;
    char *source;
    TokenArray tokens;
    AstProgram program;
    Env env;
    ModuleState state;
};

struct FunctionValue {
    const AstNode *declaration;
    Env *closure;
    ModuleValue *module;
};

typedef enum {
    EXEC_NORMAL,
    EXEC_RETURN,
    EXEC_ERROR
} ExecStatus;

typedef struct {
    ExecStatus status;
    Value value;
} ExecResult;

typedef struct {
    ModuleValue **modules;
    size_t module_count;
    size_t module_capacity;
    FunctionValue **functions;
    size_t function_count;
    size_t function_capacity;
    char *root_directory;
    ModuleValue *current_module;
    Env *current_env;
    size_t function_depth;
    FILE *output;
    RuntimeError *error;
} Interpreter;

static ExecResult execute_statement(Interpreter *interpreter,
                                    const AstNode *statement);
static ExecResult execute_block(Interpreter *interpreter, const AstNode *block);

static char *copy_string(const char *source) {
    size_t length = strlen(source);
    char *copy = malloc(length + 1);
    if (copy != NULL) memcpy(copy, source, length + 1);
    return copy;
}

static char *directory_name(const char *path) {
    const char *slash = strrchr(path, '/');
    const char *backslash = strrchr(path, '\\');
    const char *separator = slash;
    size_t length;
    char *directory;
    if (backslash != NULL && (separator == NULL || backslash > separator)) separator = backslash;
    if (separator == NULL) return copy_string(".");
    length = (size_t)(separator - path);
    if (length == 0) length = 1;
    directory = malloc(length + 1);
    if (directory == NULL) return NULL;
    memcpy(directory, path, length);
    directory[length] = '\0';
    return directory;
}

static char *module_path(const Interpreter *interpreter, const char *name) {
    size_t directory_length = strlen(interpreter->root_directory);
    size_t name_length = strlen(name);
    char *path = malloc(directory_length + name_length + 5);
    if (path == NULL) return NULL;
    snprintf(path, directory_length + name_length + 5, "%s/%s.fl",
             interpreter->root_directory, name);
    return path;
}

static char *read_source(const char *path) {
    FILE *file = fopen(path, "rb");
    long length;
    size_t bytes_read;
    char *source;
    if (file == NULL) return NULL;
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    source = malloc((size_t)length + 1);
    if (source == NULL) {
        fclose(file);
        return NULL;
    }
    bytes_read = fread(source, 1, (size_t)length, file);
    if (bytes_read != (size_t)length || ferror(file)) {
        free(source);
        fclose(file);
        return NULL;
    }
    source[bytes_read] = '\0';
    fclose(file);
    return source;
}

static bool runtime_fail(Interpreter *interpreter, const AstNode *node,
                         const char *message) {
    if (!interpreter->error->has_error) {
        interpreter->error->has_error = true;
        interpreter->error->line = node->line;
        interpreter->error->column = node->column;
        if (interpreter->current_module != NULL &&
            interpreter->current_module->path != NULL) {
            snprintf(interpreter->error->path, sizeof(interpreter->error->path),
                     "%s", interpreter->current_module->path);
        }
        snprintf(interpreter->error->message, sizeof(interpreter->error->message),
                 "%s", message);
        if (strstr(message, "backward() requires scalar tensor") != NULL)
            snprintf(interpreter->error->hint, sizeof(interpreter->error->hint),
                     "use tensor.sum().backward() or tensor.mean().backward()");
        else if (strstr(message, "grad graph") != NULL)
            snprintf(interpreter->error->hint, sizeof(interpreter->error->hint),
                     "construct the tensor with requires_grad set to true");
        else if (strstr(message, "gradient is not available") != NULL)
            snprintf(interpreter->error->hint, sizeof(interpreter->error->hint),
                     "call backward() on a scalar loss before reading .grad");
    }
    return false;
}

static bool module_syntax_fail(Interpreter *interpreter, const char *path,
                               size_t line, size_t column, const char *kind,
                               const char *message) {
    if (!interpreter->error->has_error) {
        interpreter->error->has_error = true;
        interpreter->error->line = line;
        interpreter->error->column = column;
        snprintf(interpreter->error->path, sizeof(interpreter->error->path), "%s", path);
        snprintf(interpreter->error->message, sizeof(interpreter->error->message),
                 "%.24s error in imported module: %.200s", kind, message);
    }
    return false;
}

static ExecResult exec_normal(void) {
    ExecResult result;
    result.status = EXEC_NORMAL;
    result.value = value_null();
    return result;
}

static ExecResult exec_return(Value value) {
    ExecResult result = {EXEC_RETURN, value};
    return result;
}

static ExecResult exec_error(void) {
    ExecResult result;
    result.status = EXEC_ERROR;
    result.value = value_null();
    return result;
}

static bool add_checked(int64_t left, int64_t right, int64_t *result) {
    if ((right > 0 && left > INT64_MAX - right) ||
        (right < 0 && left < INT64_MIN - right)) return false;
    *result = left + right;
    return true;
}

static bool subtract_checked(int64_t left, int64_t right, int64_t *result) {
    if ((right > 0 && left < INT64_MIN + right) ||
        (right < 0 && left > INT64_MAX + right)) return false;
    *result = left - right;
    return true;

}

static bool multiply_checked(int64_t left, int64_t right, int64_t *result) {
    if (left > 0) {
        if ((right > 0 && left > INT64_MAX / right) ||
            (right < 0 && right < INT64_MIN / left)) return false;
    } else if (left < 0) {
        if ((right > 0 && left < INT64_MIN / right) ||
            (right < 0 && left < INT64_MAX / right)) return false;
    }
    *result = left * right;
    return true;
}

static const char *value_type_name(ValueKind kind) {
    switch (kind) {
        case VALUE_INT: return "int";
        case VALUE_BOOL: return "bool";
        case VALUE_NULL: return "void";
        case VALUE_STRING: return "str";
        case VALUE_ARRAY: return "list";
        case VALUE_TENSOR: return "tensor";
        case VALUE_RANGE: return "range";
        case VALUE_FUNCTION: return "function";
        case VALUE_MODULE: return "module";
        case VALUE_NATIVE: return "native function";
    }
    return "unknown";
}

static const char *type_name(TypeKind type) {
    switch (type) {
        case TYPE_INT: return "int";
        case TYPE_BOOL: return "bool";
        case TYPE_VOID: return "void";
        case TYPE_STRING: return "str";
        case TYPE_LIST: return "list";
        case TYPE_TENSOR: return "tensor";
    }
    return "unknown";
}

static bool value_matches_type(Value value, TypeKind type) {
    return (type == TYPE_INT && value.kind == VALUE_INT) ||
           (type == TYPE_BOOL && value.kind == VALUE_BOOL) ||
           (type == TYPE_VOID && value.kind == VALUE_NULL) ||
           (type == TYPE_STRING && value.kind == VALUE_STRING) ||
           (type == TYPE_LIST && value.kind == VALUE_ARRAY) ||
           (type == TYPE_TENSOR && value.kind == VALUE_TENSOR);
}

static bool expect_binary_kind(Interpreter *interpreter, const AstNode *node,
                               Value left, Value right, ValueKind kind,
                               const char *expected) {
    if (left.kind != kind || right.kind != kind) {
        char message[256];
        snprintf(message, sizeof(message), "Operator '%s' expects %s and %s",
                 token_type_name(node->as.binary.operator_type), expected, expected);
        return runtime_fail(interpreter, node, message);
    }
    return true;
}

static bool append_function(Interpreter *interpreter, FunctionValue *function) {
    FunctionValue **new_functions;
    size_t new_capacity;
    if (interpreter->function_count == interpreter->function_capacity) {
        new_capacity = interpreter->function_capacity == 0
                           ? 8
                           : interpreter->function_capacity * 2;
        new_functions = realloc(interpreter->functions,
                                new_capacity * sizeof(*new_functions));
        if (new_functions == NULL) return false;
        interpreter->functions = new_functions;
        interpreter->function_capacity = new_capacity;
    }
    interpreter->functions[interpreter->function_count++] = function;
    return true;
}

static ModuleValue *find_module(const Interpreter *interpreter, const char *name) {
    size_t index;
    for (index = 0; index < interpreter->module_count; ++index) {
        if (strcmp(interpreter->modules[index]->name, name) == 0)
            return interpreter->modules[index];
    }
    return NULL;
}

static bool append_module(Interpreter *interpreter, ModuleValue *module) {
    ModuleValue **new_modules;
    size_t new_capacity;
    if (interpreter->module_count == interpreter->module_capacity) {
        new_capacity = interpreter->module_capacity == 0
                           ? 4
                           : interpreter->module_capacity * 2;
        new_modules = realloc(interpreter->modules,
                              new_capacity * sizeof(*new_modules));
        if (new_modules == NULL) return false;
        interpreter->modules = new_modules;
        interpreter->module_capacity = new_capacity;
    }
    interpreter->modules[interpreter->module_count++] = module;
    return true;
}

static bool print_value(FILE *output, const Value *value);

static bool print_value(FILE *output, const Value *value) {
    size_t index;
    switch (value->kind) {
        case VALUE_INT:
            return fprintf(output, "%" PRId64, value->as.integer) >= 0;
        case VALUE_BOOL:
            return fputs(value->as.boolean ? "true" : "false", output) != EOF;
        case VALUE_STRING:
            return fputs(value->as.string.data, output) != EOF;
        case VALUE_ARRAY:
            if (fputc('[', output) == EOF) return false;
            for (index = 0; index < value->as.array.count; ++index) {
                if (index > 0 && fputs(", ", output) == EOF) return false;
                if (!print_value(output, &value->as.array.items[index])) return false;
            }
            return fputc(']', output) != EOF;
        case VALUE_TENSOR:
            return tensor_print(output, value->as.tensor);
        default:
            return false;
    }
}


static bool is_builtin_name(const char *name) {
    return strcmp(name, "len") == 0 || strcmp(name, "range") == 0 ||
           strcmp(name, "tensor") == 0 || strcmp(name, "shape") == 0 ||
           strcmp(name, "rank") == 0 || strcmp(name, "zeros") == 0 ||
           strcmp(name, "ones") == 0;
}

static bool evaluate(Interpreter *interpreter, const AstNode *node, Value *result);

static bool call_builtin(Interpreter *interpreter, const AstNode *call,
                         const char *name, Value *result) {
    size_t count = call->as.call.argument_count;
    Value *arguments = NULL;
    size_t index;
    size_t expected = strcmp(name, "range") == 0 ? 2 : 1;
    bool success = false;
    char error[256] = {0};

    if ((strcmp(name, "tensor") == 0 && count != 1 && count != 2) ||
        (strcmp(name, "tensor") != 0 && count != expected)) {
        char message[256];
        if (strcmp(name, "tensor") == 0)
            snprintf(message, sizeof(message),
                     "Builtin 'tensor' expected 1 or 2 arguments but got %zu",
                     count);
        else
            snprintf(message, sizeof(message),
                     "Builtin '%s' expected %zu arguments but got %zu",
                     name, expected, count);
        return runtime_fail(interpreter, call, message);
    }
    arguments = calloc(count, sizeof(*arguments));
    if (arguments == NULL) return runtime_fail(interpreter, call, "Out of memory");
    for (index = 0; index < count; ++index) {
        if (!evaluate(interpreter, call->as.call.arguments[index], &arguments[index]))
            goto cleanup;
    }

    if (strcmp(name, "len") == 0) {
        if (arguments[0].kind == VALUE_STRING)
            *result = value_int((int64_t)arguments[0].as.string.length);
        else if (arguments[0].kind == VALUE_ARRAY)
            *result = value_int((int64_t)arguments[0].as.array.count);
        else {
            runtime_fail(interpreter, call, "len() expects a string or list");
            goto cleanup;
        }
        success = true;
    } else if (strcmp(name, "range") == 0) {
        if (arguments[0].kind != VALUE_INT || arguments[1].kind != VALUE_INT) {
            runtime_fail(interpreter, call, "range() expects int and int");
            goto cleanup;
        }
        *result = value_range(arguments[0].as.integer, arguments[1].as.integer);
        success = true;
    } else if (strcmp(name, "tensor") == 0) {
        success = native_call(NATIVE_TENSOR_CREATE, arguments, count, result,
                              error, sizeof(error));
    } else if (strcmp(name, "shape") == 0) {
        if (arguments[0].kind != VALUE_TENSOR) {
            runtime_fail(interpreter, call, "shape() expects a tensor");
            goto cleanup;
        }
        value_array(result);
        for (index = 0; index < (size_t)arguments[0].as.tensor->rank; ++index) {
            if (!value_array_append(result,
                                    value_int(arguments[0].as.tensor->shape[index]))) {
                value_free(result);

                runtime_fail(interpreter, call, "Out of memory");
                goto cleanup;
            }
        }
        success = true;
    } else if (strcmp(name, "rank") == 0) {
        if (arguments[0].kind != VALUE_TENSOR) {
            runtime_fail(interpreter, call, "rank() expects a tensor");
            goto cleanup;
        }
        *result = value_int((int64_t)arguments[0].as.tensor->rank);
        success = true;
    } else {
        success = native_call(strcmp(name, "ones") == 0
                                  ? NATIVE_TENSOR_ONES
                                  : NATIVE_TENSOR_ZEROS,
                              arguments, count, result, error, sizeof(error));
    }

    if (!success && !interpreter->error->has_error && error[0] != '\0')
        runtime_fail(interpreter, call, error);

cleanup:
    for (index = 0; index < count; ++index) value_free(&arguments[index]);
    free(arguments);
    return success;
}

static bool apply_integer_operator(Interpreter *interpreter, const AstNode *node,
                                   TokenType operator_type, int64_t left,
                                   int64_t right, int64_t *result) {
    if (operator_type == TOKEN_PLUS) {
        if (!add_checked(left, right, result))
            return runtime_fail(interpreter, node, "Integer overflow");
    } else if (operator_type == TOKEN_MINUS) {
        if (!subtract_checked(left, right, result))
            return runtime_fail(interpreter, node, "Integer overflow");
    } else if (operator_type == TOKEN_STAR) {
        if (!multiply_checked(left, right, result))
            return runtime_fail(interpreter, node, "Integer overflow");
    } else {
        if (right == 0) return runtime_fail(interpreter, node, "Division by zero");
        if (left == INT64_MIN && right == -1)
            return runtime_fail(interpreter, node, "Integer overflow");
        *result = left / right;
    }
    return true;
}


static bool tensor_arithmetic(Interpreter *interpreter, const AstNode *node,
                              TokenType operator_type, const Value *left,
                              const Value *right, Value *result) {
    TensorOp op;
    Tensor *output;
    char error[256] = {0};
    if ((left->kind != VALUE_TENSOR && left->kind != VALUE_INT) ||
        (right->kind != VALUE_TENSOR && right->kind != VALUE_INT))
        return false;
    op = operator_type == TOKEN_PLUS ? TENSOR_OP_ADD
         : operator_type == TOKEN_MINUS ? TENSOR_OP_SUB
         : operator_type == TOKEN_STAR ? TENSOR_OP_MUL
                                       : TENSOR_OP_DIV;
    output = tensor_binary(op,
                           left->kind == VALUE_TENSOR ? left->as.tensor : NULL,
                           left->kind == VALUE_INT ? (double)left->as.integer : 0.0,
                           right->kind == VALUE_TENSOR ? right->as.tensor : NULL,
                           right->kind == VALUE_INT ? (double)right->as.integer : 0.0,
                           error, sizeof(error));
    if (output == NULL) return runtime_fail(interpreter, node, error);
    *result = value_tensor(output);
    return true;
}

static bool evaluate_binary(Interpreter *interpreter, const AstNode *node,
                            Value *result) {
    Value left = value_null();
    Value right = value_null();
    TokenType operator_type = node->as.binary.operator_type;
    bool success = false;
    int64_t integer;

    if (!evaluate(interpreter, node->as.binary.left, &left) ||
        !evaluate(interpreter, node->as.binary.right, &right))
        goto cleanup;

    if (operator_type == TOKEN_PLUS && left.kind == VALUE_STRING &&
        right.kind == VALUE_STRING) {
        size_t length;
        if (left.as.string.length > SIZE_MAX - right.as.string.length - 1) {
            runtime_fail(interpreter, node, "String is too large");
            goto cleanup;
        }
        length = left.as.string.length + right.as.string.length;
        *result = value_null();
        result->kind = VALUE_STRING;
        result->as.string.data = malloc(length + 1);
        if (result->as.string.data == NULL) {
            runtime_fail(interpreter, node, "Out of memory");
            goto cleanup;
        }
        memcpy(result->as.string.data, left.as.string.data, left.as.string.length);
        memcpy(result->as.string.data + left.as.string.length,
               right.as.string.data, right.as.string.length + 1);
        result->as.string.length = length;
        success = true;
        goto cleanup;
    }
    if ((operator_type == TOKEN_PLUS || operator_type == TOKEN_MINUS ||
         operator_type == TOKEN_STAR || operator_type == TOKEN_SLASH) &&
        (left.kind == VALUE_TENSOR || right.kind == VALUE_TENSOR)) {
        success = tensor_arithmetic(interpreter, node, operator_type,
                                    &left, &right, result);
        if (!success && !interpreter->error->has_error)
            runtime_fail(interpreter, node,
                         "Tensor arithmetic expects tensor and tensor or tensor and int");
        goto cleanup;
    }
    if (operator_type == TOKEN_PLUS || operator_type == TOKEN_MINUS ||
        operator_type == TOKEN_STAR || operator_type == TOKEN_SLASH) {
        if (!expect_binary_kind(interpreter, node, left, right, VALUE_INT, "int"))
            goto cleanup;
        if (!apply_integer_operator(interpreter, node, operator_type,
                                    left.as.integer, right.as.integer, &integer))
            goto cleanup;
        *result = value_int(integer);
        success = true;
    } else if (operator_type == TOKEN_EQUAL_EQUAL || operator_type == TOKEN_BANG_EQUAL) {
        bool equal;
        if (left.kind != right.kind ||
            (left.kind != VALUE_INT && left.kind != VALUE_BOOL &&
             left.kind != VALUE_STRING)) {
            char message[256];
            snprintf(message, sizeof(message),
                     "Operator '%s' cannot compare %s and %s",
                     token_type_name(operator_type), value_type_name(left.kind),
                     value_type_name(right.kind));
            runtime_fail(interpreter, node, message);
            goto cleanup;
        }
        if (left.kind == VALUE_INT)
            equal = left.as.integer == right.as.integer;
        else if (left.kind == VALUE_BOOL)
            equal = left.as.boolean == right.as.boolean;
        else
            equal = strcmp(left.as.string.data, right.as.string.data) == 0;
        *result = value_bool(operator_type == TOKEN_EQUAL_EQUAL ? equal : !equal);
        success = true;
    } else if (operator_type == TOKEN_LESS ||
               operator_type == TOKEN_LESS_EQUAL ||
               operator_type == TOKEN_GREATER ||
               operator_type == TOKEN_GREATER_EQUAL) {
        if (!expect_binary_kind(interpreter, node, left, right, VALUE_INT, "int"))
            goto cleanup;
        if (operator_type == TOKEN_LESS)
            *result = value_bool(left.as.integer < right.as.integer);
        else if (operator_type == TOKEN_LESS_EQUAL)
            *result = value_bool(left.as.integer <= right.as.integer);
        else if (operator_type == TOKEN_GREATER)
            *result = value_bool(left.as.integer > right.as.integer);
        else
            *result = value_bool(left.as.integer >= right.as.integer);
        success = true;
    } else if (operator_type == TOKEN_AND || operator_type == TOKEN_OR) {
        if (!expect_binary_kind(interpreter, node, left, right, VALUE_BOOL, "bool"))
            goto cleanup;
        *result = value_bool(operator_type == TOKEN_AND
                                 ? left.as.boolean && right.as.boolean
                                 : left.as.boolean || right.as.boolean);
        success = true;
    } else {
        runtime_fail(interpreter, node, "Unknown binary operator");
    }
cleanup:
    value_free(&left);
    value_free(&right);
    return success;
}

static bool evaluate_index(Interpreter *interpreter, const AstNode *node,
                           Value *result) {
    Value object = value_null();
    Value index = value_null();
    bool success = false;
    if (!evaluate(interpreter, node->as.index.object, &object) ||
        !evaluate(interpreter, node->as.index.index, &index))
        goto cleanup;
    if (index.kind != VALUE_INT) {
        runtime_fail(interpreter, node, "Index must be int");
        goto cleanup;
    }
    if (index.as.integer < 0) {
        runtime_fail(interpreter, node, "Index out of bounds");
        goto cleanup;
    }
    if (object.kind == VALUE_ARRAY) {
        if ((uint64_t)index.as.integer >= object.as.array.count) {
            runtime_fail(interpreter, node, "Array index out of bounds");
            goto cleanup;
        }
        if (!value_clone(&object.as.array.items[(size_t)index.as.integer], result)) {
            runtime_fail(interpreter, node, "Out of memory");
            goto cleanup;
        }
        success = true;
    } else if (object.kind == VALUE_STRING) {
        char character[2];
        if ((uint64_t)index.as.integer >= object.as.string.length) {
            runtime_fail(interpreter, node, "String index out of bounds");
            goto cleanup;
        }
        character[0] = object.as.string.data[(size_t)index.as.integer];
        character[1] = '\0';
        if (!value_string(character, result)) {
            runtime_fail(interpreter, node, "Out of memory");
            goto cleanup;
        }
        success = true;
    } else if (object.kind == VALUE_TENSOR) {
        Tensor *tensor = object.as.tensor;
        char error[256] = {0};
        if (tensor->rank == 0 || index.as.integer >= tensor->shape[0]) {
            runtime_fail(interpreter, node, "Tensor index out of bounds");
            goto cleanup;
        }
        if (tensor->rank == 1) {
            *result = value_int((int64_t)tensor->data[index.as.integer]);
        } else {
            Tensor *slice = tensor_copy_slice(tensor, index.as.integer,
                                              error, sizeof(error));
            if (slice == NULL) {
                runtime_fail(interpreter, node, error[0] == '\0'
                                                   ? "Tensor index out of bounds"
                                                   : error);
                goto cleanup;
            }
            *result = value_tensor(slice);
        }
        success = true;
    } else {
        runtime_fail(interpreter, node, "Value is not indexable");
    }
cleanup:
    value_free(&object);
    value_free(&index);
    return success;
}

static bool call_function(Interpreter *interpreter, const AstNode *call,
                          FunctionValue *function, Value *result) {
    const AstNode *declaration = function->declaration;
    size_t count = declaration->as.function_declaration.parameter_count;
    Value *arguments = count == 0 ? NULL : calloc(count, sizeof(*arguments));
    Env call_env;
    Env *saved_env;
    ModuleValue *saved_module;
    ExecResult execution;
    size_t index;
    bool success = false;
    if (call->as.call.argument_count != count) {
        char message[256];
        snprintf(message, sizeof(message),
                 "Function '%s' expected %zu arguments but got %zu",
                 declaration->as.function_declaration.name, count,
                 call->as.call.argument_count);
        return runtime_fail(interpreter, call, message);
    }
    if (count > 0 && arguments == NULL)
        return runtime_fail(interpreter, call, "Out of memory");
    for (index = 0; index < count; ++index) {
        arguments[index] = value_null();
        if (!evaluate(interpreter, call->as.call.arguments[index],
                      &arguments[index]))
            goto cleanup;
        if (!value_matches_type(
                arguments[index],
                declaration->as.function_declaration.parameters[index].type)) {
            char message[256];
            snprintf(message, sizeof(message),
                     "Function '%s' expected argument %zu to be %s, got %s",
                     declaration->as.function_declaration.name, index + 1,
                     type_name(declaration->as.function_declaration
                                   .parameters[index].type),
                     value_type_name(arguments[index].kind));
            runtime_fail(interpreter, call->as.call.arguments[index], message);
            goto cleanup;
        }
    }
    env_init(&call_env, function->closure);
    for (index = 0; index < count; ++index) {
        if (!env_define(&call_env,
                        declaration->as.function_declaration.parameters[index].name,
                        arguments[index], false)) {
            runtime_fail(interpreter, call, "Out of memory");
            env_free(&call_env);
            goto cleanup;
        }
    }
    saved_env = interpreter->current_env;
    saved_module = interpreter->current_module;
    interpreter->current_env = &call_env;
    interpreter->current_module = function->module;
    ++interpreter->function_depth;
    execution = execute_block(
        interpreter, declaration->as.function_declaration.body);
    --interpreter->function_depth;
    interpreter->current_env = saved_env;
    interpreter->current_module = saved_module;
    env_free(&call_env);
    if (execution.status == EXEC_ERROR) {
        value_free(&execution.value);
        goto cleanup;
    }
    if (declaration->as.function_declaration.return_type == TYPE_VOID) {
        if (execution.status == EXEC_RETURN && execution.value.kind != VALUE_NULL) {
            runtime_fail(interpreter, call, "Void function returned a value");
            value_free(&execution.value);
            goto cleanup;
        }
        value_free(&execution.value);
        *result = value_null();
        success = true;
    } else if (execution.status != EXEC_RETURN) {
        char message[256];
        snprintf(message, sizeof(message), "Function '%s' did not return a value",
                 declaration->as.function_declaration.name);
        runtime_fail(interpreter, call, message);
        value_free(&execution.value);
    } else if (!value_matches_type(
                   execution.value,
                   declaration->as.function_declaration.return_type)) {
        char message[256];
        snprintf(message, sizeof(message), "Function '%s' returned %s instead of %s",
                 declaration->as.function_declaration.name,
                 value_type_name(execution.value.kind),
                 type_name(declaration->as.function_declaration.return_type));
        runtime_fail(interpreter, call, message);
        value_free(&execution.value);
    } else {
        *result = execution.value;
        success = true;
    }
cleanup:
    for (index = 0; index < count; ++index) value_free(&arguments[index]);
    free(arguments);
    return success;
}

static bool call_native_value(Interpreter *interpreter, const AstNode *call,
                              NativeId id, Value *result) {
    size_t count = call->as.call.argument_count;
    Value *arguments = count == 0 ? NULL : calloc(count, sizeof(*arguments));
    size_t index;
    char error[256] = {0};
    bool success = false;
    if (count > 0 && arguments == NULL)
        return runtime_fail(interpreter, call, "Out of memory");
    for (index = 0; index < count; ++index) {
        arguments[index] = value_null();
        if (!evaluate(interpreter, call->as.call.arguments[index],
                      &arguments[index]))
            goto cleanup;
    }
    success = native_call(id, arguments, count, result, error, sizeof(error));
    if (!success) runtime_fail(interpreter, call, error);
cleanup:
    for (index = 0; index < count; ++index) value_free(&arguments[index]);
    free(arguments);
    return success;
}

static bool call_tensor_method(Interpreter *interpreter, const AstNode *call,
                               Tensor *tensor, AstNativeMethod method,
                               Value *result) {
    Tensor *output;
    char error[256] = {0};
    if (method == AST_NATIVE_TENSOR_SUM || method == AST_NATIVE_TENSOR_MEAN) {
        output = tensor_reduce(tensor, method == AST_NATIVE_TENSOR_MEAN,
                               error, sizeof(error));
        if (output == NULL) return runtime_fail(interpreter, call, error);
        *result = value_tensor(output);
        return true;
    }
    if (method == AST_NATIVE_TENSOR_BACKWARD) {
        if (!tensor_backward(tensor, error, sizeof(error)))
            return runtime_fail(interpreter, call, error);
        *result = value_null();
        return true;
    }
    if (method == AST_NATIVE_TENSOR_ZERO_GRAD) {
        tensor_zero_grad(tensor);
        *result = value_null();
        return true;
    }
    return runtime_fail(interpreter, call, "Unknown lowered tensor method");
}

static bool evaluate(Interpreter *interpreter, const AstNode *node,
                     Value *result) {
    Value left = value_null();
    Value right = value_null();
    switch (node->kind) {
        case AST_NUMBER:
            *result = value_int(node->as.number);
            return true;
        case AST_BOOL:
            *result = value_bool(node->as.boolean);
            return true;
        case AST_STRING:
            if (!value_string(node->as.string, result))
                return runtime_fail(interpreter, node, "Out of memory");
            return true;
        case AST_ARRAY: {
            size_t index;
            value_array(result);
            for (index = 0; index < node->as.array.count; ++index) {
                Value element = value_null();
                if (!evaluate(interpreter, node->as.array.elements[index], &element)) {
                    value_free(result);
                    return false;
                }
                if (!value_array_append(result, element)) {
                    value_free(&element);
                    value_free(result);
                    return runtime_fail(interpreter, node, "Out of memory");
                }
            }
            return true;
        }
        case AST_IDENTIFIER: {
            Value *slot = env_get_slot(interpreter->current_env,
                                       node->as.identifier);
            if (slot == NULL) {
                char message[256];
                snprintf(message, sizeof(message), "Undefined variable '%s'",
                         node->as.identifier);
                return runtime_fail(interpreter, node, message);
            }
            if (!value_clone(slot, result))
                return runtime_fail(interpreter, node, "Out of memory");
            return true;
        }
        case AST_BINARY:
            return evaluate_binary(interpreter, node, result);
        case AST_INDEX:
            return evaluate_index(interpreter, node, result);
        case AST_MEMBER: {
            const Symbol *symbol;
            if (!evaluate(interpreter, node->as.member.object, &left)) return false;
            if (left.kind == VALUE_MODULE) {
                symbol = env_find_local(&left.as.module->env,
                                        node->as.member.member);
                if (symbol == NULL || !symbol->is_public) {
                    char message[256];
                    snprintf(message, sizeof(message),
                             "Symbol '%s' not found or private in module '%s'",
                             node->as.member.member, left.as.module->name);
                    value_free(&left);
                    return runtime_fail(interpreter, node, message);
                }
                if (!value_clone(&symbol->value, result)) {
                    value_free(&left);
                    return runtime_fail(interpreter, node, "Out of memory");
                }
            } else if ((left.kind == VALUE_ARRAY || left.kind == VALUE_STRING) &&
                       strcmp(node->as.member.member, "len") == 0) {
                *result = value_int((int64_t)(left.kind == VALUE_ARRAY
                                                  ? left.as.array.count
                                                  : left.as.string.length));
            } else if (left.kind == VALUE_TENSOR &&
                       strcmp(node->as.member.member, "rank") == 0) {
                *result = value_int((int64_t)left.as.tensor->rank);
            } else if (left.kind == VALUE_TENSOR &&
                       strcmp(node->as.member.member, "len") == 0) {
                *result = value_int(left.as.tensor->len);
            } else if (left.kind == VALUE_TENSOR &&
                       strcmp(node->as.member.member, "shape") == 0) {
                int dimension;
                value_array(result);
                for (dimension = 0; dimension < left.as.tensor->rank; ++dimension) {
                    if (!value_array_append(
                            result, value_int(left.as.tensor->shape[dimension]))) {
                        value_free(result);
                        value_free(&left);
                        return runtime_fail(interpreter, node, "Out of memory");
                    }
                }
            } else if (left.kind == VALUE_TENSOR &&
                       strcmp(node->as.member.member, "requires_grad") == 0) {
                *result = value_bool(left.as.tensor->requires_grad);
            } else if (left.kind == VALUE_TENSOR &&
                       strcmp(node->as.member.member, "grad") == 0) {
                char error[256] = {0};
                Tensor *gradient = tensor_gradient_copy(
                    left.as.tensor, error, sizeof(error));
                if (gradient == NULL) {
                    value_free(&left);
                    return runtime_fail(interpreter, node, error);
                }
                *result = value_tensor(gradient);
            } else {
                char message[256];
                snprintf(message, sizeof(message), "Type %s has no property '%s'",
                         value_type_name(left.kind), node->as.member.member);
                value_free(&left);
                return runtime_fail(interpreter, node, message);
            }
            value_free(&left);
            return true;
        }
        case AST_CALL: {
            if (node->as.call.callee->kind == AST_MEMBER) {
                const AstNode *member = node->as.call.callee;
                if (!evaluate(interpreter, member->as.member.object, &right))
                    return false;
                if (right.kind == VALUE_TENSOR) {
                    bool ok = call_tensor_method(interpreter, node, right.as.tensor,
                                                 node->as.call.native_method, result);
                    value_free(&right);
                    return ok;
                }
                if (right.kind == VALUE_MODULE) {
                    const Symbol *symbol = env_find_local(
                        &right.as.module->env, member->as.member.member);
                    if (symbol == NULL || !symbol->is_public) {
                        char message[256];
                        snprintf(message, sizeof(message),
                                 "Symbol '%s' not found or private in module '%s'",
                                 member->as.member.member, right.as.module->name);
                        value_free(&right);
                        return runtime_fail(interpreter, member, message);
                    }
                    if (!value_clone(&symbol->value, &left)) {
                        value_free(&right);
                        return runtime_fail(interpreter, node, "Out of memory");
                    }
                    value_free(&right);
                } else {
                    char message[256];
                    snprintf(message, sizeof(message), "Type %s has no method '%s'",
                             value_type_name(right.kind), member->as.member.member);
                    value_free(&right);
                    return runtime_fail(interpreter, member, message);
                }
            } else if (node->as.call.callee->kind == AST_IDENTIFIER) {
                const char *name = node->as.call.callee->as.identifier;
                Value *slot = env_get_slot(interpreter->current_env, name);
                if (slot == NULL && is_builtin_name(name))
                    return call_builtin(interpreter, node, name, result);
                if (slot == NULL) {
                    char message[256];
                    snprintf(message, sizeof(message), "Undefined function '%s'", name);
                    return runtime_fail(interpreter, node->as.call.callee, message);
                }
                if (!value_clone(slot, &left))
                    return runtime_fail(interpreter, node, "Out of memory");
            } else if (!evaluate(interpreter, node->as.call.callee, &left)) {
                return false;
            }
            if (left.kind == VALUE_NATIVE) {
                NativeId id = (NativeId)left.as.native_id;
                value_free(&left);
                return call_native_value(interpreter, node, id, result);
            }
            if (left.kind != VALUE_FUNCTION) {
                char message[256];
                snprintf(message, sizeof(message), "Value of type %s is not callable",
                         value_type_name(left.kind));
                value_free(&left);
                return runtime_fail(interpreter, node, message);
            }
            {
                FunctionValue *function = left.as.function;
                bool ok = call_function(interpreter, node, function, result);
                value_free(&left);
                return ok;
            }
        }
        case AST_UNARY:
            if (!evaluate(interpreter, node->as.unary.operand, &right)) return false;
            if (node->as.unary.operator_type == TOKEN_MINUS) {
                if (right.kind != VALUE_INT || right.as.integer == INT64_MIN) {
                    value_free(&right);
                    return runtime_fail(interpreter, node,
                                        right.kind == VALUE_INT
                                            ? "Integer overflow"
                                            : "Operator '-' expects int");
                }
                *result = value_int(-right.as.integer);
                value_free(&right);
                return true;
            }
            if (node->as.unary.operator_type == TOKEN_NOT) {
                if (right.kind != VALUE_BOOL) {
                    value_free(&right);
                    return runtime_fail(interpreter, node,
                                        "Operator 'not' expects bool");
                }
                *result = value_bool(!right.as.boolean);
                value_free(&right);
                return true;
            }
            value_free(&right);
            return runtime_fail(interpreter, node, "Unknown unary operator");
        default:
            return runtime_fail(interpreter, node,
                                "Statement cannot be used as an expression");
    }
}

static bool require_condition(Interpreter *interpreter,
                              const AstNode *condition, Value value) {
    if (value.kind != VALUE_BOOL)
        return runtime_fail(interpreter, condition, "Condition must be bool");
    return true;
}

static ExecResult execute_program(Interpreter *interpreter,
                                  const AstProgram *program) {
    size_t index;
    for (index = 0; index < program->count; ++index) {
        ExecResult result = execute_statement(
            interpreter, program->statements[index]);
        if (result.status != EXEC_NORMAL) return result;
    }
    return exec_normal();
}

static ModuleValue *load_module(Interpreter *interpreter, const char *name,
                                const AstNode *import_node) {
    ModuleValue *module = find_module(interpreter, name);
    LexerError lexer_error;
    ParserError parser_error;
    ModuleValue *saved_module;
    Env *saved_env;
    ExecResult execution;
    if (module != NULL) {
        if (module->state == MODULE_LOADING) {
            char message[256];
            snprintf(message, sizeof(message),
                     "Circular import detected involving module '%s'", name);
            runtime_fail(interpreter, import_node, message);
            return NULL;
        }
        return module;
    }
    module = calloc(1, sizeof(*module));
    if (module == NULL) {
        runtime_fail(interpreter, import_node, "Out of memory");
        return NULL;
    }
    module->name = copy_string(name);
    module->path = module_path(interpreter, name);
    module->state = MODULE_LOADING;
    env_init(&module->env, NULL);
    if (module->name == NULL || module->path == NULL ||
        !append_module(interpreter, module)) {
        free(module->name);
        free(module->path);
        env_free(&module->env);
        free(module);
        runtime_fail(interpreter, import_node, "Out of memory");
        return NULL;
    }

    module->source = read_source(module->path);
    if (module->source == NULL) {
        char *standard_path = stdlib_module_path(name);
        char *standard_source = standard_path == NULL
                                    ? NULL
                                    : read_source(standard_path);
        if (standard_source != NULL) {
            free(module->path);
            module->path = standard_path;
            module->source = standard_source;
        } else {
            free(standard_path);
        }
    }
    if (module->source == NULL) {
        if (stdlib_is_native_module(name)) {
            size_t native_index;
            free(module->path);
            module->path = copy_string("<native:tensor>");
            module->source = copy_string("");
            if (module->path == NULL || module->source == NULL) {
                runtime_fail(interpreter, import_node, "Out of memory");
                return NULL;
            }
            for (native_index = 0; native_index < native_symbol_count();
                 ++native_index) {
                const NativeSymbol *symbol = native_symbol_at(native_index);
                if (strcmp(symbol->module, name) == 0 &&
                    !env_define(&module->env, symbol->name,
                                value_native((int)symbol->id), true)) {
                    runtime_fail(interpreter, import_node, "Out of memory");
                    return NULL;
                }
            }
            module->state = MODULE_LOADED;
            return module;
        }
        char message[256];
        snprintf(message, sizeof(message), "Module '%s' not found at '%s'",
                 name, module->path);
        runtime_fail(interpreter, import_node, message);
        return NULL;
    }
    if (!lexer_scan(module->source, &module->tokens, &lexer_error)) {
        module_syntax_fail(interpreter, module->path, lexer_error.line,
                           lexer_error.column, "Lexer", lexer_error.message);
        return NULL;
    }
    if (!parser_parse(&module->tokens, &module->program, &parser_error)) {
        module_syntax_fail(interpreter, module->path, parser_error.line,
                           parser_error.column, "Parser", parser_error.message);
        return NULL;
    }

    saved_module = interpreter->current_module;
    saved_env = interpreter->current_env;
    interpreter->current_module = module;
    interpreter->current_env = &module->env;
    execution = execute_program(interpreter, &module->program);
    interpreter->current_module = saved_module;

    interpreter->current_env = saved_env;
    if (execution.status == EXEC_ERROR) {
        value_free(&execution.value);
        return NULL;
    }
    if (execution.status == EXEC_RETURN) {
        value_free(&execution.value);
        runtime_fail(interpreter, import_node, "Cannot return outside of function");
        return NULL;
    }
    module->state = MODULE_LOADED;
    return module;
}

static ExecResult execute_block(Interpreter *interpreter, const AstNode *block) {
    size_t index;
    for (index = 0; index < block->as.block.count; ++index) {
        ExecResult result = execute_statement(interpreter, block->as.block.statements[index]);
        if (result.status != EXEC_NORMAL) return result;
    }
    return exec_normal();
}

static bool annotation_matches(const char *annotation, Value value) {
    if (strcmp(annotation, "int") == 0) return value.kind == VALUE_INT;
    if (strcmp(annotation, "bool") == 0) return value.kind == VALUE_BOOL;
    if (strcmp(annotation, "str") == 0) return value.kind == VALUE_STRING;
    if (strcmp(annotation, "list") == 0) return value.kind == VALUE_ARRAY;
    if (strcmp(annotation, "tensor") == 0) return value.kind == VALUE_TENSOR;
    return false;
}

static Value *resolve_index_slot(Interpreter *interpreter,
                                 const AstNode *target) {
    Value *container;
    Value index = value_null();
    if (target->kind != AST_INDEX) {
        runtime_fail(interpreter, target, "Invalid assignment target");
        return NULL;
    }
    if (target->as.index.object->kind == AST_IDENTIFIER) {
        container = env_get_slot(interpreter->current_env,
                                 target->as.index.object->as.identifier);
        if (container == NULL) {
            char message[256];
            snprintf(message, sizeof(message), "Undefined variable '%s'",
                     target->as.index.object->as.identifier);
            runtime_fail(interpreter, target, message);
            return NULL;
        }
    } else if (target->as.index.object->kind == AST_INDEX) {
        container = resolve_index_slot(interpreter, target->as.index.object);
        if (container == NULL) return NULL;
    } else {
        runtime_fail(interpreter, target, "Invalid assignment target");
        return NULL;
    }
    if (!evaluate(interpreter, target->as.index.index, &index)) return NULL;
    if (index.kind != VALUE_INT) {
        value_free(&index);
        runtime_fail(interpreter, target, "Array index must be int");
        return NULL;
    }
    if (container->kind != VALUE_ARRAY) {
        value_free(&index);
        runtime_fail(interpreter, target, "Index assignment target must be a list");
        return NULL;
    }
    if (index.as.integer < 0 ||
        (uint64_t)index.as.integer >= container->as.array.count) {
        value_free(&index);
        runtime_fail(interpreter, target, "Array index out of bounds");
        return NULL;
    }
    container = &container->as.array.items[(size_t)index.as.integer];
    value_free(&index);
    return container;
}

static ExecResult execute_statement(Interpreter *interpreter,
                                    const AstNode *statement) {
    Value value = value_null();
    switch (statement->kind) {
        case AST_ASSIGNMENT:
            if (!evaluate(interpreter, statement->as.assignment.value, &value))
                return exec_error();
            if (statement->as.assignment.annotation != NULL &&
                !annotation_matches(statement->as.assignment.annotation, value)) {
                char message[256];
                snprintf(message, sizeof(message),
                         "Variable '%s' declared as %s cannot receive %s",
                         statement->as.assignment.name,
                         statement->as.assignment.annotation,
                         value_type_name(value.kind));
                value_free(&value);
                runtime_fail(interpreter, statement, message);
                return exec_error();
            }
            if (statement->as.assignment.is_declaration) {
                if (!env_define(interpreter->current_env,
                                statement->as.assignment.name, value,
                                statement->as.assignment.is_public)) {
                    value_free(&value);
                    runtime_fail(interpreter, statement, "Out of memory");
                    return exec_error();
                }
            } else if (env_get_slot(interpreter->current_env,
                                    statement->as.assignment.name) != NULL) {
                if (!env_assign(interpreter->current_env,
                                statement->as.assignment.name, value)) {
                    value_free(&value);
                    runtime_fail(interpreter, statement, "Out of memory");
                    return exec_error();
                }
            } else {
                if (!env_define(interpreter->current_env,
                                statement->as.assignment.name, value,
                                statement->as.assignment.is_public)) {
                    value_free(&value);
                    runtime_fail(interpreter, statement, "Out of memory");
                    return exec_error();
                }
            }
            value_free(&value);
            return exec_normal();
        case AST_INDEX_ASSIGNMENT: {
            Value *slot = resolve_index_slot(interpreter,
                                             statement->as.index_assignment.target);
            Value replacement;
            if (slot == NULL) return exec_error();
            if (!evaluate(interpreter, statement->as.index_assignment.value, &value))
                return exec_error();
            if (!value_clone(&value, &replacement)) {
                value_free(&value);
                runtime_fail(interpreter, statement, "Out of memory");
                return exec_error();
            }
            value_free(slot);
            *slot = replacement;
            value_free(&value);
            return exec_normal();
        }
        case AST_FUNCTION_DECLARATION: {
            FunctionValue *function = malloc(sizeof(*function));
            if (function == NULL) {
                runtime_fail(interpreter, statement, "Out of memory");
                return exec_error();
            }
            function->declaration = statement;
            function->closure = interpreter->current_env;
            function->module = interpreter->current_module;
            if (!append_function(interpreter, function) ||
                !env_define(interpreter->current_env,
                            statement->as.function_declaration.name,
                            value_function(function),
                            statement->as.function_declaration.is_public)) {
                if (interpreter->function_count == 0 ||
                    interpreter->functions[interpreter->function_count - 1] != function)
                    free(function);
                runtime_fail(interpreter, statement, "Out of memory");
                return exec_error();
            }
            return exec_normal();
        }
        case AST_RETURN:
            if (interpreter->function_depth == 0) {
                runtime_fail(interpreter, statement, "Cannot return outside of function");
                return exec_error();
            }
            if (statement->as.return_statement.value == NULL)
                return exec_return(value_null());
            if (!evaluate(interpreter, statement->as.return_statement.value, &value))
                return exec_error();
            return exec_return(value);
        case AST_PRINT:
            if (!evaluate(interpreter, statement->as.print_statement.expression, &value))
                return exec_error();
            if (value.kind == VALUE_NULL || value.kind == VALUE_FUNCTION ||
                value.kind == VALUE_MODULE || value.kind == VALUE_RANGE) {
                char message[256];
                snprintf(message, sizeof(message), "Cannot print value of type %s",
                         value_type_name(value.kind));
                value_free(&value);
                runtime_fail(interpreter, statement, message);
                return exec_error();
            }
            if (!print_value(interpreter->output, &value) ||
                fputc('\n', interpreter->output) == EOF) {
                value_free(&value);
                runtime_fail(interpreter, statement, "Unable to write output");
                return exec_error();
            }
            value_free(&value);
            return exec_normal();
        case AST_EXPRESSION_STATEMENT:
            if (!evaluate(interpreter,
                          statement->as.expression_statement.expression, &value))
                return exec_error();
            value_free(&value);
            return exec_normal();

        case AST_BLOCK:
            return execute_block(interpreter, statement);
        case AST_IF:
            if (!evaluate(interpreter, statement->as.if_statement.condition, &value))
                return exec_error();
            if (!require_condition(interpreter,
                                   statement->as.if_statement.condition, value)) {
                value_free(&value);
                return exec_error();
            }
            if (value.as.boolean) {
                value_free(&value);
                return execute_block(interpreter,
                                     statement->as.if_statement.then_block);
            }
            value_free(&value);
            if (statement->as.if_statement.else_block != NULL)
                return execute_block(interpreter,
                                     statement->as.if_statement.else_block);
            return exec_normal();
        case AST_WHILE:
            for (;;) {
                ExecResult body_result;
                if (!evaluate(interpreter,
                              statement->as.while_statement.condition, &value))
                    return exec_error();
                if (!require_condition(interpreter,
                                       statement->as.while_statement.condition, value)) {
                    value_free(&value);
                    return exec_error();
                }
                if (!value.as.boolean) {
                    value_free(&value);
                    return exec_normal();
                }
                value_free(&value);
                body_result = execute_block(interpreter,
                                            statement->as.while_statement.body);
                if (body_result.status != EXEC_NORMAL) return body_result;
            }
        case AST_FOR: {
            Env loop_env;
            Env *saved_env;
            ExecResult body_result = exec_normal();
            size_t index;
            if (!evaluate(interpreter, statement->as.for_statement.iterable, &value))
                return exec_error();
            if (value.kind != VALUE_ARRAY && value.kind != VALUE_RANGE) {
                value_free(&value);
                runtime_fail(interpreter, statement,
                             "For loop target must be a list or range");
                return exec_error();
            }
            env_init(&loop_env, interpreter->current_env);
            saved_env = interpreter->current_env;
            interpreter->current_env = &loop_env;
            if (value.kind == VALUE_ARRAY) {
                for (index = 0; index < value.as.array.count; ++index) {
                    if (!env_define(&loop_env, statement->as.for_statement.name,
                                    value.as.array.items[index], false)) {
                        runtime_fail(interpreter, statement, "Out of memory");
                        body_result = exec_error();
                        break;
                    }
                    body_result = execute_block(interpreter,
                                                statement->as.for_statement.body);
                    if (body_result.status != EXEC_NORMAL) break;
                }
            } else {
                int64_t item;
                for (item = value.as.range.start; item < value.as.range.end; ++item) {
                    if (!env_define(&loop_env, statement->as.for_statement.name,
                                    value_int(item), false)) {
                        runtime_fail(interpreter, statement, "Out of memory");
                        body_result = exec_error();
                        break;
                    }
                    body_result = execute_block(interpreter,
                                                statement->as.for_statement.body);
                    if (body_result.status != EXEC_NORMAL || item == INT64_MAX) break;
                }
            }
            interpreter->current_env = saved_env;
            env_free(&loop_env);
            value_free(&value);
            return body_result;
        }
        case AST_IMPORT: {
            ModuleValue *module = load_module(
                interpreter, statement->as.import_statement.module_name, statement);
            const char *binding_name = statement->as.import_statement.alias == NULL
                                           ? statement->as.import_statement.module_name
                                           : statement->as.import_statement.alias;
            if (module == NULL) return exec_error();
            if (!env_define(interpreter->current_env, binding_name,
                            value_module(module), false)) {
                runtime_fail(interpreter, statement, "Out of memory");
                return exec_error();
            }
            return exec_normal();
        }
        case AST_FROM_IMPORT: {
            ModuleValue *module = load_module(
                interpreter, statement->as.from_import.module_name, statement);
            size_t index;
            if (module == NULL) return exec_error();
            for (index = 0; index < statement->as.from_import.name_count; ++index) {
                const char *name = statement->as.from_import.names[index];
                const Symbol *symbol = env_find_local(&module->env, name);
                if (symbol == NULL) {
                    char message[256];
                    snprintf(message, sizeof(message),
                             "Symbol '%s' not found in module '%s'",
                             name, module->name);
                    runtime_fail(interpreter, statement, message);
                    return exec_error();
                }
                if (!symbol->is_public) {
                    char message[256];
                    snprintf(message, sizeof(message),
                             "Cannot import private symbol '%s' from module '%s'",
                             name, module->name);
                    runtime_fail(interpreter, statement, message);
                    return exec_error();
                }
                if (!env_define(interpreter->current_env, name,
                                symbol->value, false)) {
                    runtime_fail(interpreter, statement, "Out of memory");
                    return exec_error();
                }
            }
            return exec_normal();
        }
        default:
            runtime_fail(interpreter, statement, "Invalid statement");
            return exec_error();
    }
}

static void free_module(ModuleValue *module) {
    if (module == NULL) return;
    env_free(&module->env);
    ast_program_free(&module->program);
    token_array_free(&module->tokens);
    free(module->source);
    free(module->name);
    free(module->path);
    free(module);
}

bool interpreter_run(const AstProgram *program, const char *entry_path,
                     FILE *output, RuntimeError *error) {
    Interpreter interpreter = {0};
    ModuleValue entry_module = {0};
    ExecResult execution;
    size_t index;

    *error = (RuntimeError){0};
    interpreter.output = output;
    interpreter.error = error;
    interpreter.root_directory = directory_name(entry_path);
    if (interpreter.root_directory == NULL) {
        snprintf(error->message, sizeof(error->message), "Out of memory");
        error->has_error = true;
        return false;
    }

    entry_module.name = "__main__";
    entry_module.path = (char *)entry_path;
    entry_module.state = MODULE_LOADED;
    env_init(&entry_module.env, NULL);
    interpreter.current_module = &entry_module;
    interpreter.current_env = &entry_module.env;
    execution = execute_program(&interpreter, program);

    env_free(&entry_module.env);
    for (index = 0; index < interpreter.function_count; ++index)
        free(interpreter.functions[index]);
    free(interpreter.functions);
    for (index = 0; index < interpreter.module_count; ++index)
        free_module(interpreter.modules[index]);
    free(interpreter.modules);
    free(interpreter.root_directory);
    value_free(&execution.value);
    return execution.status == EXEC_NORMAL && !error->has_error;
}
