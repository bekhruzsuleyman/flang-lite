#include "semantic.h"

#include "lexer.h"
#include "parser.h"
#include "runtime/native.h"
#include "runtime/stdlib.h"
#include "source.h"
#include "token.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define SEMANTIC_MAX_TENSOR_RANK 16

typedef enum {
    SEM_UNKNOWN,
    SEM_INT,
    SEM_BOOL,
    SEM_STR,
    SEM_LIST,
    SEM_TENSOR,
    SEM_RANGE,
    SEM_FUNCTION,
    SEM_MODULE,
    SEM_VOID,
    SEM_ERROR
} SemanticKind;

typedef struct SemanticModule SemanticModule;

typedef struct {
    SemanticKind kind;
    const AstNode *function;
    SemanticModule *module;
    SemanticKind element_kind;
    bool shape_known;
    bool is_native;
    NativeId native_id;
    size_t rank;
    size_t shape[SEMANTIC_MAX_TENSOR_RANK];
} SemanticType;

typedef struct {
    char *name;
    SemanticType type;
    bool is_public;
} SemanticSymbol;

typedef struct SemanticScope {
    SemanticSymbol *symbols;
    size_t count;
    size_t capacity;
    struct SemanticScope *parent;
} SemanticScope;

typedef enum {
    SEM_MODULE_LOADING,
    SEM_MODULE_LOADED
} SemanticModuleState;

struct SemanticModule {
    char *name;
    char *path;
    char *source;
    TokenArray tokens;
    AstProgram program;
    SemanticScope globals;
    SemanticModuleState state;
};

typedef struct {
    bool type_mode;
    Diagnostic *diagnostic;
    char *root_directory;
    SemanticModule **modules;
    size_t module_count;
    size_t module_capacity;
    SemanticModule *current_module;
    SemanticScope *current_scope;
    const AstNode *current_function;
} SemanticChecker;

static SemanticType check_expression(SemanticChecker *checker,
                                     const AstNode *node);
static bool check_statement(SemanticChecker *checker, const AstNode *node);
static bool check_program(SemanticChecker *checker, const AstProgram *program);

static SemanticType semantic_type(SemanticKind kind) {
    SemanticType type;
    memset(&type, 0, sizeof(type));
    type.kind = kind;
    return type;
}

static const char *semantic_type_name(SemanticType type) {
    switch (type.kind) {
        case SEM_UNKNOWN: return "unknown";
        case SEM_INT: return "int";
        case SEM_BOOL: return "bool";
        case SEM_STR: return "str";
        case SEM_LIST: return "list";
        case SEM_TENSOR: return "tensor";
        case SEM_RANGE: return "range";
        case SEM_FUNCTION: return "function";
        case SEM_MODULE: return "module";
        case SEM_VOID: return "void";
        case SEM_ERROR: return "error";
    }
    return "unknown";
}

static void format_semantic_type(SemanticType type, char *buffer, size_t size) {
    if (type.kind == SEM_LIST && type.element_kind != SEM_UNKNOWN) {
        SemanticType element = semantic_type(type.element_kind);
        snprintf(buffer, size, "array[%s]", semantic_type_name(element));
    } else {
        snprintf(buffer, size, "%s", semantic_type_name(type));
    }
}

static SemanticType source_type(TypeKind type) {
    switch (type) {
        case TYPE_INT: return semantic_type(SEM_INT);
        case TYPE_BOOL: return semantic_type(SEM_BOOL);
        case TYPE_VOID: return semantic_type(SEM_VOID);
        case TYPE_STRING: return semantic_type(SEM_STR);
        case TYPE_LIST: return semantic_type(SEM_LIST);
        case TYPE_TENSOR: return semantic_type(SEM_TENSOR);
    }
    return semantic_type(SEM_UNKNOWN);
}

static SemanticType annotation_type(const char *annotation) {
    if (strcmp(annotation, "int") == 0) return semantic_type(SEM_INT);
    if (strcmp(annotation, "bool") == 0) return semantic_type(SEM_BOOL);
    if (strcmp(annotation, "str") == 0) return semantic_type(SEM_STR);
    if (strcmp(annotation, "list") == 0) return semantic_type(SEM_LIST);
    if (strcmp(annotation, "tensor") == 0) return semantic_type(SEM_TENSOR);
    return semantic_type(SEM_ERROR);
}

static bool types_compatible(SemanticType expected, SemanticType actual) {
    if (expected.kind == SEM_UNKNOWN || actual.kind == SEM_UNKNOWN) return true;
    if (expected.kind != actual.kind) return false;
    if (expected.kind == SEM_LIST && expected.element_kind != SEM_UNKNOWN &&
        actual.element_kind != SEM_UNKNOWN)
        return expected.element_kind == actual.element_kind;
    return true;
}

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
    if (backslash != NULL && (separator == NULL || backslash > separator))
        separator = backslash;
    if (separator == NULL) return copy_string(".");
    length = (size_t)(separator - path);
    if (length == 0) length = 1;
    directory = malloc(length + 1);
    if (directory == NULL) return NULL;
    memcpy(directory, path, length);
    directory[length] = '\0';
    return directory;
}

static char *make_module_path(const SemanticChecker *checker, const char *name) {
    size_t directory_length = strlen(checker->root_directory);
    size_t name_length = strlen(name);
    char *path = malloc(directory_length + name_length + 5);
    if (path == NULL) return NULL;
    snprintf(path, directory_length + name_length + 5, "%s/%s.fl",
             checker->root_directory, name);
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

static bool semantic_fail(SemanticChecker *checker, DiagnosticKind kind,
                          const AstNode *node, const char *format, ...) {
    va_list arguments;
    if (checker->diagnostic->has_error) return false;
    checker->diagnostic->has_error = true;
    checker->diagnostic->kind = kind;
    checker->diagnostic->line = node == NULL ? 1 : node->line;
    checker->diagnostic->column = node == NULL ? 1 : node->column;
    snprintf(checker->diagnostic->path, sizeof(checker->diagnostic->path), "%s",
             checker->current_module->path);
    va_start(arguments, format);
    vsnprintf(checker->diagnostic->message,
              sizeof(checker->diagnostic->message), format, arguments);
    va_end(arguments);
    {
        size_t length = 1;
        const char *hint = NULL;
        if (node != NULL && node->kind == AST_IDENTIFIER)
            length = strlen(node->as.identifier);
        else if (node != NULL && node->kind == AST_MEMBER)
            length = strlen(node->as.member.member) + 1;
        else if (node != NULL && node->kind == AST_STRING)
            length = strlen(node->as.string) + 2;
        if (strstr(checker->diagnostic->message, "Unknown variable") != NULL)
            hint = "declare it first with 'name = ...'";
        else if (strstr(checker->diagnostic->message, "Cannot assign") != NULL)
            hint = "keep the new value compatible with the variable's inferred type";
        else if (strstr(checker->diagnostic->message, "Cannot initialize") != NULL)
            hint = "change the annotation or use a value of the declared type";
        else if (strstr(checker->diagnostic->message, "no property") != NULL)
            hint = "check the properties supported by this value type";
        else if (strstr(checker->diagnostic->message, "private symbol") != NULL)
            hint = "mark it with 'pub' in its module if it should be exported";
        else if (strstr(checker->diagnostic->message, "Duplicate declaration") != NULL)
            hint = "rename the declaration or remove the duplicate";
        diagnostic_attach_source(checker->diagnostic,
                                 checker->current_module->source,
                                 length, hint);
    }
    return false;
}

static bool imported_syntax_fail(SemanticChecker *checker, const char *path,
                                 const char *source,
                                 size_t line, size_t column,
                                 DiagnosticKind kind, const char *message) {
    checker->diagnostic->has_error = true;
    checker->diagnostic->kind = kind;
    checker->diagnostic->line = line;
    checker->diagnostic->column = column;
    snprintf(checker->diagnostic->path, sizeof(checker->diagnostic->path), "%s", path);
    snprintf(checker->diagnostic->message, sizeof(checker->diagnostic->message),
             "%s", message);
    diagnostic_attach_source(checker->diagnostic, source,
                             1, NULL);
    return false;
}

static void scope_init(SemanticScope *scope, SemanticScope *parent) {
    *scope = (SemanticScope){0};
    scope->parent = parent;
}

static void scope_free(SemanticScope *scope) {
    size_t index;
    for (index = 0; index < scope->count; ++index) free(scope->symbols[index].name);
    free(scope->symbols);
    *scope = (SemanticScope){0};
}

static SemanticSymbol *scope_find_local(SemanticScope *scope, const char *name) {
    size_t index;
    for (index = 0; index < scope->count; ++index) {
        if (strcmp(scope->symbols[index].name, name) == 0) return &scope->symbols[index];
    }
    return NULL;
}

static SemanticSymbol *scope_find(SemanticScope *scope, const char *name) {
    SemanticScope *current;
    for (current = scope; current != NULL; current = current->parent) {
        SemanticSymbol *symbol = scope_find_local(current, name);
        if (symbol != NULL) return symbol;
    }
    return NULL;
}

static bool scope_define(SemanticChecker *checker, SemanticScope *scope,
                         const AstNode *node, const char *name,
                         SemanticType type, bool is_public) {
    SemanticSymbol *new_symbols;
    size_t new_capacity;
    char *copy;
    if (scope_find_local(scope, name) != NULL)
        return semantic_fail(checker, DIAGNOSTIC_RESOLVE, node,
                             "Duplicate declaration of '%s'", name);
    copy = copy_string(name);
    if (copy == NULL)
        return semantic_fail(checker, DIAGNOSTIC_RESOLVE, node, "Out of memory");
    if (scope->count == scope->capacity) {
        new_capacity = scope->capacity == 0 ? 8 : scope->capacity * 2;
        new_symbols = realloc(scope->symbols,
                              new_capacity * sizeof(*new_symbols));
        if (new_symbols == NULL) {
            free(copy);
            return semantic_fail(checker, DIAGNOSTIC_RESOLVE, node, "Out of memory");
        }
        scope->symbols = new_symbols;
        scope->capacity = new_capacity;
    }
    scope->symbols[scope->count++] = (SemanticSymbol){copy, type, is_public};
    return true;
}

static SemanticModule *find_module(const SemanticChecker *checker,
                                   const char *name) {
    size_t index;
    for (index = 0; index < checker->module_count; ++index) {
        if (strcmp(checker->modules[index]->name, name) == 0)
            return checker->modules[index];
    }
    return NULL;
}

static bool append_module(SemanticChecker *checker, SemanticModule *module) {
    SemanticModule **new_modules;
    size_t new_capacity;
    if (checker->module_count == checker->module_capacity) {
        new_capacity = checker->module_capacity == 0 ? 4 : checker->module_capacity * 2;
        new_modules = realloc(checker->modules,
                              new_capacity * sizeof(*new_modules));
        if (new_modules == NULL) return false;
        checker->modules = new_modules;
        checker->module_capacity = new_capacity;
    }
    checker->modules[checker->module_count++] = module;
    return true;
}

static bool predeclare_functions(SemanticChecker *checker,
                                 const AstProgram *program) {
    size_t index;
    for (index = 0; index < program->count; ++index) {
        const AstNode *node = program->statements[index];
        if (node->kind == AST_FUNCTION_DECLARATION) {
            SemanticType type = semantic_type(SEM_FUNCTION);
            type.function = node;
            if (!scope_define(checker, checker->current_scope, node,
                              node->as.function_declaration.name, type,
                              node->as.function_declaration.is_public))
                return false;
        }
    }
    return true;
}

static void free_module(SemanticModule *module) {
    if (module == NULL) return;
    scope_free(&module->globals);
    ast_program_free(&module->program);
    token_array_free(&module->tokens);
    free(module->source);
    free(module->name);
    free(module->path);
    free(module);
}

static SemanticModule *load_module(SemanticChecker *checker, const char *name,
                                   const AstNode *node) {
    SemanticModule *module = find_module(checker, name);
    LexerError lexer_error;
    ParserError parser_error;
    SemanticModule *saved_module;
    SemanticScope *saved_scope;

    if (module != NULL) {
        if (module->state == SEM_MODULE_LOADING) {
            semantic_fail(checker, DIAGNOSTIC_RESOLVE, node,
                          "Circular import detected involving module '%s'", name);
            return NULL;
        }
        return module;
    }
    module = calloc(1, sizeof(*module));
    if (module == NULL) {
        semantic_fail(checker, DIAGNOSTIC_RESOLVE, node, "Out of memory");
        return NULL;
    }
    module->name = copy_string(name);
    module->path = make_module_path(checker, name);
    module->state = SEM_MODULE_LOADING;
    scope_init(&module->globals, NULL);
    if (module->name == NULL || module->path == NULL ||
        !append_module(checker, module)) {
        free_module(module);
        semantic_fail(checker, DIAGNOSTIC_RESOLVE, node, "Out of memory");
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
                semantic_fail(checker, DIAGNOSTIC_RESOLVE, node,
                              "Out of memory");
                return NULL;
            }
            for (native_index = 0; native_index < native_symbol_count();
                 ++native_index) {
                const NativeSymbol *native = native_symbol_at(native_index);
                if (strcmp(native->module, name) == 0) {
                    SemanticType type = semantic_type(SEM_FUNCTION);
                    type.is_native = true;
                    type.native_id = native->id;
                    if (!scope_define(checker, &module->globals, node,
                                      native->name, type, true))
                        return NULL;
                }
            }
            module->state = SEM_MODULE_LOADED;
            return module;
        }
        semantic_fail(checker, DIAGNOSTIC_RESOLVE, node,
                      "Module '%s' not found at '%s'", name, module->path);
        return NULL;
    }
    if (!lexer_scan(module->source, &module->tokens, &lexer_error)) {
        imported_syntax_fail(checker, module->path, module->source, lexer_error.line,
                             lexer_error.column, DIAGNOSTIC_LEXER,
                             lexer_error.message);
        return NULL;
    }
    if (!parser_parse(&module->tokens, &module->program, &parser_error)) {
        imported_syntax_fail(checker, module->path, module->source, parser_error.line,
                             parser_error.column, DIAGNOSTIC_PARSER,
                             parser_error.message);
        return NULL;
    }
    saved_module = checker->current_module;
    saved_scope = checker->current_scope;
    checker->current_module = module;
    checker->current_scope = &module->globals;
    if (!predeclare_functions(checker, &module->program) ||
        !check_program(checker, &module->program)) {
        checker->current_module = saved_module;
        checker->current_scope = saved_scope;
        return NULL;
    }
    checker->current_module = saved_module;
    checker->current_scope = saved_scope;
    module->state = SEM_MODULE_LOADED;
    return module;
}

typedef enum {
    LITERAL_SHAPE_KNOWN,
    LITERAL_SHAPE_UNKNOWN,
    LITERAL_SHAPE_RAGGED,
    LITERAL_SHAPE_NON_INT
} LiteralShapeStatus;

static LiteralShapeStatus tensor_literal_shape(const AstNode *node,
                                               size_t *shape, size_t *rank) {
    size_t child_shape[SEMANTIC_MAX_TENSOR_RANK];
    size_t child_rank = 0;
    size_t index;
    if (node->kind == AST_NUMBER) {
        *rank = 0;
        return LITERAL_SHAPE_KNOWN;
    }
    if (node->kind == AST_BOOL || node->kind == AST_STRING)
        return LITERAL_SHAPE_NON_INT;
    if (node->kind != AST_ARRAY || *rank >= SEMANTIC_MAX_TENSOR_RANK)
        return LITERAL_SHAPE_UNKNOWN;
    if (node->as.array.count > 0) {
        LiteralShapeStatus first = tensor_literal_shape(
            node->as.array.elements[0], child_shape, &child_rank);
        if (first != LITERAL_SHAPE_KNOWN) return first;
        for (index = 1; index < node->as.array.count; ++index) {
            size_t other_shape[SEMANTIC_MAX_TENSOR_RANK];
            size_t other_rank = 0;
            size_t dimension;
            LiteralShapeStatus other = tensor_literal_shape(
                node->as.array.elements[index], other_shape, &other_rank);
            if (other != LITERAL_SHAPE_KNOWN) return other;
            if (other_rank != child_rank) return LITERAL_SHAPE_RAGGED;
            for (dimension = 0; dimension < child_rank; ++dimension) {
                if (other_shape[dimension] != child_shape[dimension])
                    return LITERAL_SHAPE_RAGGED;
            }
        }
    }
    if (child_rank + 1 > SEMANTIC_MAX_TENSOR_RANK)
        return LITERAL_SHAPE_UNKNOWN;
    shape[0] = node->as.array.count;
    if (child_rank > 0) memcpy(shape + 1, child_shape,
                               child_rank * sizeof(*shape));
    *rank = child_rank + 1;
    return LITERAL_SHAPE_KNOWN;
}

static SemanticType builtin_call_type(SemanticChecker *checker,
                                      const AstNode *call, const char *name) {
    size_t count = call->as.call.argument_count;
    size_t expected = strcmp(name, "range") == 0 ? 2 : 1;
    SemanticType arguments[2];
    size_t index;
    if ((strcmp(name, "tensor") == 0 && count != 1 && count != 2) ||
        (strcmp(name, "tensor") != 0 && count != expected)) {
        if (checker->type_mode) {
            if (strcmp(name, "tensor") == 0)
                semantic_fail(checker, DIAGNOSTIC_TYPE, call,
                              "tensor() expects 1 or 2 arguments, got %zu",
                              count);
            else
                semantic_fail(checker, DIAGNOSTIC_TYPE, call,
                              "Builtin '%s' expects %zu arguments, got %zu",
                              name, expected, count);
        }
        return semantic_type(SEM_ERROR);
    }
    for (index = 0; index < count; ++index)
        arguments[index] = check_expression(checker, call->as.call.arguments[index]);
    if (checker->diagnostic->has_error) return semantic_type(SEM_ERROR);
    if (strcmp(name, "len") == 0) {
        if (checker->type_mode && arguments[0].kind != SEM_UNKNOWN &&
            arguments[0].kind != SEM_STR && arguments[0].kind != SEM_LIST)
            semantic_fail(checker, DIAGNOSTIC_TYPE, call,
                          "len() cannot be used with %s",
                          semantic_type_name(arguments[0]));
        return semantic_type(SEM_INT);
    }
    if (strcmp(name, "range") == 0) {
        if (checker->type_mode &&
            ((arguments[0].kind != SEM_INT && arguments[0].kind != SEM_UNKNOWN) ||
             (arguments[1].kind != SEM_INT && arguments[1].kind != SEM_UNKNOWN)))
            semantic_fail(checker, DIAGNOSTIC_TYPE, call,
                          "range() expects int and int");
        return semantic_type(SEM_RANGE);
    }
    if (strcmp(name, "tensor") == 0) {
        SemanticType result = semantic_type(SEM_TENSOR);
        LiteralShapeStatus shape_status;
        if (checker->type_mode && arguments[0].kind != SEM_LIST &&
            arguments[0].kind != SEM_UNKNOWN)
            semantic_fail(checker, DIAGNOSTIC_TYPE, call,
                          "tensor() expects list, got %s",
                          semantic_type_name(arguments[0]));
        if (count == 2 && checker->type_mode &&
            arguments[1].kind != SEM_BOOL &&
            arguments[1].kind != SEM_UNKNOWN)
            semantic_fail(checker, DIAGNOSTIC_TYPE, call,
                          "tensor() requires requires_grad to be bool");
        shape_status = tensor_literal_shape(call->as.call.arguments[0],
                                            result.shape, &result.rank);
        if (shape_status == LITERAL_SHAPE_RAGGED && checker->type_mode) {
            semantic_fail(checker, DIAGNOSTIC_TYPE, call,
                          "Ragged tensor literal");
            return semantic_type(SEM_ERROR);
        }
        if (shape_status == LITERAL_SHAPE_NON_INT && checker->type_mode) {
            semantic_fail(checker, DIAGNOSTIC_TYPE, call,
                          "Tensor elements must be int");
            return semantic_type(SEM_ERROR);
        }
        if (shape_status == LITERAL_SHAPE_KNOWN)
            result.shape_known = true;
        return result;
    }
    if (strcmp(name, "shape") == 0 || strcmp(name, "rank") == 0) {
        if (checker->type_mode && arguments[0].kind != SEM_TENSOR &&
            arguments[0].kind != SEM_UNKNOWN)
            semantic_fail(checker, DIAGNOSTIC_TYPE, call,
                          "%s() expects tensor, got %s", name,
                          semantic_type_name(arguments[0]));
        return semantic_type(strcmp(name, "shape") == 0 ? SEM_LIST : SEM_INT);
    }
    if (checker->type_mode && arguments[0].kind != SEM_LIST &&
        arguments[0].kind != SEM_UNKNOWN)
        semantic_fail(checker, DIAGNOSTIC_TYPE, call,
                      "%s() expects a shape list", name);
    {
        SemanticType result = semantic_type(SEM_TENSOR);
        const AstNode *shape_node = call->as.call.arguments[0];
        if (shape_node->kind == AST_ARRAY &&
            shape_node->as.array.count <= SEMANTIC_MAX_TENSOR_RANK) {
            result.rank = shape_node->as.array.count;
            result.shape_known = true;
            for (index = 0; index < result.rank; ++index) {
                const AstNode *dimension = shape_node->as.array.elements[index];
                if (dimension->kind != AST_NUMBER || dimension->as.number < 0) {
                    result.shape_known = false;
                    break;
                }
                result.shape[index] = (size_t)dimension->as.number;
            }
        }
        return result;
    }
}

static bool is_builtin(const char *name) {
    return strcmp(name, "len") == 0 || strcmp(name, "range") == 0 ||
           strcmp(name, "tensor") == 0 || strcmp(name, "shape") == 0 ||
           strcmp(name, "rank") == 0 || strcmp(name, "zeros") == 0 ||
           strcmp(name, "ones") == 0;
}

static SemanticType native_call_type(SemanticChecker *checker,
                                     const AstNode *call, NativeId id) {
    if (id == NATIVE_TENSOR_CREATE)
        return builtin_call_type(checker, call, "tensor");
    if (id == NATIVE_TENSOR_ZEROS)
        return builtin_call_type(checker, call, "zeros");
    if (id == NATIVE_TENSOR_ONES)
        return builtin_call_type(checker, call, "ones");
    if (id == NATIVE_TENSOR_ARANGE) {
        SemanticType start;
        SemanticType end;
        SemanticType result = semantic_type(SEM_TENSOR);
        if (call->as.call.argument_count != 2) {
            if (checker->type_mode)
                semantic_fail(checker, DIAGNOSTIC_TYPE, call,
                              "arange() expects 2 arguments, got %zu",
                              call->as.call.argument_count);
            return semantic_type(SEM_ERROR);
        }
        start = check_expression(checker, call->as.call.arguments[0]);
        end = check_expression(checker, call->as.call.arguments[1]);
        if (checker->type_mode &&
            ((start.kind != SEM_INT && start.kind != SEM_UNKNOWN) ||
             (end.kind != SEM_INT && end.kind != SEM_UNKNOWN)))
            semantic_fail(checker, DIAGNOSTIC_TYPE, call,
                          "arange() expects int and int");
        if (call->as.call.arguments[0]->kind == AST_NUMBER &&
            call->as.call.arguments[1]->kind == AST_NUMBER &&
            call->as.call.arguments[1]->as.number >=
                call->as.call.arguments[0]->as.number) {
            result.shape_known = true;
            result.rank = 1;
            result.shape[0] = (size_t)(call->as.call.arguments[1]->as.number -
                                       call->as.call.arguments[0]->as.number);
        }
        return result;
    }
    return semantic_type(SEM_ERROR);
}

static SemanticType tensor_method_type(SemanticChecker *checker,
                                       const AstNode *call,
                                       SemanticType object,
                                       const char *name) {
    size_t index;
    for (index = 0; index < call->as.call.argument_count; ++index)
        check_expression(checker, call->as.call.arguments[index]);
    if (call->as.call.argument_count != 0) {
        if (checker->type_mode)
            semantic_fail(checker, DIAGNOSTIC_TYPE, call,
                          "Tensor method '%s' expects 0 arguments, got %zu",
                          name, call->as.call.argument_count);
        return semantic_type(SEM_ERROR);
    }
    if (strcmp(name, "sum") == 0 || strcmp(name, "mean") == 0) {
        SemanticType result = semantic_type(SEM_TENSOR);
        ((AstNode *)call)->as.call.native_method = strcmp(name, "mean") == 0
                                                       ? AST_NATIVE_TENSOR_MEAN
                                                       : AST_NATIVE_TENSOR_SUM;
        result.shape_known = true;
        result.rank = 0;
        return result;
    }
    if (strcmp(name, "backward") == 0 || strcmp(name, "zero_grad") == 0) {
        ((AstNode *)call)->as.call.native_method =
            strcmp(name, "backward") == 0 ? AST_NATIVE_TENSOR_BACKWARD
                                            : AST_NATIVE_TENSOR_ZERO_GRAD;
        return semantic_type(SEM_VOID);
    }
    if (checker->type_mode)
        semantic_fail(checker, DIAGNOSTIC_TYPE, call,
                      "Tensor has no method '%s'", name);
    (void)object;
    return semantic_type(SEM_ERROR);
}

static void format_shape(SemanticType type, char *buffer, size_t size) {
    size_t used = 0;
    size_t index;
    if (size == 0) return;
    buffer[used++] = '[';
    for (index = 0; index < type.rank && used < size; ++index) {
        int written = snprintf(buffer + used, size - used, "%s%zu",
                               index == 0 ? "" : ", ", type.shape[index]);
        if (written < 0 || (size_t)written >= size - used) {
            used = size - 1;
            break;
        }
        used += (size_t)written;
    }
    if (used + 1 < size) buffer[used++] = ']';
    buffer[used < size ? used : size - 1] = '\0';
}

static SemanticType check_binary(SemanticChecker *checker, const AstNode *node) {
    SemanticType left = check_expression(checker, node->as.binary.left);
    SemanticType right = check_expression(checker, node->as.binary.right);
    TokenType operator_type = node->as.binary.operator_type;
    if (checker->diagnostic->has_error) return semantic_type(SEM_ERROR);
    if (!checker->type_mode || left.kind == SEM_UNKNOWN || right.kind == SEM_UNKNOWN)
        return semantic_type(SEM_UNKNOWN);
    if (operator_type == TOKEN_AND || operator_type == TOKEN_OR) {
        if (left.kind != SEM_BOOL || right.kind != SEM_BOOL) {
            semantic_fail(checker, DIAGNOSTIC_TYPE, node,
                          "Cannot use operator '%s' with %s and %s",
                          token_type_name(operator_type), semantic_type_name(left),
                          semantic_type_name(right));
            return semantic_type(SEM_ERROR);
        }
        return semantic_type(SEM_BOOL);
    }
    if (operator_type == TOKEN_EQUAL_EQUAL || operator_type == TOKEN_BANG_EQUAL) {
        if (left.kind != right.kind ||
            (left.kind != SEM_INT && left.kind != SEM_BOOL && left.kind != SEM_STR)) {
            semantic_fail(checker, DIAGNOSTIC_TYPE, node,
                          "Cannot use operator '%s' with %s and %s",
                          token_type_name(operator_type), semantic_type_name(left),
                          semantic_type_name(right));
            return semantic_type(SEM_ERROR);
        }
        return semantic_type(SEM_BOOL);
    }
    if (operator_type == TOKEN_LESS || operator_type == TOKEN_LESS_EQUAL ||
        operator_type == TOKEN_GREATER || operator_type == TOKEN_GREATER_EQUAL) {
        if (left.kind != SEM_INT || right.kind != SEM_INT) {
            semantic_fail(checker, DIAGNOSTIC_TYPE, node,
                          "Cannot use operator '%s' with %s and %s",
                          token_type_name(operator_type), semantic_type_name(left),
                          semantic_type_name(right));
            return semantic_type(SEM_ERROR);
        }
        return semantic_type(SEM_BOOL);
    }
    if (left.kind == SEM_STR && right.kind == SEM_STR && operator_type == TOKEN_PLUS)
        return semantic_type(SEM_STR);
    if (left.kind == SEM_INT && right.kind == SEM_INT) return semantic_type(SEM_INT);
    if ((left.kind == SEM_TENSOR && right.kind == SEM_INT) ||
        (left.kind == SEM_INT && right.kind == SEM_TENSOR))
        return left.kind == SEM_TENSOR ? left : right;
    if (left.kind == SEM_TENSOR && right.kind == SEM_TENSOR) {
        if (left.shape_known && right.shape_known) {
            size_t dimension;
            bool same = left.rank == right.rank;
            for (dimension = 0; same && dimension < left.rank; ++dimension)
                same = left.shape[dimension] == right.shape[dimension];
            if (!same) {
                char left_shape[80];
                char right_shape[80];
                format_shape(left, left_shape, sizeof(left_shape));
                format_shape(right, right_shape, sizeof(right_shape));
                semantic_fail(checker, DIAGNOSTIC_TYPE, node,
                              "Tensor shape mismatch for '%s': %s vs %s",
                              token_type_name(operator_type), left_shape, right_shape);
                return semantic_type(SEM_ERROR);
            }
        }
        return left;
    }
    semantic_fail(checker, DIAGNOSTIC_TYPE, node,
                  "Cannot use operator '%s' with %s and %s",
                  token_type_name(operator_type), semantic_type_name(left),
                  semantic_type_name(right));
    return semantic_type(SEM_ERROR);
}

static SemanticType check_expression(SemanticChecker *checker,
                                     const AstNode *node) {
    switch (node->kind) {
        case AST_NUMBER: return semantic_type(SEM_INT);
        case AST_BOOL: return semantic_type(SEM_BOOL);
        case AST_STRING: return semantic_type(SEM_STR);
        case AST_ARRAY: {
            size_t index;
            SemanticType result = semantic_type(SEM_LIST);
            result.element_kind = SEM_UNKNOWN;
            for (index = 0; index < node->as.array.count; ++index) {
                SemanticType element = check_expression(
                    checker, node->as.array.elements[index]);
                if (index == 0) {
                    result.element_kind = element.kind;
                } else if (checker->type_mode && element.kind != SEM_UNKNOWN &&
                           result.element_kind != SEM_UNKNOWN &&
                           element.kind != result.element_kind) {
                    semantic_fail(checker, DIAGNOSTIC_TYPE,
                                  node->as.array.elements[index],
                                  "Array elements must have the same type; found %s and %s",
                                  semantic_type_name(semantic_type(result.element_kind)),
                                  semantic_type_name(element));
                    return semantic_type(SEM_ERROR);
                }
            }
            return result;
        }
        case AST_IDENTIFIER: {
            SemanticSymbol *symbol = scope_find(checker->current_scope,
                                                node->as.identifier);
            if (symbol == NULL) {
                semantic_fail(checker, DIAGNOSTIC_RESOLVE, node,
                              "Unknown variable '%s'", node->as.identifier);
                return semantic_type(SEM_ERROR);
            }
            return symbol->type;
        }
        case AST_BINARY:
            return check_binary(checker, node);
        case AST_UNARY: {
            SemanticType operand = check_expression(checker, node->as.unary.operand);
            SemanticKind expected = node->as.unary.operator_type == TOKEN_NOT
                                        ? SEM_BOOL
                                        : SEM_INT;
            if (checker->type_mode && operand.kind != SEM_UNKNOWN &&
                operand.kind != expected)
                semantic_fail(checker, DIAGNOSTIC_TYPE, node,
                              "Operator '%s' cannot be used with %s",
                              token_type_name(node->as.unary.operator_type),
                              semantic_type_name(operand));
            return semantic_type(expected);
        }
        case AST_INDEX: {
            SemanticType object = check_expression(checker, node->as.index.object);
            SemanticType index = check_expression(checker, node->as.index.index);
            if (checker->type_mode && index.kind != SEM_INT && index.kind != SEM_UNKNOWN)
                semantic_fail(checker, DIAGNOSTIC_TYPE, node,
                              "Index must be int, got %s", semantic_type_name(index));
            if (object.kind == SEM_STR) return semantic_type(SEM_STR);
            if (object.kind == SEM_LIST)
                return semantic_type(object.element_kind);
            if (object.kind == SEM_UNKNOWN) return semantic_type(SEM_UNKNOWN);
            if (object.kind == SEM_TENSOR) {
                if (object.shape_known && object.rank > 1) {
                    size_t dimension;
                    SemanticType result = semantic_type(SEM_TENSOR);
                    result.shape_known = true;
                    result.rank = object.rank - 1;
                    for (dimension = 0; dimension < result.rank; ++dimension)
                        result.shape[dimension] = object.shape[dimension + 1];
                    return result;
                }
                return object.shape_known && object.rank == 1
                           ? semantic_type(SEM_INT)
                           : semantic_type(SEM_UNKNOWN);
            }
            if (checker->type_mode)
                semantic_fail(checker, DIAGNOSTIC_TYPE, node,
                              "Type %s is not indexable", semantic_type_name(object));
            return semantic_type(SEM_ERROR);
        }
        case AST_MEMBER: {
            SemanticType object = check_expression(checker, node->as.member.object);
            const char *name = node->as.member.member;
            if (checker->diagnostic->has_error) return semantic_type(SEM_ERROR);
            if (object.kind == SEM_MODULE) {
                SemanticSymbol *symbol = scope_find_local(&object.module->globals, name);
                if (symbol == NULL) {
                    semantic_fail(checker, DIAGNOSTIC_RESOLVE, node,
                                  "Symbol '%s' not found in module '%s'",
                                  name, object.module->name);
                    return semantic_type(SEM_ERROR);
                }
                if (!symbol->is_public) {
                    semantic_fail(checker, DIAGNOSTIC_RESOLVE, node,
                                  "Cannot access private symbol '%s' from module '%s'",
                                  name, object.module->name);
                    return semantic_type(SEM_ERROR);
                }
                return symbol->type;
            }
            if ((object.kind == SEM_LIST || object.kind == SEM_STR) &&
                strcmp(name, "len") == 0)
                return semantic_type(SEM_INT);
            if (object.kind == SEM_TENSOR) {
                if (strcmp(name, "shape") == 0) return semantic_type(SEM_LIST);
                if (strcmp(name, "rank") == 0 || strcmp(name, "len") == 0)
                    return semantic_type(SEM_INT);
                if (strcmp(name, "requires_grad") == 0)
                    return semantic_type(SEM_BOOL);
                if (strcmp(name, "grad") == 0)
                    return semantic_type(SEM_TENSOR);
            }
            if (object.kind == SEM_UNKNOWN) return object;
            if (checker->type_mode)
                semantic_fail(checker, DIAGNOSTIC_TYPE, node,
                              "Type %s has no property '%s'",
                              semantic_type_name(object), name);
            return semantic_type(SEM_ERROR);
        }
        case AST_CALL: {
            const AstNode *callee = node->as.call.callee;
            SemanticType callee_type;
            size_t index;
            if (callee->kind == AST_MEMBER) {
                SemanticType object = check_expression(
                    checker, callee->as.member.object);
                if (checker->diagnostic->has_error)
                    return semantic_type(SEM_ERROR);
                if (object.kind == SEM_TENSOR)
                    return tensor_method_type(checker, node, object,
                                              callee->as.member.member);
            }
            if (callee->kind == AST_IDENTIFIER &&
                scope_find(checker->current_scope, callee->as.identifier) == NULL) {
                if (is_builtin(callee->as.identifier))
                    return builtin_call_type(checker, node, callee->as.identifier);
                semantic_fail(checker, DIAGNOSTIC_RESOLVE, callee,
                              "Unknown function '%s'", callee->as.identifier);
                return semantic_type(SEM_ERROR);
            }
            callee_type = check_expression(checker, callee);
            if (checker->diagnostic->has_error) return semantic_type(SEM_ERROR);
            if (callee_type.kind != SEM_FUNCTION) {
                for (index = 0; index < node->as.call.argument_count; ++index)
                    check_expression(checker, node->as.call.arguments[index]);
                if (checker->type_mode && callee_type.kind != SEM_UNKNOWN)
                    semantic_fail(checker, DIAGNOSTIC_TYPE, node,
                                  "Type %s is not callable",
                                  semantic_type_name(callee_type));
                return semantic_type(SEM_UNKNOWN);
            }
            if (callee_type.is_native)
                return native_call_type(checker, node, callee_type.native_id);
            {
                const AstNode *function = callee_type.function;
                size_t expected = function->as.function_declaration.parameter_count;
                if (checker->type_mode && node->as.call.argument_count != expected) {
                    semantic_fail(checker, DIAGNOSTIC_TYPE, node,
                                  "Function '%s' expects %zu arguments, got %zu",
                                  function->as.function_declaration.name, expected,
                                  node->as.call.argument_count);
                    return semantic_type(SEM_ERROR);
                }
                for (index = 0; index < node->as.call.argument_count; ++index) {
                    SemanticType argument = check_expression(
                        checker, node->as.call.arguments[index]);
                    if (checker->type_mode && index < expected) {
                        SemanticType parameter = source_type(
                            function->as.function_declaration.parameters[index].type);
                        if (!types_compatible(parameter, argument)) {
                            semantic_fail(checker, DIAGNOSTIC_TYPE,
                                          node->as.call.arguments[index],
                                          "Function '%s' expects argument %zu to be %s, got %s",
                                          function->as.function_declaration.name,
                                          index + 1, semantic_type_name(parameter),
                                          semantic_type_name(argument));
                            return semantic_type(SEM_ERROR);
                        }
                    }
                }
                return source_type(function->as.function_declaration.return_type);
            }
        }
        default:
            return semantic_type(SEM_UNKNOWN);
    }
}

static bool contains_return(const AstNode *node) {
    size_t index;
    if (node->kind == AST_RETURN) return true;
    if (node->kind == AST_BLOCK) {
        for (index = 0; index < node->as.block.count; ++index) {
            if (contains_return(node->as.block.statements[index])) return true;
        }
    } else if (node->kind == AST_IF) {
        return contains_return(node->as.if_statement.then_block) ||
               (node->as.if_statement.else_block != NULL &&
                contains_return(node->as.if_statement.else_block));
    } else if (node->kind == AST_WHILE) {
        return contains_return(node->as.while_statement.body);
    } else if (node->kind == AST_FOR) {
        return contains_return(node->as.for_statement.body);
    }
    return false;
}

static bool check_block(SemanticChecker *checker, const AstNode *block) {
    size_t index;
    for (index = 0; index < block->as.block.count; ++index) {
        if (!check_statement(checker, block->as.block.statements[index])) return false;
    }
    return true;
}

static bool check_function(SemanticChecker *checker, const AstNode *node) {
    SemanticScope function_scope;
    SemanticScope *saved_scope = checker->current_scope;
    const AstNode *saved_function = checker->current_function;
    size_t index;
    scope_init(&function_scope, checker->current_scope);
    checker->current_scope = &function_scope;
    checker->current_function = node;
    for (index = 0; index < node->as.function_declaration.parameter_count; ++index) {
        AstParameter parameter = node->as.function_declaration.parameters[index];
        if (!scope_define(checker, &function_scope, node, parameter.name,
                          source_type(parameter.type), false))
            break;
    }
    if (!checker->diagnostic->has_error) check_block(checker, node->as.function_declaration.body);
    if (checker->type_mode && !checker->diagnostic->has_error &&
        node->as.function_declaration.return_type != TYPE_VOID &&
        !contains_return(node->as.function_declaration.body))
        semantic_fail(checker, DIAGNOSTIC_TYPE, node,
                      "Function '%s' returning %s has no return statement",
                      node->as.function_declaration.name,
                      semantic_type_name(source_type(
                          node->as.function_declaration.return_type)));
    checker->current_scope = saved_scope;
    checker->current_function = saved_function;
    scope_free(&function_scope);
    return !checker->diagnostic->has_error;
}

static bool check_import(SemanticChecker *checker, const AstNode *node) {
    SemanticModule *module = load_module(checker,
                                         node->as.import_statement.module_name,
                                         node);
    const char *binding;
    SemanticType type;
    SemanticSymbol *existing;
    if (module == NULL) return false;
    binding = node->as.import_statement.alias == NULL
                  ? node->as.import_statement.module_name
                  : node->as.import_statement.alias;
    existing = scope_find_local(checker->current_scope, binding);
    if (existing != NULL && existing->type.kind == SEM_MODULE &&
        existing->type.module == module)
        return true;
    type = semantic_type(SEM_MODULE);
    type.module = module;
    return scope_define(checker, checker->current_scope, node, binding, type, false);
}

static bool check_from_import(SemanticChecker *checker, const AstNode *node) {
    SemanticModule *module = load_module(checker, node->as.from_import.module_name,
                                         node);
    size_t index;
    if (module == NULL) return false;
    for (index = 0; index < node->as.from_import.name_count; ++index) {
        const char *name = node->as.from_import.names[index];
        SemanticSymbol *symbol = scope_find_local(&module->globals, name);
        if (symbol == NULL)
            return semantic_fail(checker, DIAGNOSTIC_RESOLVE, node,
                                 "Symbol '%s' not found in module '%s'",
                                 name, module->name);
        if (!symbol->is_public)
            return semantic_fail(checker, DIAGNOSTIC_RESOLVE, node,
                                 "Cannot import private symbol '%s' from module '%s'",
                                 name, module->name);
        if (!scope_define(checker, checker->current_scope, node, name,
                          symbol->type, false))
            return false;
    }
    return true;
}

static bool check_statement(SemanticChecker *checker, const AstNode *node) {
    switch (node->kind) {
        case AST_ASSIGNMENT: {
            SemanticType value = check_expression(checker, node->as.assignment.value);
            SemanticSymbol *symbol;
            if (checker->diagnostic->has_error) return false;
            if (node->as.assignment.is_declaration) {
                SemanticType declared = annotation_type(node->as.assignment.annotation);
                if (declared.kind == SEM_ERROR)
                    return semantic_fail(checker, DIAGNOSTIC_TYPE, node,
                                         "Unknown type '%s'",
                                         node->as.assignment.annotation);
                if (checker->type_mode && !types_compatible(declared, value))
                    return semantic_fail(checker, DIAGNOSTIC_TYPE,
                                         node->as.assignment.value,
                                         "Cannot initialize variable '%s' of type %s with value of type %s",
                                         node->as.assignment.name,
                                         semantic_type_name(declared),
                                         semantic_type_name(value));
                if (value.kind == SEM_TENSOR && declared.kind == SEM_TENSOR)
                    declared = value;
                return scope_define(checker, checker->current_scope, node,
                                    node->as.assignment.name, declared,
                                    node->as.assignment.is_public);
            }
            symbol = scope_find(checker->current_scope,
                                node->as.assignment.name);
            if (symbol == NULL) {
                ((AstNode *)node)->as.assignment.is_inferred_declaration = true;
                return scope_define(checker, checker->current_scope, node,
                                    node->as.assignment.name, value,
                                    node->as.assignment.is_public);
            }
            ((AstNode *)node)->as.assignment.is_inferred_declaration = false;
            if (checker->type_mode && !types_compatible(symbol->type, value))
            {
                char value_name[64];
                char variable_name[64];
                format_semantic_type(value, value_name, sizeof(value_name));
                format_semantic_type(symbol->type, variable_name,
                                     sizeof(variable_name));
                return semantic_fail(checker, DIAGNOSTIC_TYPE,
                                     node->as.assignment.value,
                                     "Cannot assign %s to variable '%s' of type %s",
                                     value_name, node->as.assignment.name,
                                     variable_name);
            }
            if (symbol->type.kind == SEM_TENSOR && value.kind == SEM_TENSOR)
                symbol->type = value;
            return true;
        }
        case AST_INDEX_ASSIGNMENT: {
            SemanticType target = check_expression(
                checker, node->as.index_assignment.target);
            SemanticType value = check_expression(
                checker, node->as.index_assignment.value);
            if (checker->type_mode && !checker->diagnostic->has_error &&
                target.kind == SEM_ERROR) return false;
            if (checker->type_mode && !checker->diagnostic->has_error &&
                !types_compatible(target, value))
                return semantic_fail(checker, DIAGNOSTIC_TYPE,
                                     node->as.index_assignment.value,
                                     "Cannot assign %s to indexed value of type %s",
                                     semantic_type_name(value),
                                     semantic_type_name(target));
            return !checker->diagnostic->has_error;
        }
        case AST_FUNCTION_DECLARATION:
            if (scope_find_local(checker->current_scope,
                                 node->as.function_declaration.name) == NULL) {
                SemanticType type = semantic_type(SEM_FUNCTION);
                type.function = node;
                if (!scope_define(checker, checker->current_scope, node,
                                  node->as.function_declaration.name, type,
                                  node->as.function_declaration.is_public))
                    return false;
            }
            return check_function(checker, node);
        case AST_RETURN: {
            SemanticType actual;
            if (checker->current_function == NULL)
                return semantic_fail(checker, DIAGNOSTIC_RESOLVE, node,
                                     "Cannot return outside of function");
            actual = node->as.return_statement.value == NULL
                         ? semantic_type(SEM_VOID)
                         : check_expression(checker,
                               node->as.return_statement.value);
            if (checker->type_mode) {
                SemanticType expected = source_type(
                    checker->current_function->as.function_declaration.return_type);
                if (!types_compatible(expected, actual))
                    return semantic_fail(checker, DIAGNOSTIC_TYPE, node,
                                         "Function '%s' returns %s but got %s",
                                         checker->current_function->as.function_declaration.name,
                                         semantic_type_name(expected),
                                         semantic_type_name(actual));
            }
            return !checker->diagnostic->has_error;
        }
        case AST_IMPORT: return check_import(checker, node);
        case AST_FROM_IMPORT: return check_from_import(checker, node);
        case AST_PRINT:
            check_expression(checker, node->as.print_statement.expression);
            return !checker->diagnostic->has_error;
        case AST_EXPRESSION_STATEMENT:
            check_expression(checker, node->as.expression_statement.expression);
            return !checker->diagnostic->has_error;
        case AST_IF:
        case AST_WHILE: {
            const AstNode *condition = node->kind == AST_IF
                                           ? node->as.if_statement.condition
                                           : node->as.while_statement.condition;
            SemanticType type = check_expression(checker, condition);
            if (checker->type_mode && type.kind != SEM_BOOL &&
                type.kind != SEM_UNKNOWN)
                return semantic_fail(checker, DIAGNOSTIC_TYPE, condition,
                                     "%s condition must be bool, got %s",
                                     node->kind == AST_IF ? "If" : "While",
                                     semantic_type_name(type));
            if (node->kind == AST_IF) {
                if (!check_block(checker, node->as.if_statement.then_block)) return false;
                return node->as.if_statement.else_block == NULL ||
                       check_block(checker, node->as.if_statement.else_block);
            }
            return check_block(checker, node->as.while_statement.body);
        }
        case AST_FOR: {
            SemanticType iterable = check_expression(checker,
                                                      node->as.for_statement.iterable);
            SemanticScope loop_scope;
            bool success;
            if (checker->type_mode && iterable.kind != SEM_LIST &&
                iterable.kind != SEM_RANGE && iterable.kind != SEM_UNKNOWN)
                return semantic_fail(checker, DIAGNOSTIC_TYPE, node,
                                     "For loop target must be list or range, got %s",
                                     semantic_type_name(iterable));
            scope_init(&loop_scope, checker->current_scope);
            checker->current_scope = &loop_scope;
            success = scope_define(checker, &loop_scope, node,
                                   node->as.for_statement.name,
                                   iterable.kind == SEM_RANGE
                                       ? semantic_type(SEM_INT)
                                       : semantic_type(iterable.element_kind), false) &&
                      check_block(checker, node->as.for_statement.body);
            checker->current_scope = loop_scope.parent;
            scope_free(&loop_scope);
            return success;
        }
        case AST_BLOCK:
            return check_block(checker, node);
        default:
            return true;
    }
}

static bool check_program(SemanticChecker *checker, const AstProgram *program) {
    size_t index;
    for (index = 0; index < program->count; ++index) {
        if (!check_statement(checker, program->statements[index])) return false;
    }
    return true;
}

bool semantic_run(const AstProgram *program, const char *entry_path,
                  Diagnostic *diagnostic, bool type_mode) {
    SemanticChecker checker = {0};
    SemanticModule root = {0};
    size_t index;
    bool success;
    *diagnostic = (Diagnostic){0};
    checker.type_mode = type_mode;
    checker.diagnostic = diagnostic;
    checker.root_directory = directory_name(entry_path);
    if (checker.root_directory == NULL) {
        diagnostic->has_error = true;
        diagnostic->kind = type_mode ? DIAGNOSTIC_TYPE : DIAGNOSTIC_RESOLVE;
        snprintf(diagnostic->path, sizeof(diagnostic->path), "%s", entry_path);
        snprintf(diagnostic->message, sizeof(diagnostic->message), "Out of memory");
        return false;
    }
    root.name = "__main__";
    root.path = (char *)entry_path;
    root.source = read_source(entry_path);
    root.state = SEM_MODULE_LOADED;
    scope_init(&root.globals, NULL);
    checker.current_module = &root;
    checker.current_scope = &root.globals;
    success = predeclare_functions(&checker, program) &&
              check_program(&checker, program);
    scope_free(&root.globals);
    free(root.source);
    for (index = 0; index < checker.module_count; ++index)
        free_module(checker.modules[index]);
    free(checker.modules);
    free(checker.root_directory);
    return success && !diagnostic->has_error;
}
