#include "ir.h"

#include "runtime/native.h"

#include <stdlib.h>
#include <string.h>

typedef struct {
    IrProgram *program;
    IrFunction *function;
    bool is_main;
    bool native_tensor;
} Builder;

static char *copy_text(const char *text) {
    size_t length = strlen(text);
    char *copy = malloc(length + 1);
    if (copy != NULL) memcpy(copy, text, length + 1);
    return copy;
}

static ValueKind source_kind(TypeKind type) {
    switch (type) {
        case TYPE_INT: return VALUE_INT;
        case TYPE_BOOL: return VALUE_BOOL;
        case TYPE_STRING: return VALUE_STRING;
        case TYPE_LIST: return VALUE_ARRAY;
        case TYPE_TENSOR: return VALUE_TENSOR;
        case TYPE_VOID: return VALUE_NULL;
    }
    return VALUE_NULL;
}

static void unsupported(Builder *builder, const char *reason) {
    if (!builder->program->supported) return;
    builder->program->supported = false;
    snprintf(builder->program->unsupported_reason,
             sizeof(builder->program->unsupported_reason), "%s", reason);
}

static bool append_instruction(IrFunction *function, IrOp op, int a, int b,
                               Value constant, const AstNode *node) {
    IrInstr *grown;
    if (function->count == function->capacity) {
        size_t next = function->capacity == 0 ? 32 : function->capacity * 2;
        grown = realloc(function->code, next * sizeof(*grown));
        if (grown == NULL) return false;
        function->code = grown;
        function->capacity = next;
    }
    function->code[function->count++] = (IrInstr){
        op, a, b, constant,
        {node == NULL ? 1 : node->line, node == NULL ? 1 : node->column}
    };
    return true;
}

static int find_slot(const LocalSlot *slots, size_t count, const char *name) {
    size_t index;
    for (index = 0; index < count; ++index)
        if (strcmp(slots[index].name, name) == 0) return slots[index].slot;
    return -1;
}

static ValueKind slot_kind(const LocalSlot *slots, size_t count, int slot) {
    return slot < 0 || (size_t)slot >= count ? VALUE_NULL : slots[slot].kind;
}

static int add_slot(LocalSlot **slots, size_t *count, size_t *capacity,
                    const char *name, ValueKind kind) {
    LocalSlot *grown;
    char *copy;
    int existing = find_slot(*slots, *count, name);
    if (existing >= 0) return existing;
    if (*count == *capacity) {
        size_t next = *capacity == 0 ? 8 : *capacity * 2;
        grown = realloc(*slots, next * sizeof(*grown));
        if (grown == NULL) return -1;
        *slots = grown;
        *capacity = next;
    }
    copy = copy_text(name);
    if (copy == NULL) return -1;
    (*slots)[*count] = (LocalSlot){copy, (int)*count, kind};
    return (int)(*count)++;
}

static int function_index(const IrProgram *program, const char *name) {
    size_t index;
    for (index = 1; index < program->function_count; ++index)
        if (strcmp(program->functions[index].name, name) == 0) return (int)index;
    return -1;
}

static ValueKind build_expression(Builder *builder, const AstNode *node);
static bool build_statement(Builder *builder, const AstNode *node);

static ValueKind identifier_kind(Builder *builder, const char *name,
                                 int *slot, bool *global) {
    *slot = find_slot(builder->function->slots,
                      builder->function->slot_count, name);
    if (*slot >= 0) {
        *global = false;
        return slot_kind(builder->function->slots,
                         builder->function->slot_count, *slot);
    }
    *slot = find_slot(builder->program->globals,
                      builder->program->global_count, name);
    *global = true;
    return slot_kind(builder->program->globals,
                     builder->program->global_count, *slot);
}

static int builtin_id(const char *name) {
    if (strcmp(name, "len") == 0) return IR_NATIVE_LEN;
    if (strcmp(name, "range") == 0) return IR_NATIVE_RANGE;
    if (strcmp(name, "tensor") == 0) return IR_NATIVE_TENSOR;
    if (strcmp(name, "shape") == 0) return IR_NATIVE_SHAPE;
    if (strcmp(name, "rank") == 0) return IR_NATIVE_RANK;
    if (strcmp(name, "zeros") == 0) return IR_NATIVE_ZEROS;
    if (strcmp(name, "ones") == 0) return IR_NATIVE_ONES;
    if (strcmp(name, "arange") == 0) return IR_NATIVE_ARANGE;
    return -1;
}

static ValueKind native_return_kind(int id) {
    return id == IR_NATIVE_LEN || id == IR_NATIVE_RANK ? VALUE_INT
           : id == IR_NATIVE_RANGE ? VALUE_RANGE
                                   : VALUE_TENSOR;
}

static ValueKind build_call(Builder *builder, const AstNode *node) {
    const AstNode *callee = node->as.call.callee;
    size_t index;
    if (callee->kind == AST_IDENTIFIER) {
        int native = builtin_id(callee->as.identifier);
        int function = function_index(builder->program, callee->as.identifier);
        for (index = 0; index < node->as.call.argument_count; ++index)
            build_expression(builder, node->as.call.arguments[index]);
        if (function >= 0) {
            append_instruction(builder->function, IR_CALL, function,
                               (int)node->as.call.argument_count,
                               value_null(), node);
            return builder->program->functions[function].return_kind;
        }
        if (native >= 0) {
            append_instruction(builder->function, IR_CALL_NATIVE, native,
                               (int)node->as.call.argument_count,
                               value_null(), node);
            return native_return_kind(native);
        }
        unsupported(builder, "dynamic function call");
        return VALUE_NULL;
    }
    if (callee->kind == AST_MEMBER && node->as.call.native_method !=
                                      AST_NATIVE_METHOD_NONE) {
        build_expression(builder, callee->as.member.object);
        for (index = 0; index < node->as.call.argument_count; ++index)
            build_expression(builder, node->as.call.arguments[index]);
        append_instruction(builder->function, IR_CALL_METHOD,
                           (int)node->as.call.native_method,
                           (int)node->as.call.argument_count,
                           value_null(), node);
        return node->as.call.native_method == AST_NATIVE_TENSOR_SUM ||
                       node->as.call.native_method == AST_NATIVE_TENSOR_MEAN
                   ? VALUE_TENSOR
                   : VALUE_NULL;
    }
    if (callee->kind == AST_MEMBER && builder->native_tensor &&
        callee->as.member.object->kind == AST_IDENTIFIER &&
        strcmp(callee->as.member.object->as.identifier, "tensor") == 0) {
        int native = builtin_id(callee->as.member.member);
        if (native >= 0) {
            for (index = 0; index < node->as.call.argument_count; ++index)
                build_expression(builder, node->as.call.arguments[index]);
            append_instruction(builder->function, IR_CALL_NATIVE, native,
                               (int)node->as.call.argument_count,
                               value_null(), node);
            return native_return_kind(native);
        }
    }
    unsupported(builder, "module or dynamic method call");
    return VALUE_NULL;
}

static ValueKind build_expression(Builder *builder, const AstNode *node) {
    switch (node->kind) {
        case AST_NUMBER:
            append_instruction(builder->function, IR_CONST, 0, 0,
                               value_int(node->as.number), node);
            return VALUE_INT;
        case AST_BOOL:
            append_instruction(builder->function, IR_CONST, 0, 0,
                               value_bool(node->as.boolean), node);
            return VALUE_BOOL;
        case AST_STRING: {
            Value value = value_null();
            if (!value_string(node->as.string, &value)) {
                unsupported(builder, "out of memory");
                return VALUE_NULL;
            }
            append_instruction(builder->function, IR_CONST, 0, 0, value, node);
            return VALUE_STRING;
        }
        case AST_ARRAY: {
            size_t index;
            for (index = 0; index < node->as.array.count; ++index)
                build_expression(builder, node->as.array.elements[index]);
            append_instruction(builder->function, IR_BUILD_ARRAY,
                               (int)node->as.array.count, 0, value_null(), node);
            return VALUE_ARRAY;
        }
        case AST_IDENTIFIER: {
            int slot;
            bool global;
            ValueKind kind = identifier_kind(builder, node->as.identifier,
                                              &slot, &global);
            if (slot < 0) {
                unsupported(builder, "unresolved identifier in VM lowering");
                return VALUE_NULL;
            }
            append_instruction(builder->function,
                               global ? IR_LOAD_GLOBAL : IR_LOAD_LOCAL,
                               slot, 0, value_null(), node);
            return kind;
        }
        case AST_BINARY: {
            ValueKind left = build_expression(builder, node->as.binary.left);
            ValueKind right = build_expression(builder, node->as.binary.right);
            TokenType token = node->as.binary.operator_type;
            IrOp op;
            if (left == VALUE_TENSOR || right == VALUE_TENSOR) {
                op = token == TOKEN_PLUS ? IR_TENSOR_ADD
                     : token == TOKEN_MINUS ? IR_TENSOR_SUB
                     : token == TOKEN_STAR ? IR_TENSOR_MUL
                                           : IR_TENSOR_DIV;
                append_instruction(builder->function, op, 0, 0,
                                   value_null(), node);
                return VALUE_TENSOR;
            }
            if (left == VALUE_STRING && token == TOKEN_PLUS) op = IR_ADD_STR;
            else if (token == TOKEN_PLUS) op = IR_ADD_INT;
            else if (token == TOKEN_MINUS) op = IR_SUB_INT;
            else if (token == TOKEN_STAR) op = IR_MUL_INT;
            else if (token == TOKEN_SLASH) op = IR_DIV_INT;
            else if (token == TOKEN_EQUAL_EQUAL) op = IR_EQ;
            else if (token == TOKEN_BANG_EQUAL) op = IR_NEQ;
            else if (token == TOKEN_LESS) op = IR_LT_INT;
            else if (token == TOKEN_LESS_EQUAL) op = IR_LTE_INT;
            else if (token == TOKEN_GREATER) op = IR_GT_INT;
            else if (token == TOKEN_GREATER_EQUAL) op = IR_GTE_INT;
            else if (token == TOKEN_AND) op = IR_AND_BOOL;
            else op = IR_OR_BOOL;
            append_instruction(builder->function, op, 0, 0, value_null(), node);
            return (token == TOKEN_EQUAL_EQUAL || token == TOKEN_BANG_EQUAL ||
                    token == TOKEN_LESS || token == TOKEN_LESS_EQUAL ||
                    token == TOKEN_GREATER || token == TOKEN_GREATER_EQUAL ||
                    token == TOKEN_AND || token == TOKEN_OR)
                       ? VALUE_BOOL : left;
        }
        case AST_UNARY: {
            ValueKind kind = build_expression(builder, node->as.unary.operand);
            append_instruction(builder->function,
                               node->as.unary.operator_type == TOKEN_NOT
                                   ? IR_NOT_BOOL : IR_NEG_INT,
                               0, 0, value_null(), node);
            return kind;
        }
        case AST_INDEX:
            build_expression(builder, node->as.index.object);
            build_expression(builder, node->as.index.index);
            append_instruction(builder->function, IR_INDEX, 0, 0,
                               value_null(), node);
            return VALUE_NULL;
        case AST_MEMBER: {
            ValueKind object = build_expression(builder, node->as.member.object);
            int property = strcmp(node->as.member.member, "len") == 0 ? IR_PROP_LEN
                : strcmp(node->as.member.member, "shape") == 0 ? IR_PROP_SHAPE
                : strcmp(node->as.member.member, "rank") == 0 ? IR_PROP_RANK
                : strcmp(node->as.member.member, "grad") == 0 ? IR_PROP_GRAD
                : IR_PROP_REQUIRES_GRAD;
            append_instruction(builder->function, IR_GET_PROPERTY, property,
                               (int)object, value_null(), node);
            return property == IR_PROP_LEN || property == IR_PROP_RANK
                       ? VALUE_INT
                   : property == IR_PROP_REQUIRES_GRAD ? VALUE_BOOL
                   : property == IR_PROP_SHAPE ? VALUE_ARRAY : VALUE_TENSOR;
        }
        case AST_CALL:
            return build_call(builder, node);
        default:
            unsupported(builder, "unsupported expression in VM lowering");
            return VALUE_NULL;
    }
}

static bool build_block(Builder *builder, const AstNode *block) {
    size_t index;
    for (index = 0; index < block->as.block.count; ++index)
        if (!build_statement(builder, block->as.block.statements[index]))
            return false;
    return true;
}

static bool build_statement(Builder *builder, const AstNode *node) {
    switch (node->kind) {
        case AST_ASSIGNMENT: {
            ValueKind kind = build_expression(builder, node->as.assignment.value);
            int slot;
            bool global = builder->is_main ||
                (!node->as.assignment.is_declaration &&
                 !node->as.assignment.is_inferred_declaration &&
                 find_slot(builder->program->globals,
                           builder->program->global_count,
                           node->as.assignment.name) >= 0);
            if (global) {
                slot = add_slot(&builder->program->globals,
                                &builder->program->global_count,
                                &builder->program->global_capacity,
                                node->as.assignment.name, kind);
                builder->program->globals[slot].kind = kind;
            } else {
                slot = add_slot(&builder->function->slots,
                                &builder->function->slot_count,
                                &builder->function->slot_capacity,
                                node->as.assignment.name, kind);
                builder->function->slots[slot].kind = kind;
            }
            append_instruction(builder->function,
                               global ? IR_STORE_GLOBAL : IR_STORE_LOCAL,
                               slot, 0, value_null(), node);
            return true;
        }
        case AST_PRINT:
            build_expression(builder, node->as.print_statement.expression);
            return append_instruction(builder->function, IR_PRINT, 0, 0,
                                      value_null(), node);
        case AST_EXPRESSION_STATEMENT:
            build_expression(builder, node->as.expression_statement.expression);
            return append_instruction(builder->function, IR_POP, 0, 0,
                                      value_null(), node);
        case AST_IF: {
            size_t branch, end;
            build_expression(builder, node->as.if_statement.condition);
            branch = builder->function->count;
            append_instruction(builder->function, IR_JUMP_IF_FALSE, 0, 0,
                               value_null(), node);
            build_block(builder, node->as.if_statement.then_block);
            if (node->as.if_statement.else_block != NULL) {
                end = builder->function->count;
                append_instruction(builder->function, IR_JUMP, 0, 0,
                                   value_null(), node);
                builder->function->code[branch].a = (int)builder->function->count;
                build_block(builder, node->as.if_statement.else_block);
                builder->function->code[end].a = (int)builder->function->count;
            } else {
                builder->function->code[branch].a = (int)builder->function->count;
            }
            return true;
        }
        case AST_WHILE: {
            size_t start = builder->function->count;
            size_t branch;
            build_expression(builder, node->as.while_statement.condition);
            branch = builder->function->count;
            append_instruction(builder->function, IR_JUMP_IF_FALSE, 0, 0,
                               value_null(), node);
            build_block(builder, node->as.while_statement.body);
            append_instruction(builder->function, IR_JUMP, (int)start, 0,
                               value_null(), node);
            builder->function->code[branch].a = (int)builder->function->count;
            return true;
        }
        case AST_RETURN:
            if (node->as.return_statement.value == NULL)
                append_instruction(builder->function, IR_CONST, 0, 0,
                                   value_null(), node);
            else
                build_expression(builder, node->as.return_statement.value);
            return append_instruction(builder->function, IR_RETURN, 0, 0,
                                      value_null(), node);
        case AST_FUNCTION_DECLARATION:
            return true;
        case AST_FROM_IMPORT:
            if (strcmp(node->as.from_import.module_name, "tensor") != 0 ||
                !builder->native_tensor)
                unsupported(builder, "user module import");
            return true;
        case AST_IMPORT:
            if (strcmp(node->as.import_statement.module_name, "tensor") != 0 ||
                !builder->native_tensor)
                unsupported(builder, "user module import");
            return true;
        case AST_FOR: {
            char iterator_name[64];
            char end_name[64];
            char index_name[64];
            int iterator_slot, end_slot, variable_slot, index_slot;
            bool global = builder->is_main;
            size_t start, branch;
            const AstNode *iterable = node->as.for_statement.iterable;
            bool range_loop = iterable->kind == AST_CALL &&
                iterable->as.call.callee->kind == AST_IDENTIFIER &&
                strcmp(iterable->as.call.callee->as.identifier, "range") == 0;
            snprintf(iterator_name, sizeof(iterator_name), "$iter_%zu",
                     builder->function->count);
            snprintf(end_name, sizeof(end_name), "$end_%zu",
                     builder->function->count);
            snprintf(index_name, sizeof(index_name), "$index_%zu",
                     builder->function->count);
            if (global) {
                iterator_slot = add_slot(&builder->program->globals,
                    &builder->program->global_count,
                    &builder->program->global_capacity,
                    iterator_name, range_loop ? VALUE_INT : VALUE_ARRAY);
                end_slot = add_slot(&builder->program->globals,
                    &builder->program->global_count,
                    &builder->program->global_capacity,
                    end_name, VALUE_INT);
                variable_slot = add_slot(&builder->program->globals,
                    &builder->program->global_count,
                    &builder->program->global_capacity,
                    node->as.for_statement.name, VALUE_INT);
                index_slot = range_loop ? iterator_slot : add_slot(
                    &builder->program->globals,
                    &builder->program->global_count,
                    &builder->program->global_capacity,
                    index_name, VALUE_INT);
            } else {
                iterator_slot = add_slot(&builder->function->slots,
                    &builder->function->slot_count,
                    &builder->function->slot_capacity,
                    iterator_name, range_loop ? VALUE_INT : VALUE_ARRAY);
                end_slot = add_slot(&builder->function->slots,
                    &builder->function->slot_count,
                    &builder->function->slot_capacity, end_name, VALUE_INT);
                variable_slot = add_slot(&builder->function->slots,
                    &builder->function->slot_count,
                    &builder->function->slot_capacity,
                    node->as.for_statement.name, VALUE_INT);
                index_slot = range_loop ? iterator_slot : add_slot(
                    &builder->function->slots,
                    &builder->function->slot_count,
                    &builder->function->slot_capacity,
                    index_name, VALUE_INT);
            }
            if (range_loop) {
                build_expression(builder, iterable->as.call.arguments[0]);
                append_instruction(builder->function,
                    global ? IR_STORE_GLOBAL : IR_STORE_LOCAL,
                    iterator_slot, 0, value_null(), node);
                build_expression(builder, iterable->as.call.arguments[1]);
                append_instruction(builder->function,
                    global ? IR_STORE_GLOBAL : IR_STORE_LOCAL,
                    end_slot, 0, value_null(), node);
            } else {
                build_expression(builder, iterable);
                append_instruction(builder->function,
                    global ? IR_STORE_GLOBAL : IR_STORE_LOCAL,
                    iterator_slot, 0, value_null(), node);
                append_instruction(builder->function,
                    global ? IR_LOAD_GLOBAL : IR_LOAD_LOCAL,
                    iterator_slot, 0, value_null(), node);
                append_instruction(builder->function, IR_CALL_NATIVE,
                                   IR_NATIVE_LEN, 1, value_null(), node);
                append_instruction(builder->function,
                    global ? IR_STORE_GLOBAL : IR_STORE_LOCAL,
                    end_slot, 0, value_null(), node);
                append_instruction(builder->function, IR_CONST, 0, 0,
                                   value_int(0), node);
                append_instruction(builder->function,
                    global ? IR_STORE_GLOBAL : IR_STORE_LOCAL,
                    index_slot, 0, value_null(), node);
            }
            start = builder->function->count;
            append_instruction(builder->function,
                global ? IR_LOAD_GLOBAL : IR_LOAD_LOCAL,
                range_loop ? iterator_slot : index_slot,
                0, value_null(), node);
            append_instruction(builder->function,
                global ? IR_LOAD_GLOBAL : IR_LOAD_LOCAL,
                end_slot, 0, value_null(), node);
            append_instruction(builder->function, IR_LT_INT, 0, 0,
                               value_null(), node);
            branch = builder->function->count;
            append_instruction(builder->function, IR_JUMP_IF_FALSE, 0, 0,
                               value_null(), node);
            if (range_loop) {
                append_instruction(builder->function,
                    global ? IR_LOAD_GLOBAL : IR_LOAD_LOCAL,
                    iterator_slot, 0, value_null(), node);
                append_instruction(builder->function,
                    global ? IR_STORE_GLOBAL : IR_STORE_LOCAL,
                    variable_slot, 0, value_null(), node);
            } else {
                append_instruction(builder->function,
                    global ? IR_LOAD_GLOBAL : IR_LOAD_LOCAL,
                    iterator_slot, 0, value_null(), node);
                append_instruction(builder->function,
                    global ? IR_LOAD_GLOBAL : IR_LOAD_LOCAL,
                    index_slot, 0, value_null(), node);
                append_instruction(builder->function, IR_INDEX, 0, 0,
                                   value_null(), node);
                append_instruction(builder->function,
                    global ? IR_STORE_GLOBAL : IR_STORE_LOCAL,
                    variable_slot, 0, value_null(), node);
            }
            build_block(builder, node->as.for_statement.body);
            append_instruction(builder->function,
                global ? IR_LOAD_GLOBAL : IR_LOAD_LOCAL,
                range_loop ? iterator_slot : index_slot,
                0, value_null(), node);
            append_instruction(builder->function, IR_CONST, 0, 0,
                               value_int(1), node);
            append_instruction(builder->function, IR_ADD_INT, 0, 0,
                               value_null(), node);
            append_instruction(builder->function,
                global ? IR_STORE_GLOBAL : IR_STORE_LOCAL,
                range_loop ? iterator_slot : index_slot,
                0, value_null(), node);
            append_instruction(builder->function, IR_JUMP, (int)start, 0,
                               value_null(), node);
            builder->function->code[branch].a = (int)builder->function->count;
            return true;
        }
        case AST_INDEX_ASSIGNMENT:
            unsupported(builder, "index assignment");
            return true;
        case AST_BLOCK:
            return build_block(builder, node);
        default:
            unsupported(builder, "unsupported statement in VM lowering");
            return true;
    }
}

static bool add_function(IrProgram *program, const char *name) {
    IrFunction *grown;
    if (program->function_count == program->function_capacity) {
        size_t next = program->function_capacity == 0 ? 4
                                                       : program->function_capacity * 2;
        grown = realloc(program->functions, next * sizeof(*grown));
        if (grown == NULL) return false;
        program->functions = grown;
        program->function_capacity = next;
    }
    memset(&program->functions[program->function_count], 0,
           sizeof(*program->functions));
    program->functions[program->function_count].name = copy_text(name);
    if (program->functions[program->function_count].name == NULL) return false;
    ++program->function_count;
    return true;
}

bool ir_build(const AstProgram *ast, const char *entry_path, IrProgram *program) {
    Builder builder;
    size_t index;
    const char *slash;
    char module_path[1024];
    FILE *module_file;
    memset(program, 0, sizeof(*program));
    program->supported = true;
    if (!add_function(program, "__main__")) return false;
    for (index = 0; index < ast->count; ++index)
        if (ast->statements[index]->kind == AST_FUNCTION_DECLARATION &&
            !add_function(program,
                ast->statements[index]->as.function_declaration.name))
            return false;
    for (index = 0; index < ast->count; ++index) {
        const AstNode *node = ast->statements[index];
        if (node->kind == AST_FUNCTION_DECLARATION) {
            int fi = function_index(program, node->as.function_declaration.name);
            program->functions[fi].return_kind =
                source_kind(node->as.function_declaration.return_type);
        }
    }
    builder.program = program;
    slash = strrchr(entry_path, '/');
    {
        const char *backslash = strrchr(entry_path, '\\');
        if (backslash != NULL && (slash == NULL || backslash > slash))
            slash = backslash;
    }
    if (slash == NULL)
        snprintf(module_path, sizeof(module_path), "tensor.fl");
    else
        snprintf(module_path, sizeof(module_path), "%.*s/tensor.fl",
                 (int)(slash - entry_path), entry_path);
    module_file = fopen(module_path, "rb");
    builder.native_tensor = module_file == NULL;
    if (module_file != NULL) fclose(module_file);
    builder.function = &program->functions[0];
    builder.is_main = true;
    for (index = 0; index < ast->count; ++index)
        build_statement(&builder, ast->statements[index]);
    append_instruction(builder.function, IR_HALT, 0, 0, value_null(), NULL);
    for (index = 0; index < ast->count; ++index) {
        const AstNode *node = ast->statements[index];
        size_t parameter;
        int fi;
        if (node->kind != AST_FUNCTION_DECLARATION) continue;
        fi = function_index(program, node->as.function_declaration.name);
        builder.function = &program->functions[fi];
        builder.is_main = false;
        builder.function->parameter_count =
            node->as.function_declaration.parameter_count;
        for (parameter = 0; parameter < builder.function->parameter_count;
             ++parameter)
            add_slot(&builder.function->slots, &builder.function->slot_count,
                     &builder.function->slot_capacity,
                     node->as.function_declaration.parameters[parameter].name,
                     source_kind(node->as.function_declaration
                                     .parameters[parameter].type));
        build_block(&builder, node->as.function_declaration.body);
        if (builder.function->count == 0 ||
            builder.function->code[builder.function->count - 1].op != IR_RETURN) {
            append_instruction(builder.function, IR_CONST, 0, 0,
                               value_null(), node);
            append_instruction(builder.function, IR_RETURN, 0, 0,
                               value_null(), node);
        }
    }
    return true;
}

void ir_program_free(IrProgram *program) {
    size_t function, index;
    for (function = 0; function < program->function_count; ++function) {
        IrFunction *fn = &program->functions[function];
        for (index = 0; index < fn->count; ++index)
            if (fn->code[index].op == IR_CONST)
                value_free(&fn->code[index].constant);
        for (index = 0; index < fn->slot_count; ++index)
            free(fn->slots[index].name);
        free(fn->slots);
        free(fn->code);
        free(fn->name);
    }
    for (index = 0; index < program->global_count; ++index)
        free(program->globals[index].name);
    free(program->globals);
    free(program->functions);
    memset(program, 0, sizeof(*program));
}

const char *ir_op_name(IrOp op) {
    static const char *names[] = {
        "CONST", "LOAD_LOCAL", "STORE_LOCAL", "LOAD_GLOBAL", "STORE_GLOBAL",
        "ADD_INT", "SUB_INT", "MUL_INT", "DIV_INT", "ADD_STR", "EQ", "NEQ",
        "LT_INT", "LTE_INT", "GT_INT", "GTE_INT", "NEG_INT", "NOT_BOOL",
        "AND_BOOL", "OR_BOOL", "JUMP", "JUMP_IF_FALSE", "CALL", "CALL_NATIVE",
        "RETURN", "PRINT", "POP", "BUILD_ARRAY", "INDEX", "GET_PROPERTY",
        "CALL_METHOD", "TENSOR_ADD", "TENSOR_SUB", "TENSOR_MUL", "TENSOR_DIV",
        "HALT", "NOP"
    };
    return names[(int)op];
}

void ir_dump(FILE *output, const IrProgram *program) {
    size_t function, index;
    for (function = 0; function < program->function_count; ++function) {
        const IrFunction *fn = &program->functions[function];
        fprintf(output, "IR function %s (slots=%zu):\n", fn->name, fn->slot_count);
        for (index = 0; index < fn->count; ++index) {
            const IrInstr *in = &fn->code[index];
            fprintf(output, "  %04zu  %-20s", index, ir_op_name(in->op));
            if (in->op == IR_CONST) {
                if (in->constant.kind == VALUE_INT)
                    fprintf(output, " %lld", (long long)in->constant.as.integer);
                else if (in->constant.kind == VALUE_BOOL)
                    fprintf(output, " %s", in->constant.as.boolean ? "true" : "false");
                else if (in->constant.kind == VALUE_STRING)
                    fprintf(output, " \"%s\"", in->constant.as.string.data);
                else fprintf(output, " null");
            } else if (in->a != 0 || in->op == IR_LOAD_LOCAL ||
                       in->op == IR_STORE_LOCAL || in->op == IR_LOAD_GLOBAL ||
                       in->op == IR_STORE_GLOBAL)
                fprintf(output, " %d", in->a);
            fputc('\n', output);
        }
    }
    if (!program->supported)
        fprintf(output, "IR fallback: %s\n", program->unsupported_reason);
}
