#include "ast.h"

#include <stdlib.h>
#include <string.h>

AstNode *ast_new_node(AstKind kind, size_t line, size_t column) {
    AstNode *node = calloc(1, sizeof(*node));
    if (node != NULL) {
        node->kind = kind;
        node->line = line;
        node->column = column;
    }
    return node;
}

char *ast_copy_string(const char *source) {
    size_t length = strlen(source);
    char *copy = malloc(length + 1);
    if (copy != NULL) {
        memcpy(copy, source, length + 1);
    }
    return copy;
}

int ast_program_append(AstProgram *program, AstNode *statement) {
    AstNode **new_statements;
    size_t new_capacity;

    if (program->count == program->capacity) {
        new_capacity = program->capacity == 0 ? 8 : program->capacity * 2;
        new_statements = realloc(program->statements,
                                 new_capacity * sizeof(*new_statements));
        if (new_statements == NULL) {
            return 0;
        }
        program->statements = new_statements;
        program->capacity = new_capacity;
    }
    program->statements[program->count++] = statement;
    return 1;
}

int ast_block_append(AstNode *block, AstNode *statement) {
    AstNode **new_statements;
    size_t new_capacity;

    if (block->as.block.count == block->as.block.capacity) {
        new_capacity = block->as.block.capacity == 0
                           ? 4
                           : block->as.block.capacity * 2;
        new_statements = realloc(block->as.block.statements,
                                 new_capacity * sizeof(*new_statements));
        if (new_statements == NULL) {
            return 0;
        }
        block->as.block.statements = new_statements;
        block->as.block.capacity = new_capacity;
    }
    block->as.block.statements[block->as.block.count++] = statement;
    return 1;
}

int ast_function_append_parameter(AstNode *function, const char *name,
                                  TypeKind type) {
    AstParameter *new_parameters;
    AstParameter parameter;
    size_t new_capacity;

    parameter.name = ast_copy_string(name);
    if (parameter.name == NULL) return 0;
    parameter.type = type;
    if (function->as.function_declaration.parameter_count ==
        function->as.function_declaration.parameter_capacity) {
        new_capacity = function->as.function_declaration.parameter_capacity == 0
                           ? 4
                           : function->as.function_declaration.parameter_capacity * 2;
        new_parameters = realloc(function->as.function_declaration.parameters,
                                 new_capacity * sizeof(*new_parameters));
        if (new_parameters == NULL) {
            free(parameter.name);
            return 0;
        }
        function->as.function_declaration.parameters = new_parameters;
        function->as.function_declaration.parameter_capacity = new_capacity;
    }
    function->as.function_declaration.parameters
        [function->as.function_declaration.parameter_count++] = parameter;
    return 1;
}

int ast_call_append_argument(AstNode *call, AstNode *argument) {
    AstNode **new_arguments;
    size_t new_capacity;
    if (call->as.call.argument_count == call->as.call.argument_capacity) {
        new_capacity = call->as.call.argument_capacity == 0
                           ? 4
                           : call->as.call.argument_capacity * 2;
        new_arguments = realloc(call->as.call.arguments,
                                new_capacity * sizeof(*new_arguments));
        if (new_arguments == NULL) return 0;
        call->as.call.arguments = new_arguments;
        call->as.call.argument_capacity = new_capacity;
    }
    call->as.call.arguments[call->as.call.argument_count++] = argument;
    return 1;
}

int ast_from_import_append_name(AstNode *import_node, const char *name) {
    char **new_names;
    char *copy = ast_copy_string(name);
    size_t new_capacity;
    if (copy == NULL) return 0;
    if (import_node->as.from_import.name_count ==
        import_node->as.from_import.name_capacity) {
        new_capacity = import_node->as.from_import.name_capacity == 0
                           ? 4
                           : import_node->as.from_import.name_capacity * 2;
        new_names = realloc(import_node->as.from_import.names,
                            new_capacity * sizeof(*new_names));
        if (new_names == NULL) {
            free(copy);
            return 0;
        }
        import_node->as.from_import.names = new_names;
        import_node->as.from_import.name_capacity = new_capacity;
    }
    import_node->as.from_import.names[import_node->as.from_import.name_count++] = copy;
    return 1;
}

int ast_array_append_element(AstNode *array, AstNode *element) {
    AstNode **new_elements;
    size_t new_capacity;
    if (array->as.array.count == array->as.array.capacity) {
        new_capacity = array->as.array.capacity == 0 ? 4 : array->as.array.capacity * 2;
        new_elements = realloc(array->as.array.elements,
                               new_capacity * sizeof(*new_elements));
        if (new_elements == NULL) return 0;
        array->as.array.elements = new_elements;
        array->as.array.capacity = new_capacity;
    }
    array->as.array.elements[array->as.array.count++] = element;
    return 1;
}

void ast_node_free(AstNode *node) {
    if (node == NULL) {
        return;
    }
    switch (node->kind) {
        case AST_ASSIGNMENT:
            free(node->as.assignment.name);
            free(node->as.assignment.annotation);
            ast_node_free(node->as.assignment.value);
            break;
        case AST_PRINT:
            ast_node_free(node->as.print_statement.expression);
            break;
        case AST_EXPRESSION_STATEMENT:
            ast_node_free(node->as.expression_statement.expression);
            break;
        case AST_BINARY:
            ast_node_free(node->as.binary.left);
            ast_node_free(node->as.binary.right);
            break;
        case AST_UNARY:
            ast_node_free(node->as.unary.operand);
            break;
        case AST_IDENTIFIER:
            free(node->as.identifier);
            break;
        case AST_STRING:
            free(node->as.string);
            break;
        case AST_ARRAY: {
            size_t index;
            for (index = 0; index < node->as.array.count; ++index)
                ast_node_free(node->as.array.elements[index]);
            free(node->as.array.elements);
            break;
        }
        case AST_INDEX:
            ast_node_free(node->as.index.object);
            ast_node_free(node->as.index.index);
            break;
        case AST_NUMBER:
        case AST_BOOL:
            break;
        case AST_BLOCK: {
            size_t index;
            for (index = 0; index < node->as.block.count; ++index) {
                ast_node_free(node->as.block.statements[index]);
            }
            free(node->as.block.statements);
            break;
        }
        case AST_IF:
            ast_node_free(node->as.if_statement.condition);
            ast_node_free(node->as.if_statement.then_block);
            ast_node_free(node->as.if_statement.else_block);
            break;
        case AST_WHILE:
            ast_node_free(node->as.while_statement.condition);
            ast_node_free(node->as.while_statement.body);
            break;
        case AST_FOR:
            free(node->as.for_statement.name);
            ast_node_free(node->as.for_statement.iterable);
            ast_node_free(node->as.for_statement.body);
            break;
        case AST_INDEX_ASSIGNMENT:
            ast_node_free(node->as.index_assignment.target);
            ast_node_free(node->as.index_assignment.value);
            break;
        case AST_FUNCTION_DECLARATION: {
            size_t index;
            free(node->as.function_declaration.name);
            for (index = 0; index < node->as.function_declaration.parameter_count; ++index) {
                free(node->as.function_declaration.parameters[index].name);
            }
            free(node->as.function_declaration.parameters);
            ast_node_free(node->as.function_declaration.body);
            break;
        }
        case AST_RETURN:
            ast_node_free(node->as.return_statement.value);
            break;
        case AST_IMPORT:
            free(node->as.import_statement.module_name);
            free(node->as.import_statement.alias);
            break;
        case AST_FROM_IMPORT: {
            size_t index;
            free(node->as.from_import.module_name);
            for (index = 0; index < node->as.from_import.name_count; ++index) {
                free(node->as.from_import.names[index]);
            }
            free(node->as.from_import.names);
            break;
        }
        case AST_CALL: {
            size_t index;
            ast_node_free(node->as.call.callee);
            for (index = 0; index < node->as.call.argument_count; ++index) {
                ast_node_free(node->as.call.arguments[index]);
            }
            free(node->as.call.arguments);
            break;
        }
        case AST_MEMBER:
            ast_node_free(node->as.member.object);
            free(node->as.member.member);
            break;
    }
    free(node);
}

void ast_program_free(AstProgram *program) {
    size_t index;

    if (program == NULL) {
        return;
    }
    for (index = 0; index < program->count; ++index) {
        ast_node_free(program->statements[index]);
    }
    free(program->statements);
    *program = (AstProgram){0};
}
