#ifndef FLANG_AST_H
#define FLANG_AST_H

#include "token.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

typedef enum {
    AST_ASSIGNMENT,
    AST_PRINT,
    AST_BINARY,
    AST_UNARY,
    AST_NUMBER,
    AST_BOOL,
    AST_STRING,
    AST_ARRAY,
    AST_INDEX,
    AST_IDENTIFIER,
    AST_EXPRESSION_STATEMENT,
    AST_BLOCK,
    AST_IF,
    AST_WHILE,
    AST_FOR,
    AST_INDEX_ASSIGNMENT,
    AST_FUNCTION_DECLARATION,
    AST_RETURN,
    AST_IMPORT,
    AST_FROM_IMPORT,
    AST_CALL,
    AST_MEMBER
} AstKind;

typedef struct AstNode AstNode;

typedef enum {
    AST_NATIVE_METHOD_NONE,
    AST_NATIVE_TENSOR_SUM,
    AST_NATIVE_TENSOR_MEAN,
    AST_NATIVE_TENSOR_BACKWARD,
    AST_NATIVE_TENSOR_ZERO_GRAD
} AstNativeMethod;

typedef enum {
    TYPE_INT,
    TYPE_BOOL,
    TYPE_VOID,
    TYPE_STRING,
    TYPE_LIST,
    TYPE_TENSOR
} TypeKind;

typedef struct {
    char *name;
    TypeKind type;
} AstParameter;

struct AstNode {
    AstKind kind;
    size_t line;
    size_t column;
    union {
        struct {
            char *name;
            char *annotation;
            AstNode *value;
            bool is_declaration;
            bool is_inferred_declaration;
            bool is_public;
        } assignment;
        struct {
            AstNode *expression;
        } print_statement;
        struct {
            AstNode *expression;
        } expression_statement;
        struct {
            TokenType operator_type;
            AstNode *left;
            AstNode *right;
        } binary;
        struct {
            TokenType operator_type;
            AstNode *operand;
        } unary;
        int64_t number;
        bool boolean;
        char *string;
        char *identifier;
        struct {
            AstNode **elements;
            size_t count;
            size_t capacity;
        } array;
        struct {
            AstNode *object;
            AstNode *index;
        } index;
        struct {
            AstNode **statements;
            size_t count;
            size_t capacity;
        } block;
        struct {
            AstNode *condition;
            AstNode *then_block;
            AstNode *else_block;
        } if_statement;
        struct {
            AstNode *condition;
            AstNode *body;
        } while_statement;
        struct {
            char *name;
            AstNode *iterable;
            AstNode *body;
        } for_statement;
        struct {
            AstNode *target;
            AstNode *value;
        } index_assignment;
        struct {
            char *name;
            AstParameter *parameters;
            size_t parameter_count;
            size_t parameter_capacity;
            TypeKind return_type;
            AstNode *body;
            bool is_public;
        } function_declaration;
        struct {
            AstNode *value;
        } return_statement;
        struct {
            char *module_name;
            char *alias;
        } import_statement;
        struct {
            char *module_name;
            char **names;
            size_t name_count;
            size_t name_capacity;
        } from_import;
        struct {
            AstNode *callee;
            AstNode **arguments;
            size_t argument_count;
            size_t argument_capacity;
            AstNativeMethod native_method;
        } call;
        struct {
            AstNode *object;
            char *member;
        } member;
    } as;
};

typedef struct {
    AstNode **statements;
    size_t count;
    size_t capacity;
} AstProgram;

AstNode *ast_new_node(AstKind kind, size_t line, size_t column);
char *ast_copy_string(const char *source);
int ast_program_append(AstProgram *program, AstNode *statement);
int ast_block_append(AstNode *block, AstNode *statement);
int ast_function_append_parameter(AstNode *function, const char *name,
                                  TypeKind type);
int ast_call_append_argument(AstNode *call, AstNode *argument);
int ast_from_import_append_name(AstNode *import_node, const char *name);
int ast_array_append_element(AstNode *array, AstNode *element);
void ast_node_free(AstNode *node);
void ast_program_free(AstProgram *program);

#endif
