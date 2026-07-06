#ifndef FLANG_IR_H
#define FLANG_IR_H

#include "ast.h"
#include "value.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

typedef enum {
    IR_CONST,
    IR_LOAD_LOCAL, IR_STORE_LOCAL, IR_LOAD_GLOBAL, IR_STORE_GLOBAL,
    IR_ADD_INT, IR_SUB_INT, IR_MUL_INT, IR_DIV_INT,
    IR_ADD_STR,
    IR_EQ, IR_NEQ, IR_LT_INT, IR_LTE_INT, IR_GT_INT, IR_GTE_INT,
    IR_NEG_INT, IR_NOT_BOOL, IR_AND_BOOL, IR_OR_BOOL,
    IR_JUMP, IR_JUMP_IF_FALSE,
    IR_CALL, IR_CALL_NATIVE, IR_RETURN, IR_PRINT, IR_POP,
    IR_BUILD_ARRAY, IR_INDEX, IR_GET_PROPERTY, IR_CALL_METHOD,
    IR_TENSOR_ADD, IR_TENSOR_SUB, IR_TENSOR_MUL, IR_TENSOR_DIV,
    IR_HALT, IR_NOP
} IrOp;

typedef enum {
    IR_NATIVE_LEN,
    IR_NATIVE_RANGE,
    IR_NATIVE_TENSOR,
    IR_NATIVE_SHAPE,
    IR_NATIVE_RANK,
    IR_NATIVE_ZEROS,
    IR_NATIVE_ONES,
    IR_NATIVE_ARANGE
} IrNative;

typedef enum {
    IR_PROP_LEN,
    IR_PROP_SHAPE,
    IR_PROP_RANK,
    IR_PROP_GRAD,
    IR_PROP_REQUIRES_GRAD
} IrProperty;

typedef struct {
    size_t line;
    size_t column;
} SourceSpan;

typedef struct {
    IrOp op;
    int a;
    int b;
    Value constant;
    SourceSpan span;
} IrInstr;

typedef struct {
    char *name;
    int slot;
    ValueKind kind;
} LocalSlot;

typedef struct {
    char *name;
    IrInstr *code;
    size_t count;
    size_t capacity;
    LocalSlot *slots;
    size_t slot_count;
    size_t slot_capacity;
    size_t parameter_count;
    ValueKind return_kind;
} IrFunction;

typedef struct {
    IrFunction *functions;
    size_t function_count;
    size_t function_capacity;
    LocalSlot *globals;
    size_t global_count;
    size_t global_capacity;
    bool supported;
    char unsupported_reason[256];
} IrProgram;

bool ir_build(const AstProgram *ast, const char *entry_path, IrProgram *program);
void ir_program_free(IrProgram *program);
void ir_dump(FILE *output, const IrProgram *program);
const char *ir_op_name(IrOp op);

#endif
