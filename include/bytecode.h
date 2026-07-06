#ifndef FLANG_BYTECODE_H
#define FLANG_BYTECODE_H

#include "ir.h"

#include <stdint.h>

typedef enum {
    OP_CONST,
    OP_LOAD_LOCAL, OP_STORE_LOCAL, OP_LOAD_GLOBAL, OP_STORE_GLOBAL,
    OP_ADD_INT, OP_SUB_INT, OP_MUL_INT, OP_DIV_INT, OP_ADD_STR,
    OP_EQ, OP_NEQ, OP_LT_INT, OP_LTE_INT, OP_GT_INT, OP_GTE_INT,
    OP_NEG_INT, OP_NOT_BOOL, OP_AND_BOOL, OP_OR_BOOL,
    OP_JUMP, OP_JUMP_IF_FALSE,
    OP_CALL, OP_CALL_NATIVE, OP_RETURN, OP_PRINT, OP_POP,
    OP_BUILD_ARRAY, OP_INDEX, OP_GET_PROPERTY, OP_CALL_METHOD,
    OP_TENSOR_ADD, OP_TENSOR_SUB, OP_TENSOR_MUL, OP_TENSOR_DIV,
    OP_HALT, OP_NOP
} OpCode;

typedef struct {
    uint8_t op;
    int32_t a;
    int32_t b;
    Value constant;
    SourceSpan span;
} BytecodeInstr;

typedef struct {
    char *name;
    BytecodeInstr *code;
    size_t count;
    size_t local_count;
    size_t parameter_count;
} BytecodeFunction;

typedef struct {
    BytecodeFunction *functions;
    size_t function_count;
    size_t global_count;
} BytecodeProgram;

bool bytecode_compile(const IrProgram *ir, BytecodeProgram *program);
void bytecode_program_free(BytecodeProgram *program);
void bytecode_dump(FILE *output, const BytecodeProgram *program);
const char *opcode_name(OpCode op);

#endif
