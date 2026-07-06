#include "bytecode.h"

#include <stdlib.h>
#include <string.h>

static char *copy_text(const char *text) {
    size_t length = strlen(text);
    char *copy = malloc(length + 1);
    if (copy != NULL) memcpy(copy, text, length + 1);
    return copy;
}

bool bytecode_compile(const IrProgram *ir, BytecodeProgram *program) {
    size_t function, index;
    memset(program, 0, sizeof(*program));
    program->functions = calloc(ir->function_count, sizeof(*program->functions));
    if (program->functions == NULL && ir->function_count > 0) return false;
    program->function_count = ir->function_count;
    program->global_count = ir->global_count;
    for (function = 0; function < ir->function_count; ++function) {
        const IrFunction *source = &ir->functions[function];
        BytecodeFunction *target = &program->functions[function];
        target->name = copy_text(source->name);
        target->count = source->count;
        target->local_count = source->slot_count;
        target->parameter_count = source->parameter_count;
        target->code = calloc(source->count, sizeof(*target->code));
        if (target->name == NULL ||
            (target->code == NULL && source->count > 0)) {
            bytecode_program_free(program);
            return false;
        }
        for (index = 0; index < source->count; ++index) {
            const IrInstr *in = &source->code[index];
            BytecodeInstr *out = &target->code[index];
            out->op = (uint8_t)in->op;
            out->a = (int32_t)in->a;
            out->b = (int32_t)in->b;
            out->span = in->span;
            out->constant = value_null();
            if (in->op == IR_CONST &&
                !value_clone(&in->constant, &out->constant)) {
                bytecode_program_free(program);
                return false;
            }
        }
    }
    return true;
}

void bytecode_program_free(BytecodeProgram *program) {
    size_t function, index;
    for (function = 0; function < program->function_count; ++function) {
        BytecodeFunction *fn = &program->functions[function];
        for (index = 0; index < fn->count; ++index)
            if (fn->code[index].op == OP_CONST)
                value_free(&fn->code[index].constant);
        free(fn->code);
        free(fn->name);
    }
    free(program->functions);
    memset(program, 0, sizeof(*program));
}

const char *opcode_name(OpCode op) {
    static const char *names[] = {
        "OP_CONST", "OP_LOAD_LOCAL", "OP_STORE_LOCAL", "OP_LOAD_GLOBAL",
        "OP_STORE_GLOBAL", "OP_ADD_INT", "OP_SUB_INT", "OP_MUL_INT",
        "OP_DIV_INT", "OP_ADD_STR", "OP_EQ", "OP_NEQ", "OP_LT_INT",
        "OP_LTE_INT", "OP_GT_INT", "OP_GTE_INT", "OP_NEG_INT",
        "OP_NOT_BOOL", "OP_AND_BOOL", "OP_OR_BOOL", "OP_JUMP",
        "OP_JUMP_IF_FALSE", "OP_CALL", "OP_CALL_NATIVE", "OP_RETURN",
        "OP_PRINT", "OP_POP", "OP_BUILD_ARRAY", "OP_INDEX",
        "OP_GET_PROPERTY", "OP_CALL_METHOD", "OP_TENSOR_ADD",
        "OP_TENSOR_SUB", "OP_TENSOR_MUL", "OP_TENSOR_DIV", "OP_HALT",
        "OP_NOP"
    };
    return names[(int)op];
}

void bytecode_dump(FILE *output, const BytecodeProgram *program) {
    size_t function, index;
    for (function = 0; function < program->function_count; ++function) {
        const BytecodeFunction *fn = &program->functions[function];
        fprintf(output, "bytecode function %s (locals=%zu):\n",
                fn->name, fn->local_count);
        for (index = 0; index < fn->count; ++index) {
            const BytecodeInstr *in = &fn->code[index];
            fprintf(output, "  %04zu  %-24s", index,
                    opcode_name((OpCode)in->op));
            if (in->op == OP_CONST) {
                if (in->constant.kind == VALUE_INT)
                    fprintf(output, " %lld", (long long)in->constant.as.integer);
                else if (in->constant.kind == VALUE_BOOL)
                    fprintf(output, " %s", in->constant.as.boolean ? "true" : "false");
                else if (in->constant.kind == VALUE_STRING)
                    fprintf(output, " \"%s\"", in->constant.as.string.data);
                else fprintf(output, " null");
            } else if (in->a != 0 || in->op == OP_LOAD_LOCAL ||
                       in->op == OP_STORE_LOCAL || in->op == OP_LOAD_GLOBAL ||
                       in->op == OP_STORE_GLOBAL)
                fprintf(output, " %d", in->a);
            fprintf(output, "  ; %zu:%zu\n", in->span.line, in->span.column);
        }
    }
}
