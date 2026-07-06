#include "optimizer.h"

#include <limits.h>
#include <stdlib.h>

static bool fold_int(IrOp op, int64_t left, int64_t right, Value *result) {
    if (op == IR_ADD_INT) {
        if ((right > 0 && left > INT64_MAX - right) ||
            (right < 0 && left < INT64_MIN - right)) return false;
        *result = value_int(left + right);
    } else if (op == IR_SUB_INT) {
        if ((right > 0 && left < INT64_MIN + right) ||
            (right < 0 && left > INT64_MAX + right)) return false;
        *result = value_int(left - right);
    } else if (op == IR_MUL_INT) {
        if (left != 0 && right != 0 &&
            (left == INT64_MIN ? right != 1
             : right == INT64_MIN ? left != 1
             : left > 0 ? (right > 0 ? left > INT64_MAX / right
                                      : right < INT64_MIN / left)
                        : (right > 0 ? left < INT64_MIN / right
                                     : left < INT64_MAX / right)))
            return false;
        *result = value_int(left * right);
    } else if (op == IR_DIV_INT) {
        if (right == 0 || (left == INT64_MIN && right == -1)) return false;
        *result = value_int(left / right);
    } else if (op == IR_EQ) *result = value_bool(left == right);
    else if (op == IR_NEQ) *result = value_bool(left != right);
    else if (op == IR_LT_INT) *result = value_bool(left < right);
    else if (op == IR_LTE_INT) *result = value_bool(left <= right);
    else if (op == IR_GT_INT) *result = value_bool(left > right);
    else if (op == IR_GTE_INT) *result = value_bool(left >= right);
    else return false;
    return true;
}

static void optimize_function(IrFunction *function) {
    size_t index;
    bool changed;
    do {
        changed = false;
        for (index = 2; index < function->count; ++index) {
            size_t right_index = index;
            size_t left_index;
            IrInstr *left;
            IrInstr *right;
            IrInstr *op = &function->code[index];
            Value folded = value_null();
            while (right_index > 0 &&
                   function->code[right_index - 1].op == IR_NOP)
                --right_index;
            if (right_index == 0) continue;
            --right_index;
            left_index = right_index;
            while (left_index > 0 &&
                   function->code[left_index - 1].op == IR_NOP)
                --left_index;
            if (left_index == 0) continue;
            --left_index;
            left = &function->code[left_index];
            right = &function->code[right_index];
            if (left->op == IR_CONST && right->op == IR_CONST &&
            left->constant.kind == VALUE_INT &&
            right->constant.kind == VALUE_INT &&
            fold_int(op->op, left->constant.as.integer,
                     right->constant.as.integer, &folded)) {
            value_free(&left->constant);
            value_free(&right->constant);
            left->op = IR_NOP;
            right->op = IR_NOP;
            op->op = IR_CONST;
            op->constant = folded;
                changed = true;
            } else if (left->op == IR_CONST && right->op == IR_CONST &&
                   left->constant.kind == VALUE_BOOL &&
                   right->constant.kind == VALUE_BOOL &&
                   (op->op == IR_AND_BOOL || op->op == IR_OR_BOOL)) {
            bool value = op->op == IR_AND_BOOL
                             ? left->constant.as.boolean && right->constant.as.boolean
                             : left->constant.as.boolean || right->constant.as.boolean;
            value_free(&left->constant);
            value_free(&right->constant);
            left->op = IR_NOP;
            right->op = IR_NOP;
            op->op = IR_CONST;
            op->constant = value_bool(value);
                changed = true;
            }
        }
    } while (changed);
    for (index = 0; index < function->count; ++index) {
        IrInstr *instruction = &function->code[index];
        if (instruction->op == IR_JUMP && instruction->a == (int)index + 1)
            instruction->op = IR_NOP;
        if (instruction->op == IR_NOT_BOOL && index > 0 &&
            function->code[index - 1].op == IR_CONST &&
            function->code[index - 1].constant.kind == VALUE_BOOL) {
            function->code[index - 1].constant.as.boolean =
                !function->code[index - 1].constant.as.boolean;
            instruction->op = IR_NOP;
        }
    }
    {
        bool has_jumps = false;
        for (index = 0; index < function->count; ++index)
            if (function->code[index].op == IR_JUMP ||
                function->code[index].op == IR_JUMP_IF_FALSE)
                has_jumps = true;
        if (!has_jumps) {
            bool after_return = false;
            for (index = 0; index < function->count; ++index) {
                if (after_return) {
                    if (function->code[index].op == IR_CONST)
                        value_free(&function->code[index].constant);
                    function->code[index].op = IR_NOP;
                } else if (function->code[index].op == IR_RETURN) {
                    after_return = true;
                }
            }
        }
    }
    {
        size_t old_count = function->count;
        size_t next = 0;
        size_t *map = malloc((old_count + 1) * sizeof(*map));
        if (map == NULL) return;
        for (index = 0; index < old_count; ++index) {
            map[index] = next;
            if (function->code[index].op != IR_NOP) ++next;
        }
        map[old_count] = next;
        next = 0;
        for (index = 0; index < old_count; ++index) {
            IrInstr instruction = function->code[index];
            if (instruction.op == IR_NOP) continue;
            if (instruction.op == IR_JUMP ||
                instruction.op == IR_JUMP_IF_FALSE)
                instruction.a = (int)map[(size_t)instruction.a];
            function->code[next++] = instruction;
        }
        function->count = next;
        free(map);
    }
}

void optimizer_run(IrProgram *program) {
    size_t index;
    for (index = 0; index < program->function_count; ++index)
        optimize_function(&program->functions[index]);
}
