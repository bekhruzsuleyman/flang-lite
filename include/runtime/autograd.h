#ifndef FLANG_RUNTIME_AUTOGRAD_H
#define FLANG_RUNTIME_AUTOGRAD_H

#include "runtime/tensor.h"

typedef enum {
    GRAD_ADD,
    GRAD_SUB,
    GRAD_MUL,
    GRAD_DIV,
    GRAD_SUM,
    GRAD_MEAN
} GradOp;

struct GradNode {
    GradOp op;
    Tensor *left;
    Tensor *right;
    double left_scalar;
    double right_scalar;
};

GradNode *autograd_binary_node(GradOp op, Tensor *left, double left_scalar,
                               Tensor *right, double right_scalar);
GradNode *autograd_reduce_node(GradOp op, Tensor *input);
void autograd_node_free(GradNode *node);
bool autograd_backward(Tensor *loss, char *error, size_t error_size);

#endif
