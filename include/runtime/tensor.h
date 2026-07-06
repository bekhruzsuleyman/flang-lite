#ifndef FLANG_RUNTIME_TENSOR_H
#define FLANG_RUNTIME_TENSOR_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

typedef struct GradNode GradNode;

typedef struct Tensor {
    double *data;
    double *grad;
    int64_t *shape;
    int rank;
    int64_t len;
    bool requires_grad;
    GradNode *grad_node;
    int ref_count;
    uint64_t visit_mark;
} Tensor;

typedef enum {
    TENSOR_OP_ADD,
    TENSOR_OP_SUB,
    TENSOR_OP_MUL,
    TENSOR_OP_DIV
} TensorOp;

Tensor *tensor_new(const int64_t *shape, int rank, bool requires_grad);
Tensor *tensor_filled(const int64_t *shape, int rank, double fill,
                      bool requires_grad, char *error, size_t error_size);
Tensor *tensor_arange(int64_t start, int64_t end, char *error,
                      size_t error_size);
Tensor *tensor_copy_slice(const Tensor *source, int64_t index, char *error,
                          size_t error_size);
Tensor *tensor_binary(TensorOp op, Tensor *left, double left_scalar,
                      Tensor *right, double right_scalar, char *error,
                      size_t error_size);
Tensor *tensor_reduce(Tensor *input, bool mean, char *error,
                      size_t error_size);
Tensor *tensor_gradient_copy(const Tensor *tensor, char *error,
                             size_t error_size);
bool tensor_backward(Tensor *loss, char *error, size_t error_size);
void tensor_zero_grad(Tensor *tensor);
void tensor_retain(Tensor *tensor);
void tensor_release(Tensor *tensor);
bool tensor_same_shape(const Tensor *left, const Tensor *right);
void tensor_format_shape(const Tensor *tensor, char *buffer, size_t size);
bool tensor_print(FILE *output, const Tensor *tensor);

#endif
