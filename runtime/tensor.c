#include "runtime/tensor.h"

#include "runtime/autograd.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>

static bool shape_length(const int64_t *shape, int rank, int64_t *length,
                         char *error, size_t error_size) {
    int dimension;
    int64_t count = 1;
    if (rank < 0) return false;
    for (dimension = 0; dimension < rank; ++dimension) {
        if (shape[dimension] < 0) {
            snprintf(error, error_size,
                     "Tensor shape dimensions must be non-negative ints");
            return false;
        }
        if (shape[dimension] != 0 && count > INT64_MAX / shape[dimension]) {
            snprintf(error, error_size, "Tensor is too large");
            return false;
        }
        count *= shape[dimension];
    }
    *length = count;
    return true;
}

Tensor *tensor_new(const int64_t *shape, int rank, bool requires_grad) {
    Tensor *tensor = calloc(1, sizeof(*tensor));
    int64_t length;
    char ignored[1];
    if (tensor == NULL || !shape_length(shape, rank, &length, ignored, 0)) {
        free(tensor);
        return NULL;
    }
    if (rank > 0) {
        tensor->shape = malloc((size_t)rank * sizeof(*tensor->shape));
        if (tensor->shape == NULL) {
            free(tensor);
            return NULL;
        }
        memcpy(tensor->shape, shape, (size_t)rank * sizeof(*tensor->shape));
    }
    if (length > 0) {
        tensor->data = malloc((size_t)length * sizeof(*tensor->data));
        if (tensor->data == NULL) {
            free(tensor->shape);
            free(tensor);
            return NULL;
        }
    }
    tensor->rank = rank;
    tensor->len = length;
    tensor->requires_grad = requires_grad;
    tensor->ref_count = 1;
    return tensor;
}

Tensor *tensor_filled(const int64_t *shape, int rank, double fill,
                      bool requires_grad, char *error, size_t error_size) {
    Tensor *tensor;
    int64_t length;
    int64_t index;
    if (!shape_length(shape, rank, &length, error, error_size)) return NULL;
    tensor = tensor_new(shape, rank, requires_grad);
    if (tensor == NULL) {
        snprintf(error, error_size, "Out of memory");
        return NULL;
    }
    for (index = 0; index < tensor->len; ++index) tensor->data[index] = fill;
    return tensor;
}

Tensor *tensor_arange(int64_t start, int64_t end, char *error,
                      size_t error_size) {
    int64_t shape[1];
    Tensor *tensor;
    int64_t index;
    if (end < start) {
        snprintf(error, error_size, "arange() end must be at least start");
        return NULL;
    }
    shape[0] = end - start;
    tensor = tensor_new(shape, 1, false);
    if (tensor == NULL) {
        snprintf(error, error_size, "Out of memory");
        return NULL;
    }
    for (index = 0; index < tensor->len; ++index)
        tensor->data[index] = (double)(start + index);
    return tensor;
}

void tensor_retain(Tensor *tensor) {
    if (tensor != NULL) ++tensor->ref_count;
}

void tensor_release(Tensor *tensor) {
    if (tensor == NULL || --tensor->ref_count > 0) return;
    autograd_node_free(tensor->grad_node);
    free(tensor->data);
    free(tensor->grad);
    free(tensor->shape);
    free(tensor);
}

bool tensor_same_shape(const Tensor *left, const Tensor *right) {
    int dimension;
    if (left->rank != right->rank) return false;
    for (dimension = 0; dimension < left->rank; ++dimension)
        if (left->shape[dimension] != right->shape[dimension]) return false;
    return true;
}

void tensor_format_shape(const Tensor *tensor, char *buffer, size_t size) {
    size_t used = 0;
    int dimension;
    if (size == 0) return;
    buffer[used++] = '[';
    for (dimension = 0; dimension < tensor->rank && used < size; ++dimension) {
        int written = snprintf(buffer + used, size - used, "%s%lld",
                               dimension == 0 ? "" : ", ",
                               (long long)tensor->shape[dimension]);
        if (written < 0 || (size_t)written >= size - used) {
            used = size - 1;
            break;
        }
        used += (size_t)written;
    }
    if (used + 1 < size) buffer[used++] = ']';
    buffer[used < size ? used : size - 1] = '\0';
}

Tensor *tensor_binary(TensorOp op, Tensor *left, double left_scalar,
                      Tensor *right, double right_scalar, char *error,
                      size_t error_size) {
    Tensor *source = left != NULL ? left : right;
    Tensor *output;
    int64_t index;
    bool requires_grad;
    GradOp grad_op;
    if (left != NULL && right != NULL && !tensor_same_shape(left, right)) {
        char left_shape[96], right_shape[96];
        tensor_format_shape(left, left_shape, sizeof(left_shape));
        tensor_format_shape(right, right_shape, sizeof(right_shape));
        snprintf(error, error_size, "Tensor shape mismatch: %s vs %s",
                 left_shape, right_shape);
        return NULL;
    }
    requires_grad = (left != NULL && left->requires_grad) ||
                    (right != NULL && right->requires_grad);
    output = tensor_new(source->shape, source->rank, requires_grad);
    if (output == NULL) {
        snprintf(error, error_size, "Out of memory");
        return NULL;
    }
    for (index = 0; index < source->len; ++index) {
        double a = left == NULL ? left_scalar : left->data[index];
        double b = right == NULL ? right_scalar : right->data[index];
        if (op == TENSOR_OP_DIV && b == 0.0) {
            tensor_release(output);
            snprintf(error, error_size, "Division by zero");
            return NULL;
        }
        if (op == TENSOR_OP_ADD) output->data[index] = a + b;
        else if (op == TENSOR_OP_SUB) output->data[index] = a - b;
        else if (op == TENSOR_OP_MUL) output->data[index] = a * b;
        else output->data[index] = a / b;
    }
    if (!requires_grad) return output;
    grad_op = op == TENSOR_OP_ADD ? GRAD_ADD
              : op == TENSOR_OP_SUB ? GRAD_SUB
              : op == TENSOR_OP_MUL ? GRAD_MUL
                                    : GRAD_DIV;
    output->grad_node = autograd_binary_node(grad_op, left, left_scalar,
                                             right, right_scalar);
    if (output->grad_node == NULL) {
        tensor_release(output);
        snprintf(error, error_size, "Out of memory");
        return NULL;
    }
    return output;
}

Tensor *tensor_reduce(Tensor *input, bool mean, char *error,
                      size_t error_size) {
    Tensor *output = tensor_new(NULL, 0, input->requires_grad);
    int64_t index;
    double total = 0.0;
    if (output == NULL) {
        snprintf(error, error_size, "Out of memory");
        return NULL;
    }
    if (mean && input->len == 0) {
        tensor_release(output);
        snprintf(error, error_size, "mean() cannot reduce an empty tensor");
        return NULL;
    }
    for (index = 0; index < input->len; ++index) total += input->data[index];
    output->data[0] = mean ? total / (double)input->len : total;
    if (input->requires_grad) {
        output->grad_node = autograd_reduce_node(mean ? GRAD_MEAN : GRAD_SUM,
                                                 input);
        if (output->grad_node == NULL) {
            tensor_release(output);
            snprintf(error, error_size, "Out of memory");
            return NULL;
        }
    }
    return output;
}

Tensor *tensor_gradient_copy(const Tensor *tensor, char *error,
                             size_t error_size) {
    Tensor *copy;
    if (tensor->grad == NULL) {
        snprintf(error, error_size,
                 "Tensor gradient is not available; call backward() first");
        return NULL;
    }
    copy = tensor_new(tensor->shape, tensor->rank, false);
    if (copy == NULL) {
        snprintf(error, error_size, "Out of memory");
        return NULL;
    }
    memcpy(copy->data, tensor->grad,
           (size_t)tensor->len * sizeof(*copy->data));
    return copy;
}

Tensor *tensor_copy_slice(const Tensor *source, int64_t index, char *error,
                          size_t error_size) {
    Tensor *result;
    int64_t slice_len;
    if (source->rank < 2 || index < 0 || index >= source->shape[0]) return NULL;
    result = tensor_new(source->shape + 1, source->rank - 1, false);
    if (result == NULL) {
        snprintf(error, error_size, "Out of memory");
        return NULL;
    }
    slice_len = source->len / source->shape[0];
    memcpy(result->data, source->data + index * slice_len,
           (size_t)slice_len * sizeof(*result->data));
    return result;
}

bool tensor_backward(Tensor *loss, char *error, size_t error_size) {
    return autograd_backward(loss, error, error_size);
}

void tensor_zero_grad(Tensor *tensor) {
    if (tensor->grad != NULL)
        memset(tensor->grad, 0, (size_t)tensor->len * sizeof(*tensor->grad));
}

static bool print_number(FILE *output, double value) {
    if (value == 0.0) value = 0.0;
    return fprintf(output, "%.15g", value) >= 0;
}

static bool print_dimension(FILE *output, const Tensor *tensor, int dimension,
                            int64_t *offset) {
    int64_t index;
    if (fputc('[', output) == EOF) return false;
    for (index = 0; index < tensor->shape[dimension]; ++index) {
        if (index > 0 && fputs(", ", output) == EOF) return false;
        if (dimension + 1 == tensor->rank) {
            if (!print_number(output, tensor->data[(*offset)++])) return false;
        } else if (!print_dimension(output, tensor, dimension + 1, offset)) {
            return false;
        }
    }
    return fputc(']', output) != EOF;
}

bool tensor_print(FILE *output, const Tensor *tensor) {
    int64_t offset = 0;
    if (tensor->rank == 0) return print_number(output, tensor->data[0]);
    return print_dimension(output, tensor, 0, &offset);
}
