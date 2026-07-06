#include "runtime/autograd.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    Tensor *tensor;
    bool expanded;
} VisitFrame;

static bool set_error(char *error, size_t size, const char *message) {
    if (size > 0) snprintf(error, size, "%s", message);
    return false;
}

static uint64_t next_visit_mark(void) {
    static uint64_t mark = 0;
    ++mark;
    if (mark == 0) ++mark;
    return mark;
}

GradNode *autograd_binary_node(GradOp op, Tensor *left, double left_scalar,
                               Tensor *right, double right_scalar) {
    GradNode *node = calloc(1, sizeof(*node));
    if (node == NULL) return NULL;
    node->op = op;
    node->left = left;
    node->right = right;
    node->left_scalar = left_scalar;
    node->right_scalar = right_scalar;
    tensor_retain(left);
    tensor_retain(right);
    return node;
}

GradNode *autograd_reduce_node(GradOp op, Tensor *input) {
    return autograd_binary_node(op, input, 0.0, NULL, 0.0);
}

void autograd_node_free(GradNode *node) {
    if (node == NULL) return;
    tensor_release(node->left);
    tensor_release(node->right);
    free(node);
}

static bool ensure_grad(Tensor *tensor) {
    if (tensor->grad != NULL) return true;
    tensor->grad = calloc((size_t)tensor->len, sizeof(*tensor->grad));
    return tensor->len == 0 || tensor->grad != NULL;
}

static bool append_tensor(Tensor ***items, size_t *count, size_t *capacity,
                          Tensor *tensor) {
    Tensor **grown;
    if (*count == *capacity) {
        size_t next = *capacity == 0 ? 16 : *capacity * 2;
        grown = realloc(*items, next * sizeof(*grown));
        if (grown == NULL) return false;
        *items = grown;
        *capacity = next;
    }
    (*items)[(*count)++] = tensor;
    return true;
}

static bool append_frame(VisitFrame **items, size_t *count, size_t *capacity,
                         Tensor *tensor, bool expanded) {
    VisitFrame *grown;
    if (*count == *capacity) {
        size_t next = *capacity == 0 ? 16 : *capacity * 2;
        grown = realloc(*items, next * sizeof(*grown));
        if (grown == NULL) return false;
        *items = grown;
        *capacity = next;
    }
    (*items)[(*count)++] = (VisitFrame){tensor, expanded};
    return true;
}

bool autograd_backward(Tensor *loss, char *error, size_t error_size) {
    VisitFrame *stack = NULL;
    Tensor **topo = NULL;
    size_t stack_count = 0, stack_capacity = 0;
    size_t topo_count = 0, topo_capacity = 0;
    size_t index;
    uint64_t visit_mark = next_visit_mark();
    bool ok = false;

    if (loss->len != 1) {
        char shape[128];
        tensor_format_shape(loss, shape, sizeof(shape));
        snprintf(error, error_size,
                 "backward() requires scalar tensor; got shape %s", shape);
        return false;
    }
    if (!loss->requires_grad || loss->grad_node == NULL)
        return set_error(error, error_size,
                         "backward() requires a tensor with a grad graph");
    if (!append_frame(&stack, &stack_count, &stack_capacity, loss, false))
        goto memory_error;
    while (stack_count > 0) {
        VisitFrame frame = stack[--stack_count];
        GradNode *node = frame.tensor->grad_node;
        if (frame.expanded) {
            if (!append_tensor(&topo, &topo_count, &topo_capacity, frame.tensor))
                goto memory_error;
            continue;
        }
        if (frame.tensor->visit_mark == visit_mark) continue;
        frame.tensor->visit_mark = visit_mark;
        if (!append_frame(&stack, &stack_count, &stack_capacity,
                          frame.tensor, true))
            goto memory_error;
        if (node != NULL) {
            if (node->right != NULL &&
                !append_frame(&stack, &stack_count, &stack_capacity,
                              node->right, false))
                goto memory_error;
            if (node->left != NULL &&
                !append_frame(&stack, &stack_count, &stack_capacity,
                              node->left, false))
                goto memory_error;
        }
    }
    for (index = 0; index < topo_count; ++index) {
        Tensor *tensor = topo[index];
        if (tensor->grad_node != NULL && ensure_grad(tensor))
            memset(tensor->grad, 0, (size_t)tensor->len * sizeof(*tensor->grad));
        else if (tensor->grad_node != NULL)
            goto memory_error;
    }
    if (!ensure_grad(loss)) goto memory_error;
    loss->grad[0] = 1.0;

    for (index = topo_count; index-- > 0;) {
        Tensor *output = topo[index];
        GradNode *node = output->grad_node;
        int64_t item;
        if (node == NULL) continue;
        if (node->left != NULL && !ensure_grad(node->left)) goto memory_error;
        if (node->right != NULL && !ensure_grad(node->right)) goto memory_error;
        for (item = 0; item < output->len; ++item) {
            double upstream = output->grad[item];
            double left_value = node->left == NULL
                                    ? node->left_scalar
                                    : node->left->data[item];
            double right_value = node->right == NULL
                                     ? node->right_scalar
                                     : node->right->data[item];
            if (node->op == GRAD_ADD) {
                if (node->left != NULL) node->left->grad[item] += upstream;
                if (node->right != NULL) node->right->grad[item] += upstream;
            } else if (node->op == GRAD_SUB) {
                if (node->left != NULL) node->left->grad[item] += upstream;
                if (node->right != NULL) node->right->grad[item] -= upstream;
            } else if (node->op == GRAD_MUL) {
                if (node->left != NULL)
                    node->left->grad[item] += upstream * right_value;
                if (node->right != NULL)
                    node->right->grad[item] += upstream * left_value;
            } else if (node->op == GRAD_DIV) {
                if (node->left != NULL)
                    node->left->grad[item] += upstream / right_value;
                if (node->right != NULL)
                    node->right->grad[item] -= upstream * left_value /
                                               (right_value * right_value);
            } else if (node->op == GRAD_SUM || node->op == GRAD_MEAN) {
                int64_t parent_item;
                double scale = node->op == GRAD_MEAN
                                   ? upstream / (double)node->left->len
                                   : upstream;
                for (parent_item = 0; parent_item < node->left->len;
                     ++parent_item)
                    node->left->grad[parent_item] += scale;
                break;
            }
        }
    }
    ok = true;
    goto cleanup;

memory_error:
    set_error(error, error_size, "Out of memory during backward()");
cleanup:
    free(stack);
    free(topo);
    return ok;
}
