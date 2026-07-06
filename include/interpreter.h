#ifndef FLANG_INTERPRETER_H
#define FLANG_INTERPRETER_H

#include "ast.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

typedef struct {
    bool has_error;
    size_t line;
    size_t column;
    char path[1024];
    char message[256];
    char hint[256];
} RuntimeError;

bool interpreter_run(const AstProgram *program, const char *entry_path, FILE *output,
                     RuntimeError *error);

#endif
