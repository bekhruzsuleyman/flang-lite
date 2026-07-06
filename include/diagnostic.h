#ifndef FLANG_DIAGNOSTIC_H
#define FLANG_DIAGNOSTIC_H

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>

typedef enum {
    DIAGNOSTIC_LEXER,
    DIAGNOSTIC_PARSER,
    DIAGNOSTIC_RESOLVE,
    DIAGNOSTIC_TYPE,
    DIAGNOSTIC_RUNTIME
} DiagnosticKind;

typedef struct {
    bool has_error;
    DiagnosticKind kind;
    char path[1024];
    size_t line;
    size_t column;
    size_t length;
    char message[256];
    char source_line[512];
    char hint[256];
} Diagnostic;

const char *diagnostic_kind_name(DiagnosticKind kind);
void diagnostic_set(Diagnostic *diagnostic, DiagnosticKind kind,
                    const char *path, size_t line, size_t column,
                    const char *message);
void diagnostic_attach_source(Diagnostic *diagnostic, const char *source,
                              size_t length, const char *hint);
void diagnostic_print(FILE *output, const Diagnostic *diagnostic);

#endif
