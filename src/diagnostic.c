#include "diagnostic.h"

#include "source.h"

#include <string.h>

const char *diagnostic_kind_name(DiagnosticKind kind) {
    switch (kind) {
        case DIAGNOSTIC_LEXER: return "lexer";
        case DIAGNOSTIC_PARSER: return "parser";
        case DIAGNOSTIC_RESOLVE: return "resolve";
        case DIAGNOSTIC_TYPE: return "type";
        case DIAGNOSTIC_RUNTIME: return "runtime";
    }
    return "unknown";
}

void diagnostic_set(Diagnostic *diagnostic, DiagnosticKind kind,
                    const char *path, size_t line, size_t column,
                    const char *message) {
    diagnostic->has_error = true;
    diagnostic->kind = kind;
    diagnostic->line = line;
    diagnostic->column = column;
    diagnostic->length = 1;
    diagnostic->source_line[0] = '\0';
    diagnostic->hint[0] = '\0';
    snprintf(diagnostic->path, sizeof(diagnostic->path), "%s", path);
    snprintf(diagnostic->message, sizeof(diagnostic->message), "%s", message);
}

void diagnostic_attach_source(Diagnostic *diagnostic, const char *source,
                              size_t length, const char *hint) {
    diagnostic->length = length == 0 ? 1 : length;
    source_get_line(source, diagnostic->line, diagnostic->source_line,
                    sizeof(diagnostic->source_line));
    if (hint != NULL)
        snprintf(diagnostic->hint, sizeof(diagnostic->hint), "%s", hint);
}

void diagnostic_print(FILE *output, const Diagnostic *diagnostic) {
    fprintf(output, "%s:%zu:%zu: %s error: %s\n", diagnostic->path,
            diagnostic->line, diagnostic->column,
            diagnostic_kind_name(diagnostic->kind), diagnostic->message);
    if (diagnostic->source_line[0] != '\0') {
        size_t index;
        size_t caret_count = diagnostic->length;
        fprintf(output, "\n    %s\n    ", diagnostic->source_line);
        for (index = 1; index < diagnostic->column; ++index) fputc(' ', output);
        if (caret_count > 64) caret_count = 64;
        for (index = 0; index < caret_count; ++index) fputc('^', output);
        fputc('\n', output);
    }
    if (diagnostic->hint[0] != '\0')
        fprintf(output, "\nhint: %s\n", diagnostic->hint);
}
