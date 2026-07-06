#ifndef FLANG_LEXER_H
#define FLANG_LEXER_H

#include "token.h"

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    bool has_error;
    size_t line;
    size_t column;
    char message[256];
} LexerError;

bool lexer_scan(const char *source, TokenArray *tokens, LexerError *error);

#endif
