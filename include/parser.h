#ifndef FLANG_PARSER_H
#define FLANG_PARSER_H

#include "ast.h"
#include "token.h"

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    bool has_error;
    size_t line;
    size_t column;
    char message[256];
} ParserError;

bool parser_parse(const TokenArray *tokens, AstProgram *program,
                  ParserError *error);

#endif
