#include "token.h"

#include <stdlib.h>

const char *token_type_name(TokenType type) {
    switch (type) {
        case TOKEN_EOF: return "end of file";
        case TOKEN_NEWLINE: return "newline";
        case TOKEN_NUMBER: return "number";
        case TOKEN_STRING: return "string";
        case TOKEN_IDENTIFIER: return "identifier";
        case TOKEN_PRINT: return "print";
        case TOKEN_IF: return "if";
        case TOKEN_ELSE: return "else";
        case TOKEN_WHILE: return "while";
        case TOKEN_TRUE: return "true";
        case TOKEN_FALSE: return "false";
        case TOKEN_AND: return "and";
        case TOKEN_OR: return "or";
        case TOKEN_NOT: return "not";
        case TOKEN_FN: return "fn";
        case TOKEN_RETURN: return "return";
        case TOKEN_PUB: return "pub";
        case TOKEN_IMPORT: return "import";
        case TOKEN_FROM: return "from";
        case TOKEN_AS: return "as";
        case TOKEN_VOID: return "void";
        case TOKEN_FOR: return "for";
        case TOKEN_IN: return "in";
        case TOKEN_EQUAL: return "=";
        case TOKEN_EQUAL_EQUAL: return "==";
        case TOKEN_BANG_EQUAL: return "!=";
        case TOKEN_LESS: return "<";
        case TOKEN_LESS_EQUAL: return "<=";
        case TOKEN_GREATER: return ">";
        case TOKEN_GREATER_EQUAL: return ">=";
        case TOKEN_COLON: return ":";
        case TOKEN_PLUS: return "+";
        case TOKEN_MINUS: return "-";
        case TOKEN_ARROW: return "->";
        case TOKEN_STAR: return "*";
        case TOKEN_SLASH: return "/";
        case TOKEN_LEFT_PAREN: return "(";
        case TOKEN_RIGHT_PAREN: return ")";
        case TOKEN_LEFT_BRACKET: return "[";
        case TOKEN_RIGHT_BRACKET: return "]";
        case TOKEN_DOT: return ".";
        case TOKEN_COMMA: return ",";
        case TOKEN_INDENT: return "indent";
        case TOKEN_DEDENT: return "dedent";
    }
    return "unknown token";
}

void token_array_free(TokenArray *tokens) {
    size_t index;

    if (tokens == NULL) {
        return;
    }
    for (index = 0; index < tokens->count; ++index) {
        free(tokens->items[index].lexeme);
    }
    free(tokens->items);
    tokens->items = NULL;
    tokens->count = 0;
    tokens->capacity = 0;
}
