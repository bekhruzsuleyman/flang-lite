#ifndef FLANG_TOKEN_H
#define FLANG_TOKEN_H

#include <stddef.h>
#include <stdint.h>

typedef enum {
    TOKEN_EOF,
    TOKEN_NEWLINE,
    TOKEN_NUMBER,
    TOKEN_STRING,
    TOKEN_IDENTIFIER,
    TOKEN_PRINT,
    TOKEN_IF,
    TOKEN_ELSE,
    TOKEN_WHILE,
    TOKEN_TRUE,
    TOKEN_FALSE,
    TOKEN_AND,
    TOKEN_OR,
    TOKEN_NOT,
    TOKEN_FN,
    TOKEN_RETURN,
    TOKEN_PUB,
    TOKEN_IMPORT,
    TOKEN_FROM,
    TOKEN_AS,
    TOKEN_VOID,
    TOKEN_FOR,
    TOKEN_IN,
    TOKEN_EQUAL,
    TOKEN_EQUAL_EQUAL,
    TOKEN_BANG_EQUAL,
    TOKEN_LESS,
    TOKEN_LESS_EQUAL,
    TOKEN_GREATER,
    TOKEN_GREATER_EQUAL,
    TOKEN_COLON,
    TOKEN_PLUS,
    TOKEN_MINUS,
    TOKEN_ARROW,
    TOKEN_STAR,
    TOKEN_SLASH,
    TOKEN_LEFT_PAREN,
    TOKEN_RIGHT_PAREN,
    TOKEN_LEFT_BRACKET,
    TOKEN_RIGHT_BRACKET,
    TOKEN_DOT,
    TOKEN_COMMA,
    TOKEN_INDENT,
    TOKEN_DEDENT
} TokenType;

typedef struct {
    TokenType type;
    char *lexeme;
    int64_t number;
    size_t line;
    size_t column;
} Token;

typedef struct {
    Token *items;
    size_t count;
    size_t capacity;
} TokenArray;

const char *token_type_name(TokenType type);
void token_array_free(TokenArray *tokens);

#endif
