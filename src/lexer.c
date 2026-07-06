#include "lexer.h"

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_INDENT_DEPTH 128

typedef struct {
    const char *source;
    size_t current;
    size_t line;
    size_t column;
    int indent_stack[MAX_INDENT_DEPTH];
    size_t indent_top;
    bool at_line_start;
    size_t grouping_depth;
    TokenArray *tokens;
    LexerError *error;
} Lexer;

static char *copy_lexeme(const char *start, size_t length) {
    char *copy = malloc(length + 1);
    if (copy == NULL) return NULL;
    memcpy(copy, start, length);
    copy[length] = '\0';
    return copy;
}

static bool fail(Lexer *lexer, size_t line, size_t column,
                 const char *message) {
    lexer->error->has_error = true;
    lexer->error->line = line;
    lexer->error->column = column;
    snprintf(lexer->error->message, sizeof(lexer->error->message), "%s", message);
    return false;
}

static bool append_token(Lexer *lexer, TokenType type, const char *start,
                         size_t length, int64_t number, size_t line,
                         size_t column) {
    Token token;
    Token *new_items;
    size_t new_capacity;

    if (lexer->tokens->count == lexer->tokens->capacity) {
        new_capacity = lexer->tokens->capacity == 0 ? 32 : lexer->tokens->capacity * 2;
        new_items = realloc(lexer->tokens->items, new_capacity * sizeof(*new_items));
        if (new_items == NULL) return fail(lexer, line, column, "Out of memory");
        lexer->tokens->items = new_items;
        lexer->tokens->capacity = new_capacity;
    }
    token.type = type;
    token.lexeme = copy_lexeme(start, length);
    if (token.lexeme == NULL) return fail(lexer, line, column, "Out of memory");
    token.number = number;
    token.line = line;
    token.column = column;
    lexer->tokens->items[lexer->tokens->count++] = token;
    return true;
}

static void consume_linebreak(Lexer *lexer) {
    if (lexer->source[lexer->current] == '\r' &&
        lexer->source[lexer->current + 1] == '\n') {
        lexer->current += 2;
    } else {
        ++lexer->current;
    }
    ++lexer->line;
    lexer->column = 1;
    lexer->at_line_start = true;
}

static bool handle_line_start(Lexer *lexer) {
    size_t start = lexer->current;
    size_t spaces = 0;

    while (lexer->source[lexer->current] == ' ') {
        ++lexer->current;
        ++lexer->column;
        ++spaces;
    }
    if (lexer->source[lexer->current] == '\t') {
        return fail(lexer, lexer->line, lexer->column,
                    "Tabs are not allowed for indentation; use spaces only");
    }

    if (lexer->source[lexer->current] == '#') {
        while (lexer->source[lexer->current] != '\0' &&
               lexer->source[lexer->current] != '\n' &&
               lexer->source[lexer->current] != '\r') {
            ++lexer->current;
            ++lexer->column;
        }
        if (lexer->source[lexer->current] != '\0') consume_linebreak(lexer);
        return true;
    }
    if (lexer->source[lexer->current] == '\n' ||
        lexer->source[lexer->current] == '\r') {
        consume_linebreak(lexer);
        return true;
    }
    if (lexer->source[lexer->current] == '\0') return true;

    if (lexer->grouping_depth > 0) {
        lexer->at_line_start = false;
        return true;
    }

    if ((int)spaces > lexer->indent_stack[lexer->indent_top]) {
        if (lexer->indent_top + 1 >= MAX_INDENT_DEPTH) {
            return fail(lexer, lexer->line, 1, "Indentation nesting is too deep");
        }
        lexer->indent_stack[++lexer->indent_top] = (int)spaces;
        if (!append_token(lexer, TOKEN_INDENT, lexer->source + start, spaces,
                          0, lexer->line, 1)) {
            return false;
        }
    } else if ((int)spaces < lexer->indent_stack[lexer->indent_top]) {
        while (lexer->indent_top > 0 &&
               (int)spaces < lexer->indent_stack[lexer->indent_top]) {
            --lexer->indent_top;
            if (!append_token(lexer, TOKEN_DEDENT, lexer->source + start, 0,
                              0, lexer->line, 1)) {
                return false;
            }
        }
        if ((int)spaces != lexer->indent_stack[lexer->indent_top]) {
            return fail(lexer, lexer->line, 1, "Invalid indentation level");
        }
    }
    lexer->at_line_start = false;
    return true;
}

static TokenType identifier_type(const char *start, size_t length) {
    struct Keyword {
        const char *text;
        TokenType type;
    };
    static const struct Keyword keywords[] = {
        {"print", TOKEN_PRINT}, {"if", TOKEN_IF},       {"else", TOKEN_ELSE},
        {"while", TOKEN_WHILE}, {"true", TOKEN_TRUE}, {"false", TOKEN_FALSE},
        {"and", TOKEN_AND},     {"or", TOKEN_OR},     {"not", TOKEN_NOT},
        {"fn", TOKEN_FN},       {"return", TOKEN_RETURN}, {"pub", TOKEN_PUB},
        {"import", TOKEN_IMPORT}, {"from", TOKEN_FROM}, {"as", TOKEN_AS},
        {"void", TOKEN_VOID},
        {"for", TOKEN_FOR},     {"in", TOKEN_IN},
    };
    size_t index;

    for (index = 0; index < sizeof(keywords) / sizeof(keywords[0]); ++index) {
        if (strlen(keywords[index].text) == length &&
            strncmp(start, keywords[index].text, length) == 0) {
            return keywords[index].type;
        }
    }
    return TOKEN_IDENTIFIER;
}

static bool scan_number(Lexer *lexer, size_t start, size_t line,
                        size_t column) {
    char *lexeme;
    char *end;
    intmax_t value;

    while (isdigit((unsigned char)lexer->source[lexer->current])) {
        ++lexer->current;
        ++lexer->column;
    }
    lexeme = copy_lexeme(lexer->source + start, lexer->current - start);
    if (lexeme == NULL) return fail(lexer, line, column, "Out of memory");
    errno = 0;
    value = strtoimax(lexeme, &end, 10);
    if (errno == ERANGE || *end != '\0' || value < INT64_MIN || value > INT64_MAX) {
        free(lexeme);
        return fail(lexer, line, column, "Integer literal is out of range");
    }
    free(lexeme);
    return append_token(lexer, TOKEN_NUMBER, lexer->source + start,
                        lexer->current - start, (int64_t)value, line, column);
}

static bool scan_identifier(Lexer *lexer, size_t start, size_t line,
                            size_t column) {
    size_t length;

    while (isalnum((unsigned char)lexer->source[lexer->current]) ||
           lexer->source[lexer->current] == '_') {
        ++lexer->current;
        ++lexer->column;
    }
    length = lexer->current - start;
    return append_token(lexer, identifier_type(lexer->source + start, length),
                        lexer->source + start, length, 0, line, column);
}

static bool scan_string(Lexer *lexer, size_t line, size_t column) {
    char *buffer = NULL;
    size_t count = 0;
    size_t capacity = 0;

    while (lexer->source[lexer->current] != '"') {
        char value;
        char *new_buffer;
        size_t new_capacity;
        char character = lexer->source[lexer->current++];
        ++lexer->column;
        if (character == '\0' || character == '\n' || character == '\r') {
            free(buffer);
            return fail(lexer, line, column, "Unterminated string literal");
        }
        if (character == '\\') {
            char escaped = lexer->source[lexer->current];
            if (escaped == '\0' || escaped == '\n' || escaped == '\r') {
                free(buffer);
                return fail(lexer, line, column, "Unterminated string literal");
            }
            ++lexer->current;
            ++lexer->column;
            switch (escaped) {
                case 'n': value = '\n'; break;
                case 't': value = '\t'; break;
                case '"': value = '"'; break;
                case '\\': value = '\\'; break;
                default: {
                    char message[256];
                    free(buffer);
                    snprintf(message, sizeof(message),
                             "Invalid escape sequence '\\%c'", escaped);
                    return fail(lexer, line, column, message);
                }
            }
        } else {
            value = character;
        }
        if (count == capacity) {
            new_capacity = capacity == 0 ? 16 : capacity * 2;
            new_buffer = realloc(buffer, new_capacity);
            if (new_buffer == NULL) {
                free(buffer);
                return fail(lexer, line, column, "Out of memory");
            }
            buffer = new_buffer;
            capacity = new_capacity;
        }
        buffer[count++] = value;
    }
    ++lexer->current;
    ++lexer->column;
    if (!append_token(lexer, TOKEN_STRING, buffer == NULL ? "" : buffer,
                      count, 0, line, column)) {
        free(buffer);
        return false;
    }
    free(buffer);
    return true;
}

bool lexer_scan(const char *source, TokenArray *tokens, LexerError *error) {
    Lexer lexer = {0};

    *tokens = (TokenArray){0};
    *error = (LexerError){0};
    lexer.source = source;
    lexer.line = 1;
    lexer.column = 1;
    lexer.at_line_start = true;
    lexer.tokens = tokens;
    lexer.error = error;

    while (source[lexer.current] != '\0') {
        size_t start;
        size_t line;
        size_t column;
        char character;

        if (lexer.at_line_start) {
            if (!handle_line_start(&lexer)) goto failed;
            if (lexer.at_line_start || source[lexer.current] == '\0') continue;
        }

        start = lexer.current;
        line = lexer.line;
        column = lexer.column;
        character = source[lexer.current++];
        ++lexer.column;

        switch (character) {
            case ' ':
            case '\t':
                break;
            case '\r':
            case '\n':
                lexer.current = start;
                if (lexer.grouping_depth == 0 &&
                    !append_token(&lexer, TOKEN_NEWLINE, source + start,
                                  character == '\r' && source[start + 1] == '\n' ? 2 : 1,
                                  0, line, column)) goto failed;
                consume_linebreak(&lexer);
                break;
            case '#':
                while (source[lexer.current] != '\0' &&
                       source[lexer.current] != '\n' && source[lexer.current] != '\r') {
                    ++lexer.current;
                    ++lexer.column;
                }
                break;
            case '=':
                if (source[lexer.current] == '=') {
                    ++lexer.current; ++lexer.column;
                    if (!append_token(&lexer, TOKEN_EQUAL_EQUAL, source + start, 2, 0, line, column)) goto failed;
                } else if (!append_token(&lexer, TOKEN_EQUAL, source + start, 1, 0, line, column)) goto failed;
                break;
            case '!':
                if (source[lexer.current] != '=') {
                    fail(&lexer, line, column, "Unexpected character '!'"); goto failed;
                }
                ++lexer.current; ++lexer.column;
                if (!append_token(&lexer, TOKEN_BANG_EQUAL, source + start, 2, 0, line, column)) goto failed;
                break;
            case '<':
                if (source[lexer.current] == '=') {
                    ++lexer.current; ++lexer.column;
                    if (!append_token(&lexer, TOKEN_LESS_EQUAL, source + start, 2, 0, line, column)) goto failed;
                } else if (!append_token(&lexer, TOKEN_LESS, source + start, 1, 0, line, column)) goto failed;
                break;
            case '>':
                if (source[lexer.current] == '=') {
                    ++lexer.current; ++lexer.column;
                    if (!append_token(&lexer, TOKEN_GREATER_EQUAL, source + start, 2, 0, line, column)) goto failed;
                } else if (!append_token(&lexer, TOKEN_GREATER, source + start, 1, 0, line, column)) goto failed;
                break;
            case ':': if (!append_token(&lexer, TOKEN_COLON, source + start, 1, 0, line, column)) goto failed; break;
            case '+': if (!append_token(&lexer, TOKEN_PLUS, source + start, 1, 0, line, column)) goto failed; break;
            case '-':
                if (source[lexer.current] == '>') {
                    ++lexer.current; ++lexer.column;
                    if (!append_token(&lexer, TOKEN_ARROW, source + start, 2, 0, line, column)) goto failed;
                } else if (!append_token(&lexer, TOKEN_MINUS, source + start, 1, 0, line, column)) goto failed;
                break;
            case '*': if (!append_token(&lexer, TOKEN_STAR, source + start, 1, 0, line, column)) goto failed; break;
            case '/': if (!append_token(&lexer, TOKEN_SLASH, source + start, 1, 0, line, column)) goto failed; break;
            case '(':
                ++lexer.grouping_depth;
                if (!append_token(&lexer, TOKEN_LEFT_PAREN, source + start, 1, 0, line, column)) goto failed;
                break;
            case ')':
                if (lexer.grouping_depth > 0) --lexer.grouping_depth;
                if (!append_token(&lexer, TOKEN_RIGHT_PAREN, source + start, 1, 0, line, column)) goto failed;
                break;
            case '[':
                ++lexer.grouping_depth;
                if (!append_token(&lexer, TOKEN_LEFT_BRACKET, source + start, 1, 0, line, column)) goto failed;
                break;
            case ']':
                if (lexer.grouping_depth > 0) --lexer.grouping_depth;
                if (!append_token(&lexer, TOKEN_RIGHT_BRACKET, source + start, 1, 0, line, column)) goto failed;
                break;
            case '.': if (!append_token(&lexer, TOKEN_DOT, source + start, 1, 0, line, column)) goto failed; break;
            case ',': if (!append_token(&lexer, TOKEN_COMMA, source + start, 1, 0, line, column)) goto failed; break;
            case '"': if (!scan_string(&lexer, line, column)) goto failed; break;
            default:
                if (isdigit((unsigned char)character)) {
                    if (!scan_number(&lexer, start, line, column)) goto failed;
                } else if (isalpha((unsigned char)character) || character == '_') {
                    if (!scan_identifier(&lexer, start, line, column)) goto failed;
                } else {
                    char message[256];
                    snprintf(message, sizeof(message), "Unexpected character '%c'", character);
                    fail(&lexer, line, column, message); goto failed;
                }
        }
    }

    if (tokens->count > 0 && tokens->items[tokens->count - 1].type != TOKEN_NEWLINE) {
        if (!append_token(&lexer, TOKEN_NEWLINE, source + lexer.current, 0, 0,
                          lexer.line, lexer.column)) goto failed;
    }
    while (lexer.indent_top > 0) {
        --lexer.indent_top;
        if (!append_token(&lexer, TOKEN_DEDENT, source + lexer.current, 0, 0,
                          lexer.line, lexer.column)) goto failed;
    }
    if (!append_token(&lexer, TOKEN_EOF, source + lexer.current, 0, 0,
                      lexer.line, lexer.column)) goto failed;
    return true;

failed:
    token_array_free(tokens);
    return false;
}
