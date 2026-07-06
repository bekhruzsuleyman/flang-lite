#include "parser.h"

#include <stdarg.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    const TokenArray *tokens;
    size_t current;
    ParserError *error;
} Parser;

static const Token *peek(const Parser *parser) { return &parser->tokens->items[parser->current]; }
static const Token *previous(const Parser *parser) { return &parser->tokens->items[parser->current - 1]; }
static const Token *lookahead(const Parser *parser, size_t distance) {
    size_t index = parser->current + distance;
    if (index >= parser->tokens->count) index = parser->tokens->count - 1;
    return &parser->tokens->items[index];
}
static bool check(const Parser *parser, TokenType type) { return peek(parser)->type == type; }

static const Token *advance(Parser *parser) {
    const Token *token = peek(parser);
    if (token->type != TOKEN_EOF) ++parser->current;
    return token;
}

static bool match(Parser *parser, TokenType type) {
    if (!check(parser, type)) return false;
    advance(parser);
    return true;
}

static void parser_error_at(Parser *parser, const Token *token,
                            const char *format, ...) {
    va_list arguments;
    if (parser->error->has_error) return;
    parser->error->has_error = true;
    parser->error->line = token->line;
    parser->error->column = token->column;
    va_start(arguments, format);
    vsnprintf(parser->error->message, sizeof(parser->error->message), format, arguments);
    va_end(arguments);
}

static const Token *consume(Parser *parser, TokenType type, const char *message) {
    if (check(parser, type)) return advance(parser);
    parser_error_at(parser, peek(parser), "%s; found %s", message,
                    token_type_name(peek(parser)->type));
    return NULL;
}

static AstNode *out_of_memory(Parser *parser, const Token *token) {
    parser_error_at(parser, token, "Out of memory");
    return NULL;
}

static AstNode *parse_expression(Parser *parser);
static AstNode *parse_statement(Parser *parser);
static AstNode *parse_block(Parser *parser);

static AstNode *parse_primary(Parser *parser) {
    const Token *token = peek(parser);
    AstNode *node;

    if (match(parser, TOKEN_NUMBER)) {
        node = ast_new_node(AST_NUMBER, token->line, token->column);
        if (node == NULL) return out_of_memory(parser, token);
        node->as.number = token->number;
        return node;
    }
    if (match(parser, TOKEN_TRUE) || match(parser, TOKEN_FALSE)) {
        node = ast_new_node(AST_BOOL, token->line, token->column);
        if (node == NULL) return out_of_memory(parser, token);
        node->as.boolean = token->type == TOKEN_TRUE;
        return node;
    }
    if (match(parser, TOKEN_STRING)) {
        node = ast_new_node(AST_STRING, token->line, token->column);
        if (node == NULL) return out_of_memory(parser, token);
        node->as.string = ast_copy_string(token->lexeme);
        if (node->as.string == NULL) {
            ast_node_free(node);
            return out_of_memory(parser, token);
        }
        return node;
    }
    if (match(parser, TOKEN_LEFT_BRACKET)) {
        node = ast_new_node(AST_ARRAY, token->line, token->column);
        if (node == NULL) return out_of_memory(parser, token);
        if (!check(parser, TOKEN_RIGHT_BRACKET)) {
            do {
                AstNode *element = parse_expression(parser);
                if (element == NULL) {
                    ast_node_free(node);
                    return NULL;
                }
                if (!ast_array_append_element(node, element)) {
                    ast_node_free(element);
                    ast_node_free(node);
                    return out_of_memory(parser, peek(parser));
                }
            } while (match(parser, TOKEN_COMMA));
        }
        if (consume(parser, TOKEN_RIGHT_BRACKET,
                    "Expected ']' after array literal") == NULL) {
            ast_node_free(node);
            return NULL;
        }
        return node;
    }
    if (match(parser, TOKEN_IDENTIFIER)) {
        node = ast_new_node(AST_IDENTIFIER, token->line, token->column);
        if (node == NULL) return out_of_memory(parser, token);
        node->as.identifier = ast_copy_string(token->lexeme);
        if (node->as.identifier == NULL) {
            ast_node_free(node);
            return out_of_memory(parser, token);
        }
        return node;
    }
    if (match(parser, TOKEN_LEFT_PAREN)) {
        AstNode *expression = parse_expression(parser);
        if (expression == NULL) return NULL;
        if (consume(parser, TOKEN_RIGHT_PAREN, "Expected ')' after expression") == NULL) {
            ast_node_free(expression);
            return NULL;
        }
        return expression;
    }
    parser_error_at(parser, token, "Expected expression; found %s",
                    token_type_name(token->type));
    return NULL;
}

static AstNode *parse_postfix(Parser *parser) {
    AstNode *expression = parse_primary(parser);

    while (expression != NULL &&
           (check(parser, TOKEN_LEFT_PAREN) || check(parser, TOKEN_DOT) ||
            check(parser, TOKEN_LEFT_BRACKET))) {
        if (match(parser, TOKEN_LEFT_PAREN)) {
            const Token *left_paren = previous(parser);
            AstNode *call = ast_new_node(AST_CALL, left_paren->line,
                                         left_paren->column);
            if (call == NULL) {
                ast_node_free(expression);
                return out_of_memory(parser, left_paren);
            }
            call->as.call.callee = expression;
            if (!check(parser, TOKEN_RIGHT_PAREN)) {
                do {
                    AstNode *argument = parse_expression(parser);
                    if (argument == NULL) {
                        ast_node_free(call);
                        return NULL;
                    }
                    if (!ast_call_append_argument(call, argument)) {
                        ast_node_free(argument);
                        ast_node_free(call);
                        return out_of_memory(parser, peek(parser));
                    }
                } while (match(parser, TOKEN_COMMA));
            }
            if (consume(parser, TOKEN_RIGHT_PAREN,
                        "Expected ')' after arguments") == NULL) {
                ast_node_free(call);
                return NULL;
            }
            expression = call;
        } else if (check(parser, TOKEN_DOT)) {
            const Token *dot = advance(parser);
            const Token *member_name = consume(parser, TOKEN_IDENTIFIER,
                                                "Expected member name after '.'");
            AstNode *member;
            if (member_name == NULL) {
                ast_node_free(expression);
                return NULL;
            }
            member = ast_new_node(AST_MEMBER, dot->line, dot->column);
            if (member == NULL) {
                ast_node_free(expression);
                return out_of_memory(parser, dot);
            }
            member->as.member.object = expression;
            member->as.member.member = ast_copy_string(member_name->lexeme);
            if (member->as.member.member == NULL) {
                ast_node_free(member);
                return out_of_memory(parser, member_name);
            }
            expression = member;
        } else {
            const Token *left_bracket = advance(parser);
            AstNode *index_expression = parse_expression(parser);
            AstNode *index_node;
            if (index_expression == NULL) {
                ast_node_free(expression);
                return NULL;
            }
            if (consume(parser, TOKEN_RIGHT_BRACKET,
                        "Expected ']' after index") == NULL) {
                ast_node_free(expression);
                ast_node_free(index_expression);
                return NULL;
            }
            index_node = ast_new_node(AST_INDEX, left_bracket->line,
                                      left_bracket->column);
            if (index_node == NULL) {
                ast_node_free(expression);
                ast_node_free(index_expression);
                return out_of_memory(parser, left_bracket);
            }
            index_node->as.index.object = expression;
            index_node->as.index.index = index_expression;
            expression = index_node;
        }
    }
    return expression;
}

static AstNode *parse_unary(Parser *parser) {
    if (match(parser, TOKEN_MINUS) || match(parser, TOKEN_NOT)) {
        const Token *operator_token = previous(parser);
        AstNode *operand = parse_unary(parser);
        AstNode *node;
        if (operand == NULL) return NULL;
        node = ast_new_node(AST_UNARY, operator_token->line, operator_token->column);
        if (node == NULL) {
            ast_node_free(operand);
            return out_of_memory(parser, operator_token);
        }
        node->as.unary.operator_type = operator_token->type;
        node->as.unary.operand = operand;
        return node;
    }
    return parse_postfix(parser);
}

static AstNode *make_binary(Parser *parser, AstNode *left,
                            const Token *operator_token, AstNode *right) {
    AstNode *node = ast_new_node(AST_BINARY, operator_token->line,
                                 operator_token->column);
    if (node == NULL) {
        ast_node_free(left);
        ast_node_free(right);
        return out_of_memory(parser, operator_token);
    }
    node->as.binary.operator_type = operator_token->type;
    node->as.binary.left = left;
    node->as.binary.right = right;
    return node;
}

static AstNode *parse_factor(Parser *parser) {
    AstNode *left = parse_unary(parser);
    while (left != NULL && (check(parser, TOKEN_STAR) || check(parser, TOKEN_SLASH))) {
        const Token *operator_token = advance(parser);
        AstNode *right = parse_unary(parser);
        if (right == NULL) { ast_node_free(left); return NULL; }
        left = make_binary(parser, left, operator_token, right);
    }
    return left;
}

static AstNode *parse_term(Parser *parser) {
    AstNode *left = parse_factor(parser);
    while (left != NULL && (check(parser, TOKEN_PLUS) || check(parser, TOKEN_MINUS))) {
        const Token *operator_token = advance(parser);
        AstNode *right = parse_factor(parser);
        if (right == NULL) { ast_node_free(left); return NULL; }
        left = make_binary(parser, left, operator_token, right);
    }
    return left;
}

static AstNode *parse_comparison(Parser *parser) {
    AstNode *left = parse_term(parser);
    while (left != NULL &&
           (check(parser, TOKEN_LESS) || check(parser, TOKEN_LESS_EQUAL) ||
            check(parser, TOKEN_GREATER) || check(parser, TOKEN_GREATER_EQUAL))) {
        const Token *operator_token = advance(parser);
        AstNode *right = parse_term(parser);
        if (right == NULL) { ast_node_free(left); return NULL; }
        left = make_binary(parser, left, operator_token, right);
    }
    return left;
}

static AstNode *parse_equality(Parser *parser) {
    AstNode *left = parse_comparison(parser);
    while (left != NULL &&
           (check(parser, TOKEN_EQUAL_EQUAL) || check(parser, TOKEN_BANG_EQUAL))) {
        const Token *operator_token = advance(parser);
        AstNode *right = parse_comparison(parser);
        if (right == NULL) { ast_node_free(left); return NULL; }
        left = make_binary(parser, left, operator_token, right);
    }
    return left;
}

static AstNode *parse_and(Parser *parser) {
    AstNode *left = parse_equality(parser);
    while (left != NULL && check(parser, TOKEN_AND)) {
        const Token *operator_token = advance(parser);
        AstNode *right = parse_equality(parser);
        if (right == NULL) { ast_node_free(left); return NULL; }
        left = make_binary(parser, left, operator_token, right);
    }
    return left;
}

static AstNode *parse_or(Parser *parser) {
    AstNode *left = parse_and(parser);
    while (left != NULL && check(parser, TOKEN_OR)) {
        const Token *operator_token = advance(parser);
        AstNode *right = parse_and(parser);
        if (right == NULL) { ast_node_free(left); return NULL; }
        left = make_binary(parser, left, operator_token, right);
    }
    return left;
}

static AstNode *parse_expression(Parser *parser) { return parse_or(parser); }

static bool consume_statement_newline(Parser *parser) {
    return consume(parser, TOKEN_NEWLINE, "Expected newline after statement") != NULL;
}

static AstNode *parse_assignment(Parser *parser, bool is_public) {
    const Token *name = consume(parser, TOKEN_IDENTIFIER, "Expected variable name");
    const Token *annotation = NULL;
    AstNode *value;
    AstNode *node;

    if (name == NULL) return NULL;
    if (match(parser, TOKEN_COLON)) {
        annotation = consume(parser, TOKEN_IDENTIFIER, "Expected type name after ':'");
        if (annotation == NULL) return NULL;
    }
    if (consume(parser, TOKEN_EQUAL, "Expected '=' in assignment") == NULL) return NULL;
    value = parse_expression(parser);
    if (value == NULL) return NULL;
    if (!consume_statement_newline(parser)) { ast_node_free(value); return NULL; }

    node = ast_new_node(AST_ASSIGNMENT, name->line, name->column);
    if (node == NULL) { ast_node_free(value); return out_of_memory(parser, name); }
    node->as.assignment.name = ast_copy_string(name->lexeme);
    node->as.assignment.annotation = annotation == NULL ? NULL : ast_copy_string(annotation->lexeme);
    node->as.assignment.value = value;
    node->as.assignment.is_declaration = annotation != NULL;
    node->as.assignment.is_public = is_public;
    if (node->as.assignment.name == NULL ||
        (annotation != NULL && node->as.assignment.annotation == NULL)) {
        ast_node_free(node);
        return out_of_memory(parser, name);
    }
    return node;
}

static AstNode *parse_print(Parser *parser) {
    const Token *keyword = consume(parser, TOKEN_PRINT, "Expected 'print'");
    AstNode *expression;
    AstNode *node;
    if (keyword == NULL) return NULL;
    if (consume(parser, TOKEN_LEFT_PAREN, "Expected '(' after 'print'") == NULL) return NULL;
    expression = parse_expression(parser);
    if (expression == NULL) return NULL;
    if (consume(parser, TOKEN_RIGHT_PAREN, "Expected ')' after print expression") == NULL ||
        !consume_statement_newline(parser)) {
        ast_node_free(expression);
        return NULL;
    }
    node = ast_new_node(AST_PRINT, keyword->line, keyword->column);
    if (node == NULL) { ast_node_free(expression); return out_of_memory(parser, keyword); }
    node->as.print_statement.expression = expression;
    return node;
}

static bool is_valid_index_target(const AstNode *node) {
    if (node->kind != AST_INDEX) return false;
    while (node->kind == AST_INDEX) node = node->as.index.object;
    return node->kind == AST_IDENTIFIER;
}

static AstNode *parse_expression_statement(Parser *parser) {
    const Token *start = peek(parser);
    AstNode *expression = parse_expression(parser);
    AstNode *node;
    if (expression == NULL) return NULL;
    if (match(parser, TOKEN_EQUAL)) {
        AstNode *value;
        if (!is_valid_index_target(expression)) {
            parser_error_at(parser, previous(parser), "Invalid assignment target");
            ast_node_free(expression);
            return NULL;
        }
        value = parse_expression(parser);
        if (value == NULL) {
            ast_node_free(expression);
            return NULL;
        }
        if (!consume_statement_newline(parser)) {
            ast_node_free(expression);
            ast_node_free(value);
            return NULL;
        }
        node = ast_new_node(AST_INDEX_ASSIGNMENT, start->line, start->column);
        if (node == NULL) {
            ast_node_free(expression);
            ast_node_free(value);
            return out_of_memory(parser, start);
        }
        node->as.index_assignment.target = expression;
        node->as.index_assignment.value = value;
        return node;
    }
    if (!consume_statement_newline(parser)) { ast_node_free(expression); return NULL; }
    node = ast_new_node(AST_EXPRESSION_STATEMENT, start->line, start->column);
    if (node == NULL) { ast_node_free(expression); return out_of_memory(parser, start); }
    node->as.expression_statement.expression = expression;
    return node;
}

static bool parse_type(Parser *parser, TypeKind *type, const char *message) {
    const Token *token = peek(parser);
    if (match(parser, TOKEN_VOID)) {
        *type = TYPE_VOID;
        return true;
    }
    if (match(parser, TOKEN_IDENTIFIER)) {
        if (strcmp(token->lexeme, "int") == 0) {
            *type = TYPE_INT;
            return true;
        }
        if (strcmp(token->lexeme, "bool") == 0) {
            *type = TYPE_BOOL;
            return true;
        }
        if (strcmp(token->lexeme, "str") == 0) {
            *type = TYPE_STRING;
            return true;
        }
        if (strcmp(token->lexeme, "list") == 0) {
            *type = TYPE_LIST;
            return true;
        }
        if (strcmp(token->lexeme, "tensor") == 0) {
            *type = TYPE_TENSOR;
            return true;
        }
        parser_error_at(parser, token, "Unknown type '%s'", token->lexeme);
        return false;
    }
    parser_error_at(parser, token, "%s; found %s", message,
                    token_type_name(token->type));
    return false;
}

static AstNode *parse_function_declaration(Parser *parser, bool is_public) {
    const Token *keyword = consume(parser, TOKEN_FN, "Expected 'fn'");
    const Token *name;
    AstNode *function;
    AstNode *body;
    TypeKind return_type = TYPE_VOID;

    if (keyword == NULL) return NULL;
    name = consume(parser, TOKEN_IDENTIFIER, "Expected function name");
    if (name == NULL) return NULL;
    function = ast_new_node(AST_FUNCTION_DECLARATION, keyword->line,
                            keyword->column);
    if (function == NULL) return out_of_memory(parser, keyword);
    function->as.function_declaration.name = ast_copy_string(name->lexeme);
    function->as.function_declaration.is_public = is_public;
    if (function->as.function_declaration.name == NULL) {
        ast_node_free(function);
        return out_of_memory(parser, name);
    }
    if (consume(parser, TOKEN_LEFT_PAREN,
                "Expected '(' after function name") == NULL) {
        ast_node_free(function);
        return NULL;
    }
    if (!check(parser, TOKEN_RIGHT_PAREN)) {
        do {
            const Token *parameter_name = consume(parser, TOKEN_IDENTIFIER,
                                                   "Expected parameter name");
            TypeKind parameter_type;
            if (parameter_name == NULL ||
                consume(parser, TOKEN_COLON,
                        "Expected ':' after parameter name") == NULL ||
                !parse_type(parser, &parameter_type,
                            "Expected type after ':'")) {
                ast_node_free(function);
                return NULL;
            }
            if (parameter_type == TYPE_VOID) {
                parser_error_at(parser, parameter_name,
                                "Function parameter cannot have type void");
                ast_node_free(function);
                return NULL;
            }
            if (!ast_function_append_parameter(function, parameter_name->lexeme,
                                               parameter_type)) {
                ast_node_free(function);
                return out_of_memory(parser, parameter_name);
            }
        } while (match(parser, TOKEN_COMMA));
    }
    if (consume(parser, TOKEN_RIGHT_PAREN,
                "Expected ')' after parameter list") == NULL) {
        ast_node_free(function);
        return NULL;
    }
    if (match(parser, TOKEN_ARROW) &&
        !parse_type(parser, &return_type, "Expected return type after '->'")) {
        ast_node_free(function);
        return NULL;
    }
    function->as.function_declaration.return_type = return_type;
    if (consume(parser, TOKEN_COLON,
                "Expected ':' after function declaration") == NULL) {
        ast_node_free(function);
        return NULL;
    }
    body = parse_block(parser);
    if (body == NULL) {
        ast_node_free(function);
        return NULL;
    }
    function->as.function_declaration.body = body;
    return function;
}

static AstNode *parse_return(Parser *parser) {
    const Token *keyword = consume(parser, TOKEN_RETURN, "Expected 'return'");
    AstNode *node;
    AstNode *value = NULL;
    if (keyword == NULL) return NULL;
    if (!check(parser, TOKEN_NEWLINE)) {
        value = parse_expression(parser);
        if (value == NULL) return NULL;
    }
    if (!consume_statement_newline(parser)) {
        ast_node_free(value);
        return NULL;
    }
    node = ast_new_node(AST_RETURN, keyword->line, keyword->column);
    if (node == NULL) {
        ast_node_free(value);
        return out_of_memory(parser, keyword);
    }
    node->as.return_statement.value = value;
    return node;
}

static AstNode *parse_import(Parser *parser) {
    const Token *keyword = consume(parser, TOKEN_IMPORT, "Expected 'import'");
    const Token *module_name;
    const Token *alias = NULL;
    AstNode *node;
    if (keyword == NULL) return NULL;
    module_name = consume(parser, TOKEN_IDENTIFIER,
                          "Expected module name after import");
    if (module_name == NULL) return NULL;
    if (match(parser, TOKEN_AS)) {
        alias = consume(parser, TOKEN_IDENTIFIER,
                        "Expected alias name after 'as'");
        if (alias == NULL) return NULL;
    }
    if (!consume_statement_newline(parser)) return NULL;
    node = ast_new_node(AST_IMPORT, keyword->line, keyword->column);
    if (node == NULL) return out_of_memory(parser, keyword);
    node->as.import_statement.module_name = ast_copy_string(module_name->lexeme);
    node->as.import_statement.alias = alias == NULL ? NULL : ast_copy_string(alias->lexeme);
    if (node->as.import_statement.module_name == NULL ||
        (alias != NULL && node->as.import_statement.alias == NULL)) {
        ast_node_free(node);
        return out_of_memory(parser, module_name);
    }
    return node;
}

static AstNode *parse_from_import(Parser *parser) {
    const Token *keyword = consume(parser, TOKEN_FROM, "Expected 'from'");
    const Token *module_name;
    AstNode *node;
    if (keyword == NULL) return NULL;
    module_name = consume(parser, TOKEN_IDENTIFIER,
                          "Expected module name after 'from'");
    if (module_name == NULL ||
        consume(parser, TOKEN_IMPORT, "Expected 'import' after module name") == NULL)
        return NULL;
    node = ast_new_node(AST_FROM_IMPORT, keyword->line, keyword->column);
    if (node == NULL) return out_of_memory(parser, keyword);
    node->as.from_import.module_name = ast_copy_string(module_name->lexeme);
    if (node->as.from_import.module_name == NULL) {
        ast_node_free(node);
        return out_of_memory(parser, module_name);
    }
    do {
        const Token *name = consume(parser, TOKEN_IDENTIFIER,
                                    "Expected symbol name after import");
        if (name == NULL) {
            ast_node_free(node);
            return NULL;
        }
        if (!ast_from_import_append_name(node, name->lexeme)) {
            ast_node_free(node);
            return out_of_memory(parser, name);
        }
    } while (match(parser, TOKEN_COMMA));
    if (!consume_statement_newline(parser)) {
        ast_node_free(node);
        return NULL;
    }
    return node;
}

static AstNode *parse_block(Parser *parser) {
    const Token *newline = consume(parser, TOKEN_NEWLINE, "Expected newline after ':'");
    const Token *indent;
    AstNode *block;
    if (newline == NULL) return NULL;
    indent = consume(parser, TOKEN_INDENT, "Expected indented block after ':'");
    if (indent == NULL) return NULL;
    block = ast_new_node(AST_BLOCK, indent->line, indent->column);
    if (block == NULL) return out_of_memory(parser, indent);

    while (!check(parser, TOKEN_DEDENT) && !check(parser, TOKEN_EOF)) {
        AstNode *statement;
        while (match(parser, TOKEN_NEWLINE)) {}
        if (check(parser, TOKEN_DEDENT) || check(parser, TOKEN_EOF)) break;
        statement = parse_statement(parser);
        if (statement == NULL) { ast_node_free(block); return NULL; }
        if (!ast_block_append(block, statement)) {
            ast_node_free(statement);
            ast_node_free(block);
            return out_of_memory(parser, peek(parser));
        }
    }
    if (block->as.block.count == 0) {
        parser_error_at(parser, peek(parser), "Expected statement inside block");
        ast_node_free(block);
        return NULL;
    }
    if (consume(parser, TOKEN_DEDENT, "Expected dedent after block") == NULL) {
        ast_node_free(block);
        return NULL;
    }
    return block;
}

static AstNode *parse_if(Parser *parser) {
    const Token *keyword = consume(parser, TOKEN_IF, "Expected 'if'");
    AstNode *condition;
    AstNode *then_block;
    AstNode *else_block = NULL;
    AstNode *node;
    if (keyword == NULL) return NULL;
    condition = parse_expression(parser);
    if (condition == NULL) return NULL;
    if (consume(parser, TOKEN_COLON, "Expected ':' after if condition") == NULL) {
        ast_node_free(condition); return NULL;
    }
    then_block = parse_block(parser);
    if (then_block == NULL) { ast_node_free(condition); return NULL; }
    if (match(parser, TOKEN_ELSE)) {
        if (consume(parser, TOKEN_COLON, "Expected ':' after 'else'") == NULL) {
            ast_node_free(condition); ast_node_free(then_block); return NULL;
        }
        else_block = parse_block(parser);
        if (else_block == NULL) { ast_node_free(condition); ast_node_free(then_block); return NULL; }
    }
    node = ast_new_node(AST_IF, keyword->line, keyword->column);
    if (node == NULL) {
        ast_node_free(condition); ast_node_free(then_block); ast_node_free(else_block);
        return out_of_memory(parser, keyword);
    }
    node->as.if_statement.condition = condition;
    node->as.if_statement.then_block = then_block;
    node->as.if_statement.else_block = else_block;
    return node;
}

static AstNode *parse_while(Parser *parser) {
    const Token *keyword = consume(parser, TOKEN_WHILE, "Expected 'while'");
    AstNode *condition;
    AstNode *body;
    AstNode *node;
    if (keyword == NULL) return NULL;
    condition = parse_expression(parser);
    if (condition == NULL) return NULL;
    if (consume(parser, TOKEN_COLON, "Expected ':' after while condition") == NULL) {
        ast_node_free(condition); return NULL;
    }
    body = parse_block(parser);
    if (body == NULL) { ast_node_free(condition); return NULL; }
    node = ast_new_node(AST_WHILE, keyword->line, keyword->column);
    if (node == NULL) {
        ast_node_free(condition); ast_node_free(body); return out_of_memory(parser, keyword);
    }
    node->as.while_statement.condition = condition;
    node->as.while_statement.body = body;
    return node;
}

static AstNode *parse_for(Parser *parser) {
    const Token *keyword = consume(parser, TOKEN_FOR, "Expected 'for'");
    const Token *name;
    AstNode *iterable;
    AstNode *body;
    AstNode *node;
    if (keyword == NULL) return NULL;
    name = consume(parser, TOKEN_IDENTIFIER, "Expected loop variable after 'for'");
    if (name == NULL ||
        consume(parser, TOKEN_IN, "Expected 'in' after loop variable") == NULL)
        return NULL;
    iterable = parse_expression(parser);
    if (iterable == NULL) return NULL;
    if (consume(parser, TOKEN_COLON, "Expected ':' after for iterable") == NULL) {
        ast_node_free(iterable);
        return NULL;
    }
    body = parse_block(parser);
    if (body == NULL) {
        ast_node_free(iterable);
        return NULL;
    }
    node = ast_new_node(AST_FOR, keyword->line, keyword->column);
    if (node == NULL) {
        ast_node_free(iterable);
        ast_node_free(body);
        return out_of_memory(parser, keyword);
    }
    node->as.for_statement.name = ast_copy_string(name->lexeme);
    node->as.for_statement.iterable = iterable;
    node->as.for_statement.body = body;
    if (node->as.for_statement.name == NULL) {
        ast_node_free(node);
        return out_of_memory(parser, name);
    }
    return node;
}

static AstNode *parse_statement(Parser *parser) {
    if (match(parser, TOKEN_PUB)) {
        const Token *pub = previous(parser);
        if (check(parser, TOKEN_FN)) {
            return parse_function_declaration(parser, true);
        }
        if (check(parser, TOKEN_IDENTIFIER) &&
            lookahead(parser, 1)->type == TOKEN_COLON) {
            return parse_assignment(parser, true);
        }
        parser_error_at(parser, pub,
                        "'pub' can only be used with functions or typed variables");
        return NULL;
    }
    if (check(parser, TOKEN_FN)) return parse_function_declaration(parser, false);
    if (check(parser, TOKEN_RETURN)) return parse_return(parser);
    if (check(parser, TOKEN_IMPORT)) return parse_import(parser);
    if (check(parser, TOKEN_FROM)) return parse_from_import(parser);
    if (check(parser, TOKEN_IF)) return parse_if(parser);
    if (check(parser, TOKEN_WHILE)) return parse_while(parser);
    if (check(parser, TOKEN_FOR)) return parse_for(parser);
    if (check(parser, TOKEN_PRINT)) return parse_print(parser);
    if (check(parser, TOKEN_IDENTIFIER) &&
        (lookahead(parser, 1)->type == TOKEN_EQUAL ||
         lookahead(parser, 1)->type == TOKEN_COLON)) {
        return parse_assignment(parser, false);
    }
    if (check(parser, TOKEN_ELSE)) {
        parser_error_at(parser, peek(parser), "Unexpected 'else' without matching 'if'");
        return NULL;
    }
    return parse_expression_statement(parser);
}

bool parser_parse(const TokenArray *tokens, AstProgram *program,
                  ParserError *error) {
    Parser parser = {tokens, 0, error};
    *program = (AstProgram){0};
    *error = (ParserError){0};

    while (!check(&parser, TOKEN_EOF)) {
        AstNode *statement;
        while (match(&parser, TOKEN_NEWLINE)) {}
        if (check(&parser, TOKEN_EOF)) break;
        if (check(&parser, TOKEN_INDENT) || check(&parser, TOKEN_DEDENT)) {
            parser_error_at(&parser, peek(&parser), "Unexpected %s",
                            token_type_name(peek(&parser)->type));
            goto failed;
        }
        statement = parse_statement(&parser);
        if (statement == NULL) goto failed;
        if (!ast_program_append(program, statement)) {
            ast_node_free(statement);
            out_of_memory(&parser, peek(&parser));
            goto failed;
        }
    }
    return true;

failed:
    ast_program_free(program);
    return false;
}
