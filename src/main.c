#include "ast.h"
#include "diagnostic.h"
#include "bytecode.h"
#include "interpreter.h"
#include "ir.h"
#include "lexer.h"
#include "parser.h"
#include "token.h"
#include "resolver.h"
#include "typecheck.h"
#include "optimizer.h"
#include "vm.h"

#include <stdbool.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char *read_source(const char *path) {
    FILE *file = fopen(path, "rb");
    long length;
    size_t bytes_read;
    char *source;

    if (file == NULL) {
        return NULL;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return NULL;
    }
    source = malloc((size_t)length + 1);
    if (source == NULL) {
        fclose(file);
        return NULL;
    }
    bytes_read = fread(source, 1, (size_t)length, file);
    if (bytes_read != (size_t)length || ferror(file)) {
        free(source);
        fclose(file);
        return NULL;
    }
    source[bytes_read] = '\0';
    fclose(file);
    return source;
}

int main(int argc, char **argv) {
    const char *path = NULL;
    char *source;
    TokenArray tokens;
    LexerError lexer_error;
    AstProgram program;
    ParserError parser_error;
    RuntimeError runtime_error;
    Diagnostic diagnostic;
    IrProgram ir;
    BytecodeProgram bytecode;
    bool use_interpreter = false;
    bool dump_ir = false;
    bool dump_bytecode = false;
    bool optimize = true;
    bool show_time = false;
    bool used_vm = false;
    clock_t compile_start = clock();
    clock_t execute_start;
    int argument;

    for (argument = 1; argument < argc; ++argument) {
        if (strcmp(argv[argument], "--interp") == 0) use_interpreter = true;
        else if (strcmp(argv[argument], "--vm") == 0) use_interpreter = false;
        else if (strcmp(argv[argument], "--version") == 0) printf("Flang Lite 0.7.0\nCopyright (c) 2026 Bekhruz Suleyman.\n");
        else if (strcmp(argv[argument], "--dump-ir") == 0) dump_ir = true;
        else if (strcmp(argv[argument], "--dump-bytecode") == 0)
            dump_bytecode = true;
        else if (strcmp(argv[argument], "--no-opt") == 0) optimize = false;
        else if (strcmp(argv[argument], "--time") == 0) show_time = true;
        else if (argv[argument][0] == '-') {
            fprintf(stderr, "Unknown option: %s\n", argv[argument]);
            return 64;
        } else if (path == NULL) path = argv[argument];
        else {
            fprintf(stderr, "Only one script may be executed\n");
            return 64;
        }
    }
    if (path == NULL) {
        fprintf(stderr,
                "Usage: %s [--interp|--vm] [--dump-ir] "
                "[--dump-bytecode] [--no-opt] [--time] <script.fl>\n",
                argv[0]);
        return 64;
    }
    source = read_source(path);
    if (source == NULL) {
        fprintf(stderr, "%s: unable to read source file\n", path);
        return 74;
    }

    if (!lexer_scan(source, &tokens, &lexer_error)) {
        diagnostic_set(&diagnostic, DIAGNOSTIC_LEXER, path,
                       lexer_error.line, lexer_error.column,
                       lexer_error.message);
        diagnostic_attach_source(&diagnostic, source, 1, NULL);
        diagnostic_print(stderr, &diagnostic);
        free(source);
        return 65;
    }
    if (!parser_parse(&tokens, &program, &parser_error)) {
        diagnostic_set(&diagnostic, DIAGNOSTIC_PARSER, path,
                       parser_error.line, parser_error.column,
                       parser_error.message);
        diagnostic_attach_source(&diagnostic, source, 1, NULL);
        diagnostic_print(stderr, &diagnostic);
        token_array_free(&tokens);
        free(source);
        return 65;
    }
    if (!resolver_check(&program, path, &diagnostic) ||
        !typecheck_check(&program, path, &diagnostic)) {
        diagnostic_print(stderr, &diagnostic);
        ast_program_free(&program);
        token_array_free(&tokens);
        free(source);
        return 65;
    }
    memset(&ir, 0, sizeof(ir));
    memset(&bytecode, 0, sizeof(bytecode));
    if (!use_interpreter) {
        if (!ir_build(&program, path, &ir)) {
            fprintf(stderr, "Out of memory while building IR\n");
            ast_program_free(&program); token_array_free(&tokens); free(source);
            return 70;
        }
        if (optimize) optimizer_run(&ir);
        if (dump_ir) ir_dump(stdout, &ir);
        if (ir.supported) {
            if (!bytecode_compile(&ir, &bytecode)) {
                fprintf(stderr, "Out of memory while compiling bytecode\n");
                ir_program_free(&ir); ast_program_free(&program);
                token_array_free(&tokens); free(source);
                return 70;
            }
            if (dump_bytecode) bytecode_dump(stdout, &bytecode);
            used_vm = true;
        } else if (dump_bytecode) {
            fprintf(stdout, "bytecode fallback: %s\n", ir.unsupported_reason);
        }
    }
    execute_start = clock();
    if (!(used_vm ? vm_run(&bytecode, path, stdout, &runtime_error)
                  : interpreter_run(&program, path, stdout, &runtime_error))) {
        diagnostic_set(&diagnostic, DIAGNOSTIC_RUNTIME,
                       runtime_error.path[0] == '\0' ? path : runtime_error.path,
                       runtime_error.line, runtime_error.column,
                       runtime_error.message);
        if (strcmp(diagnostic.path, path) == 0)
            diagnostic_attach_source(&diagnostic, source, 1,
                                     runtime_error.hint[0] == '\0'
                                         ? NULL
                                         : runtime_error.hint);
        diagnostic_print(stderr, &diagnostic);
        bytecode_program_free(&bytecode);
        ir_program_free(&ir);
        ast_program_free(&program);
        token_array_free(&tokens);
        free(source);
        return 70;
    }

    if (show_time) {
        double compile_ms = 1000.0 * (double)(execute_start - compile_start) /
                            (double)CLOCKS_PER_SEC;
        double execute_ms = 1000.0 * (double)(clock() - execute_start) /
                            (double)CLOCKS_PER_SEC;
        fprintf(stderr, "backend: %s\ncompile: %.3f ms\nexecute: %.3f ms\n",
                used_vm ? "vm" : "interpreter fallback", compile_ms, execute_ms);
    }

    bytecode_program_free(&bytecode);
    ir_program_free(&ir);
    ast_program_free(&program);
    token_array_free(&tokens);
    free(source);
    return 0;
}
