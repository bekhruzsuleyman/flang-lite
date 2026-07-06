#include "typecheck.h"

#include "semantic.h"

bool typecheck_check(const AstProgram *program, const char *entry_path,
                     Diagnostic *diagnostic) {
    return semantic_run(program, entry_path, diagnostic, true);
}
