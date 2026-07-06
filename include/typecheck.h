#ifndef FLANG_TYPECHECK_H
#define FLANG_TYPECHECK_H

#include "ast.h"
#include "diagnostic.h"

#include <stdbool.h>

bool typecheck_check(const AstProgram *program, const char *entry_path,
                     Diagnostic *diagnostic);

#endif
