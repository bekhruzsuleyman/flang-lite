#ifndef FLANG_RESOLVER_H
#define FLANG_RESOLVER_H

#include "ast.h"
#include "diagnostic.h"

#include <stdbool.h>

bool resolver_check(const AstProgram *program, const char *entry_path,
                    Diagnostic *diagnostic);

#endif
