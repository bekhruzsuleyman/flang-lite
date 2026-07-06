#ifndef FLANG_SEMANTIC_H
#define FLANG_SEMANTIC_H

#include "ast.h"
#include "diagnostic.h"

#include <stdbool.h>

bool semantic_run(const AstProgram *program, const char *entry_path,
                  Diagnostic *diagnostic, bool type_mode);

#endif
