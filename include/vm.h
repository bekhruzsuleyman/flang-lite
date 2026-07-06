#ifndef FLANG_VM_H
#define FLANG_VM_H

#include "bytecode.h"
#include "interpreter.h"

bool vm_run(const BytecodeProgram *program, const char *entry_path,
            FILE *output, RuntimeError *error);

#endif
