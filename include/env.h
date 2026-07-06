#ifndef FLANG_ENV_H
#define FLANG_ENV_H

#include "value.h"

#include <stdbool.h>
#include <stddef.h>

typedef struct {
    char *name;
    Value value;
    bool is_public;
} Symbol;

typedef struct Env {
    Symbol *symbols;
    size_t count;
    size_t capacity;
    struct Env *parent;
} Env;

void env_init(Env *env, Env *parent);
void env_free(Env *env);
bool env_define(Env *env, const char *name, Value value, bool is_public);
bool env_assign(Env *env, const char *name, Value value);
bool env_get(const Env *env, const char *name, Value *value);
Value *env_get_slot(Env *env, const char *name);
const Symbol *env_find_local(const Env *env, const char *name);

#endif
