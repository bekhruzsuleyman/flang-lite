#include "env.h"

#include <stdlib.h>
#include <string.h>

static char *copy_string(const char *source) {
    size_t length = strlen(source);
    char *copy = malloc(length + 1);
    if (copy != NULL) memcpy(copy, source, length + 1);
    return copy;
}

void env_init(Env *env, Env *parent) {
    *env = (Env){0};
    env->parent = parent;
}

void env_free(Env *env) {
    size_t index;
    for (index = 0; index < env->count; ++index) {
        free(env->symbols[index].name);
        value_free(&env->symbols[index].value);
    }
    free(env->symbols);
    *env = (Env){0};
}

static Symbol *find_local_mutable(Env *env, const char *name) {
    size_t index;
    for (index = 0; index < env->count; ++index) {
        if (strcmp(env->symbols[index].name, name) == 0) return &env->symbols[index];
    }
    return NULL;
}

const Symbol *env_find_local(const Env *env, const char *name) {
    size_t index;
    for (index = 0; index < env->count; ++index) {
        if (strcmp(env->symbols[index].name, name) == 0) return &env->symbols[index];
    }
    return NULL;
}

bool env_define(Env *env, const char *name, Value value, bool is_public) {
    Symbol *symbol = find_local_mutable(env, name);
    Symbol *new_symbols;
    size_t new_capacity;
    char *name_copy;
    Value cloned;
    if (!value_clone(&value, &cloned)) return false;
    if (symbol != NULL) {
        value_free(&symbol->value);
        symbol->value = cloned;
        symbol->is_public = is_public;
        return true;
    }
    name_copy = copy_string(name);
    if (name_copy == NULL) {
        value_free(&cloned);
        return false;
    }
    if (env->count == env->capacity) {
        new_capacity = env->capacity == 0 ? 8 : env->capacity * 2;
        new_symbols = realloc(env->symbols, new_capacity * sizeof(*new_symbols));
        if (new_symbols == NULL) {
            free(name_copy);
            value_free(&cloned);
            return false;
        }
        env->symbols = new_symbols;
        env->capacity = new_capacity;
    }
    env->symbols[env->count++] = (Symbol){name_copy, cloned, is_public};
    return true;
}

bool env_assign(Env *env, const char *name, Value value) {
    Env *current;
    for (current = env; current != NULL; current = current->parent) {
        Symbol *symbol = find_local_mutable(current, name);
        if (symbol != NULL) {
            Value cloned;
            if (!value_clone(&value, &cloned)) return false;
            value_free(&symbol->value);
            symbol->value = cloned;
            return true;
        }
    }
    return false;
}

bool env_get(const Env *env, const char *name, Value *value) {
    const Env *current;
    for (current = env; current != NULL; current = current->parent) {
        const Symbol *symbol = env_find_local(current, name);
        if (symbol != NULL) {
            return value_clone(&symbol->value, value);
        }
    }
    return false;
}

Value *env_get_slot(Env *env, const char *name) {
    Env *current;
    for (current = env; current != NULL; current = current->parent) {
        Symbol *symbol = find_local_mutable(current, name);
        if (symbol != NULL) return &symbol->value;
    }
    return NULL;
}
