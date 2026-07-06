CC ?= cc
BASE_CFLAGS := -Wall -Wextra -Werror -pedantic -std=c11
CFLAGS ?= $(BASE_CFLAGS) -O2 -DNDEBUG
CPPFLAGS ?= -Iinclude

ifeq ($(OS),Windows_NT)
TARGET := build/flang.exe
else
TARGET := build/flang
endif
CORE_SRC := $(wildcard src/*.c)
RUNTIME_SRC := $(wildcard runtime/*.c)
OBJECTS := $(patsubst src/%.c,build/src/%.o,$(CORE_SRC)) \
	$(patsubst runtime/%.c,build/runtime/%.o,$(RUNTIME_SRC))

.PHONY: all debug release run test clean

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) $(OBJECTS) -o $@

build/src/%.o: src/%.c | build/src
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

build/runtime/%.o: runtime/%.c | build/runtime
	$(CC) $(CPPFLAGS) $(CFLAGS) -c $< -o $@

build/src build/runtime:
	mkdir -p $@

ifeq ($(OS),Windows_NT)
DEBUG_FLAGS := -O0 -g
else
DEBUG_FLAGS := -O0 -g -fsanitize=address,undefined
endif

debug:
	$(MAKE) clean
	$(MAKE) CFLAGS="$(BASE_CFLAGS) $(DEBUG_FLAGS)" all

release:
	$(MAKE) clean
	$(MAKE) CFLAGS="$(BASE_CFLAGS) -O2 -DNDEBUG" all

run: $(TARGET)
	./$(TARGET) examples/v07_vm/tensors.fl

test: $(TARGET)
	python tests/run_tests.py ./$(TARGET)

clean:
	rm -rf build
