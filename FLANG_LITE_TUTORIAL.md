# Flang Lite: Complete Tutorial

This handbook documents Flang Lite v0.7 as implemented in this repository. It
covers the language, native tensor runtime, automatic differentiation, module
system, compiler pipeline, bytecode VM, command-line tools, diagnostics, and
development workflow.
(Author: Bekhruz Suleyman)

## Contents

1. [What Flang Lite is](#1-what-flang-lite-is)
2. [Building and running](#2-building-and-running)
3. [Your first program](#3-your-first-program)
4. [Lexical rules and formatting](#4-lexical-rules-and-formatting)
5. [Values and types](#5-values-and-types)
6. [Variables and assignment](#6-variables-and-assignment)
7. [Operators and expressions](#7-operators-and-expressions)
8. [Control flow](#8-control-flow)
9. [Functions](#9-functions)
10. [Strings](#10-strings)
11. [Arrays and indexing](#11-arrays-and-indexing)
12. [Built-in functions](#12-built-in-functions)
13. [Modules and visibility](#13-modules-and-visibility)
14. [Tensors](#14-tensors)
15. [Automatic differentiation](#15-automatic-differentiation)
16. [Errors and diagnostics](#16-errors-and-diagnostics)
17. [IR, optimization, bytecode, and the VM](#17-ir-optimization-bytecode-and-the-vm)
18. [CLI reference](#18-cli-reference)
19. [Testing and development](#19-testing-and-development)
20. [Current limitations](#20-current-limitations)
21. [Worked examples](#21-worked-examples)
22. [Quick reference](#22-quick-reference)

## 1. What Flang Lite is

Flang Lite is a small Python-like language implemented in C11. Its design is
deliberately compact, but it includes several features normally found in much
larger systems:

- indentation-based blocks;
- static semantic checks before execution;
- inferred and explicitly annotated variables;
- functions, recursion, modules, and visibility;
- strings, homogeneous arrays, loops, and checked indexing;
- contiguous native f64 tensors;
- reverse-mode automatic differentiation;
- typed intermediate representation (IR);
- optimization and bytecode compilation;
- a slot-based stack virtual machine.

The normal v0.7 pipeline is:

```text
source
  -> lexer
  -> parser
  -> resolver
  -> type checker
  -> typed IR builder
  -> optimizer
  -> bytecode compiler
  -> stack VM
```

The previous AST interpreter remains available for debugging and as a
compatibility backend.

## 2. Building and running

### Requirements

- a C11 compiler such as GCC or Clang;
- GNU Make;
- Python 3 for the test suite.

### Linux or WSL

```bash
make release
./build/flang examples/v07_vm/arithmetic.fl
make test
```

`make` also builds the optimized default target. `make debug` uses `-O0 -g`
and enables AddressSanitizer/UndefinedBehaviorSanitizer when supported.

### Windows

With a GNU-compatible compiler and Make:

```powershell
make release
./build/flang.exe examples/v07_vm/arithmetic.fl
make test
```

The Windows executable is `build/flang.exe`; Linux and WSL use `build/flang`.

### Make targets

| Command | Purpose |
|---|---|
| `make` | Build the optimized executable |
| `make release` | Clean and build with `-O2 -DNDEBUG` |
| `make debug` | Clean and build a debug executable |
| `make run` | Run the current v0.7 tensor example |
| `make test` | Run the complete Python test suite |
| `make clean` | Remove generated build files |

## 3. Your first program

Create `hello.fl`:

```flang
message: str = "Hello from Flang Lite"
print(message)

x = 5
y = x * 2 + 1
print(y)
```

Run it:

```bash
./build/flang hello.fl
```

Output:

```text
Hello from Flang Lite
11
```

The `.fl` file is checked by the resolver and type checker before any statement
is executed.

## 4. Lexical rules and formatting

### Statements and newlines

Statements are newline-separated. A final newline is optional.

```flang
x = 1
y = 2
print(x + y)
```

Multiple statements cannot be separated with semicolons.

### Indentation

Leading spaces create blocks. Use a consistent number of spaces; four is
recommended. Tabs used for indentation are rejected.

```flang
if true:
    print(1)
    print(2)
```

Blank lines and comment-only lines do not change indentation.

### Comments

`#` starts a comment outside a string:

```flang
# Full-line comment
x = 5 # Inline comment
print("# is text inside a string")
```

### Keywords

```text
print  if  else  while  for  in
true   false  and  or  not
fn     return  void
pub    import  from  as
```

Identifiers contain letters, digits, and underscores and cannot begin with a
digit.

## 5. Values and types

Flang Lite exposes these source-level types:

| Type | Example | Notes |
|---|---|---|
| `int` | `42`, `-7` | Signed 64-bit integer |
| `bool` | `true`, `false` | Used by conditions and logic |
| `str` | `"hello"` | Owned UTF-8 byte string |
| `list` | `[1, 2, 3]` | Homogeneous array |
| `tensor` | `tensor([1, 2])` | Native contiguous f64 tensor |
| `void` | Function return type | No source-level value literal |

Ranges, functions, modules, and native functions are internal runtime value
kinds. There is currently no source-level floating-point scalar type. Tensor
data is stored as double precision even when constructed from integer literals.

## 6. Variables and assignment

### Inferred declarations

The first assignment creates a variable when no binding exists:

```flang
count = 0
name = "Flang"
enabled = true
```

The inferred type is stable:

```flang
x = 5
x = 8       # valid
x = "five"  # type error
```

### Explicit annotations

```flang
count: int = 0
name: str = "Flang"
flags: list = [true, false]
weights: tensor = tensor([1, 2, 3])
```

Annotations are checked before evaluation.

### Scope and shadowing

Assignments update the nearest existing binding. Explicit declarations inside
a function create local variables and can safely shadow module variables:

```flang
x: int = 10

fn local_value() -> int:
    x: int = 3
    return x

print(local_value()) # 3
print(x)             # 10
```

An inferred function local does not escape its function.

### Public variables

At module level, annotated variables can be exported:

```flang
pub VERSION: int = 7
```

## 7. Operators and expressions

### Arithmetic

```flang
print(8 + 2)  # 10
print(8 - 2)  # 6
print(8 * 2)  # 16
print(8 / 3)  # 2: integer division
print(-8)     # -8
```

Integer overflow and division by zero are runtime errors.

### Comparisons

```flang
print(2 < 3)
print(2 <= 2)
print(4 > 1)
print(4 >= 4)
print(5 == 5)
print(5 != 6)
```

Equality supports compatible `int`, `bool`, and `str` operands. Ordering
comparisons currently require integers.

### Boolean logic

```flang
ready = true
empty = false
print(ready and not empty)
print(ready or empty)
```

Conditions must be `bool`; integers are not implicitly truthy.

### String concatenation

```flang
greeting = "hello " + "world"
print(greeting)
```

### Precedence

From highest to lowest:

1. postfix calls, properties, and indexing: `f()`, `x.len`, `x[0]`;
2. unary `-` and `not`;
3. `*` and `/`;
4. `+` and `-`;
5. `<`, `<=`, `>`, `>=`;
6. `==`, `!=`;
7. `and`;
8. `or`.

Parentheses override precedence:

```flang
print(2 + 3 * 4)   # 14
print((2 + 3) * 4) # 20
```

## 8. Control flow

### `if` and `else`

```flang
score = 82
if score >= 60:
    print("pass")
else:
    print("fail")
```

There is no `elif`; nest another `if` in the `else` block when needed.

### `while`

```flang
i = 0
while i < 3:
    print(i)
    i = i + 1
```

### `for` over arrays

```flang
items = [10, 20, 30]
for item in items:
    print(item)
```

### `for` over ranges

`range(start, end)` includes `start` and excludes `end`:

```flang
for i in range(0, 4):
    print(i)
```

Output is `0`, `1`, `2`, `3` on separate lines.

`break` and `continue` are not implemented.

## 9. Functions

### Declaration and call

Parameters always require types. The return annotation is optional and defaults
to `void`.

```flang
fn add(a: int, b: int) -> int:
    return a + b

print(add(3, 4))
```

### Void functions

```flang
fn announce(message: str):
    print(message)

announce("training started")
```

A void function may use bare `return`, but cannot return a value.

### Recursion

```flang
fn factorial(n: int) -> int:
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5)) # 120
```

### Function checking

The compiler checks:

- argument count;
- argument types;
- returned value type;
- values returned from void functions;
- missing returns in non-void functions;
- `return` outside a function.

### Exported functions

```flang
pub fn square(x: int) -> int:
    return x * x
```

Functions are private to their module unless marked `pub`.

## 10. Strings

Strings use double quotes:

```flang
name = "Flang Lite"
print(name)
```

Supported escapes:

| Escape | Meaning |
|---|---|
| `\n` | Newline |
| `\t` | Tab |
| `\"` | Double quote |
| `\\` | Backslash |

Properties and operations:

```flang
text = "hello"
print(text.len) # 5
print(len(text))
print(text[1])  # e
print("a" + "b")
```

String indexing returns a one-character string. Indexes are zero-based.

## 11. Arrays and indexing

### Creating arrays

```flang
numbers = [1, 2, 3]
matrix = [[1, 2], [3, 4]]
empty = []
```

Non-empty arrays are homogeneous. `[1, "two"]` is rejected by the type checker.

### Reading elements

```flang
numbers = [10, 20, 30]
print(numbers[0])
print(numbers[2])

matrix = [[1, 2], [3, 4]]
print(matrix[1][0]) # 3
```

### Assigning elements

```flang
numbers = [1, 2, 3]
numbers[1] = 9
print(numbers) # [1, 9, 3]
```

Nested index assignment is supported by the compatibility interpreter.

### Length

```flang
print(numbers.len)
print(len(numbers))
```

### Copy semantics

Arrays use deep-copy value semantics:

```flang
a = [[1, 2], [3, 4]]
b = a
b[0][0] = 9
print(a) # [[1, 2], [3, 4]]
print(b) # [[9, 2], [3, 4]]
```

## 12. Built-in functions

| Function | Result | Example |
|---|---|---|
| `print(value)` | `void` | `print(42)` |
| `len(value)` | `int` | `len([1, 2])` |
| `range(start, end)` | range | `range(0, 3)` |
| `tensor(values[, requires_grad])` | `tensor` | `tensor([1, 2], true)` |
| `shape(tensor)` | `list` | `shape(x)` |
| `rank(tensor)` | `int` | `rank(x)` |
| `zeros(shape)` | `tensor` | `zeros([2, 3])` |
| `ones(shape)` | `tensor` | `ones([2, 3])` |

For new tensor code, importing the native `tensor` module is preferred.

## 13. Modules and visibility

### Creating a module

`math.fl`:

```flang
pub SCALE: int = 2

pub fn square(x: int) -> int:
    return x * x

fn private_helper(x: int) -> int:
    return x + 1
```

### Qualified import

`main.fl`:

```flang
import math
print(math.square(6))
print(math.SCALE)
```

Aliases are supported:

```flang
import math as m
print(m.square(5))
```

### Selective import

```flang
from math import square, SCALE
print(square(SCALE + 2))
```

Imported names must be public. Accessing `private_helper` from another module is
a resolver error, so no module code runs before the error is reported.

### Resolution order

Modules are searched in this order:

1. `<module>.fl` beside the entry script;
2. the project `stdlib/` directory;
3. registered native runtime modules.

A local `tensor.fl` therefore shadows the native tensor module. Modules are
cached and initialized once. Circular imports are detected.

### VM note

Native tensor imports compile to bytecode. User-defined module initialization
and index assignment currently use the AST compatibility backend automatically.
Use `--time` to see which backend executed a program.

## 14. Tensors

### Importing the native module

Preferred selective style:

```flang
from tensor import tensor, zeros, ones, arange
```

Qualified style:

```flang
import tensor
x = tensor.tensor([1, 2, 3])
```

### Constructing tensors

```flang
vector = tensor([1, 2, 3])
matrix = tensor([[1, 2], [3, 4]])
trainable = tensor([1, 2, 3], true)
```

The second positional argument controls gradient tracking. Named arguments are
not supported.

Tensor inputs must be rectangular integer arrays. Ragged arrays are rejected:

```flang
bad = tensor([[1, 2], [3]]) # error
```

### Factories

```flang
from tensor import zeros, ones, arange

print(zeros([2, 3])) # [[0, 0, 0], [0, 0, 0]]
print(ones([2, 2]))  # [[1, 1], [1, 1]]
print(arange(2, 6))  # [2, 3, 4, 5]
```

### Properties

```flang
x = tensor([[1, 2], [3, 4]])
print(x.shape)         # [2, 2]
print(x.rank)          # 2
print(x.len)           # 4
print(x.requires_grad) # false
```

After backpropagation, `.grad` returns a tensor containing accumulated gradients.

### Arithmetic

```flang
a = tensor([1, 2, 3])
b = tensor([4, 5, 6])

print(a + b) # [5, 7, 9]
print(a - b) # [-3, -3, -3]
print(a * b) # [4, 10, 18]
print(b / a) # [4, 2.5, 2]
```

Tensor/scalar arithmetic works in either direction:

```flang
print(a + 10)
print(10 + a)
print(a * 3)
print(12 / a)
```

Two tensor operands must have exactly the same shape. Broadcasting is not yet
implemented.

### Reduction methods

```flang
x = tensor([1, 2, 3, 4])
print(x.sum())  # 10
print(x.mean()) # 2.5
```

Reductions return rank-zero scalar tensors, not Flang `int` values.

### Indexing

```flang
x = tensor([[1, 2], [3, 4]])
print(x[0])    # [1, 2]
print(x[1][0]) # 3
```

Tensor slices are compact copies; advanced views, strides, and slicing syntax
are not implemented.

### Runtime representation

Tensors use one f64 dtype and contiguous owned storage. Shape metadata and
autograd graph references are reference-counted. Arithmetic dispatch occurs
once per tensor operation, followed by a tight native C loop over raw doubles.

## 15. Automatic differentiation

### Basic gradient

```flang
from tensor import tensor

x = tensor([1, 2, 3], true)
y = x * x
loss = y.sum()
loss.backward()
print(x.grad) # [2, 4, 6]
```

The graph is created only when at least one operand has `requires_grad = true`.
No-grad tensor arithmetic takes the faster graph-free path.

### Supported differentiable operations

```text
+  -  *  /  sum()  mean()
```

Gradient rules include:

```text
y = a + b     da += dy;                 db += dy
y = a - b     da += dy;                 db -= dy
y = a * b     da += dy * b;             db += dy * a
y = a / b     da += dy / b;             db -= dy*a/(b*b)
y = sum(a)    da += dy
y = mean(a)   da += dy / a.len
```

### Linear expression

```flang
x = tensor([1, 2, 3], true)
y = x * 3 + 5
y.sum().backward()
print(x.grad) # [3, 3, 3]
```

### Gradient accumulation

Gradients accumulate across backward calls:

```flang
x = tensor([1, 2, 3], true)
loss = (x * x).sum()
loss.backward()
loss.backward()
print(x.grad) # [4, 8, 12]
```

Reset them explicitly:

```flang
x.zero_grad()
print(x.grad) # [0, 0, 0]
```

### Scalar loss requirement

`backward()` requires a rank-zero tensor:

```flang
x = tensor([1, 2, 3], true)
x.backward() # error
```

Use `x.sum().backward()` or `x.mean().backward()` instead.

### Backward implementation

The runtime builds an iterative topological ordering and walks it in reverse.
It does not recursively traverse the graph, avoiding C stack overflow on deep
graphs. Shared operands are visited once, while their gradient contributions
are accumulated correctly.

## 16. Errors and diagnostics

Flang Lite reports the phase, location, source line, caret, and often a hint:

```text
example.fl:2:10: runtime error: Division by zero

    print(10 / x)
             ^
```

Diagnostic phases:

| Phase | Typical errors |
|---|---|
| Lexer | Unknown character, invalid escape, bad indentation |
| Parser | Missing `:`, `)`, expression, or indented block |
| Resolver | Unknown name/module, duplicate declaration, private symbol |
| Type checker | Wrong operand, argument, assignment, property, or shape |
| Runtime/VM | Division by zero, overflow, bad index, dynamic shape mismatch |

Semantic failures happen before evaluation, preventing partial output and
module side effects.

Useful debugging procedure:

1. read the first diagnostic;
2. inspect its source line and caret;
3. apply the hint when present;
4. use `--dump-ir` if lowering is relevant;
5. compare `--vm` with `--interp` if investigating an execution backend.

## 17. IR, optimization, bytecode, and the VM

### Slot assignment

Names are resolved to integer slots before VM execution. A function such as:

```flang
fn add(a: int, b: int) -> int:
    c = a + b
    return c
```

uses slots `a = 0`, `b = 1`, and `c = 2`. The VM does not hash or compare local
names in its hot execution loop.

### Inspecting IR

```bash
./build/flang --dump-ir examples/v07_vm/arithmetic.fl
```

Optimized output resembles:

```text
IR function __main__ (slots=0):
  0000  CONST                14
  0001  STORE_GLOBAL         0
  0002  LOAD_GLOBAL          0
  0003  PRINT
  0004  HALT
```

The optimizer folds `2 + 3 * 4` to `14`. Use `--no-opt` to see the unfused
instruction sequence.

### Optimizer passes

- safe integer and boolean constant folding;
- preservation of division-by-zero and overflow behavior;
- dead instruction cleanup after simple returns;
- redundant jump cleanup;
- removal of optimizer-generated no-ops.

### Inspecting bytecode

```bash
./build/flang --dump-bytecode examples/v07_vm/tensors.fl
```

Tensor expressions use specialized operations such as:

```text
OP_TENSOR_MUL
OP_TENSOR_ADD
OP_GET_PROPERTY
```

The VM dispatches once to the native tensor kernel rather than boxing each
element.

### VM execution

The VM owns:

- a dynamically growing, preallocated value stack;
- slot arrays for globals and function locals;
- bytecode function metadata;
- source spans for runtime diagnostics;
- the existing native tensor/autograd runtime.

Function parameters occupy the first local slots. Calls transfer arguments to
the callee slots and return one value to the caller.

### Backend fallback

The VM is the default. Programs containing user-module execution or index
assignment currently fall back automatically to the AST interpreter. This is a
migration mechanism, not a semantic difference. Check the selected backend:

```bash
./build/flang --time program.fl
```

## 18. CLI reference

General form:

```text
flang [options] script.fl
```

| Option | Behavior |
|---|---|
| `--vm` | Select the VM pipeline (default) |
| `--interp` | Select the AST compatibility interpreter |
| `--dump-ir` | Print optimized IR before execution |
| `--dump-bytecode` | Disassemble generated bytecode |
| `--no-opt` | Disable optimizer passes |
| `--time` | Print backend, compilation time, and execution time to stderr |

Options may appear before the script path. Only one script is accepted.

Examples:

```bash
./build/flang program.fl
./build/flang --interp program.fl
./build/flang --dump-ir --no-opt program.fl
./build/flang --dump-bytecode --time program.fl
```

## 19. Testing and development

### Run all tests

```bash
make test
```

Or directly:

```bash
python3 tests/run_tests.py ./build/flang
```

On Windows:

```powershell
python tests/run_tests.py build/flang.exe
```

The suite covers every language generation from v0.1 through v0.7, including
parser behavior, scopes, type failures, modules, arrays, tensors, autograd,
IR dumps, bytecode, backend selection, and VM source-span diagnostics.

### Strict manual build

```bash
cc -Wall -Wextra -Werror -pedantic -std=c11 -O2 -DNDEBUG \
   -Iinclude src/*.c runtime/*.c -o build/flang
```

### Performance comparison

```bash
./build/flang --time examples/v07_vm/loop_benchmark.fl
./build/flang --interp --time examples/v07_vm/loop_benchmark.fl
```

Do not include compilation time when comparing execution engines. Run each
benchmark several times and compare the execution line or an external timer.

### Source layout

```text
include/       public compiler/runtime headers
src/           lexer, parser, semantics, IR, optimizer, bytecode, VM, CLI
runtime/       values, native tensor kernels, autograd, native registry
stdlib/        interpreted standard-library module placeholders
examples/      versioned runnable programs
tests/         end-to-end Python tests
build/         generated objects and executable
```

## 20. Current limitations

Flang Lite v0.7 intentionally does not include:

- floating-point source literals or scalar `float` type;
- broadcasting;
- matrix multiplication;
- tensor slicing, views, or strides;
- multiple tensor dtypes;
- higher-order gradients;
- neural-network layers or optimizers;
- GPU execution or SIMD kernels;
- classes, structs, dictionaries, or generics;
- `break`, `continue`, `elif`, exceptions, or named arguments;
- package management;
- FFI/`extern C`;
- JIT, LLVM, or native machine-code generation.

User modules and index assignment retain interpreter fallback in v0.7. The
next compiler milestone can move those final constructs into cached module
bytecode objects.

## 21. Worked examples

### Example A: classify values

```flang
fn is_large(value: int) -> bool:
    return value >= 10

values = [4, 10, 15]
for value in values:
    if is_large(value):
        print("large")
    else:
        print("small")
```

### Example B: sum a range

```flang
total = 0
for i in range(1, 6):
    total = total + i
print(total) # 15
```

### Example C: tensor expression

```flang
from tensor import tensor

a = tensor([1, 2, 3])
b = tensor([4, 5, 6])
c = a * b + 10
print(c)       # [14, 20, 28]
print(c.shape) # [3]
print(c.rank)  # 1
```

### Example D: quadratic gradient

```flang
from tensor import tensor

x = tensor([1, 2, 3], true)
prediction = x * x + 2
loss = prediction.mean()
loss.backward()

print(prediction)
print(loss)
print(x.grad)
```

For `mean(x*x + 2)`, the gradient is `2*x/3`:

```text
[0.666666666666667, 1.33333333333333, 2]
```

### Example E: module-based program

`model.fl`:

```flang
pub fn predict(weight: int, bias: int, inputs: tensor) -> tensor:
    return inputs * weight + bias
```

`main.fl`:

```flang
import model
from tensor import tensor

inputs = tensor([1, 2, 3])
print(model.predict(2, 1, inputs)) # [3, 5, 7]
```

## 22. Quick reference

```flang
# Values
x: int = 1
ok: bool = true
name: str = "Flang"
items: list = [1, 2, 3]
t: tensor = tensor([1, 2, 3])

# Branches and loops
if ok:
    print(name)
else:
    print("not ready")

while x < 3:
    x = x + 1

for item in items:
    print(item)

# Functions
pub fn double(value: int) -> int:
    return value * 2

# Modules
import math as m
from math import square

# Arrays and strings
print(items[0])
items[0] = 9
print(items.len)
print(name.len)

# Tensors and autograd
from tensor import tensor, zeros, ones, arange
weights = tensor([1, 2, 3], true)
loss = (weights * weights).sum()
loss.backward()
print(weights.grad)
weights.zero_grad()
```

For more runnable programs, browse `examples/v07_vm/`,
`examples/v06_autograd/`, `examples/v05_stability/`, and `examples/v04_data/`.
