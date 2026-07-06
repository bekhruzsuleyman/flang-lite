# Flang Lite v0.7

Flang Lite is a minimal Python-like scripting language implemented in C.
Version 0.7 makes Flang compiler-shaped: typed AST is lowered to IR, optimized,
compiled to bytecode, and executed by a slot-based stack VM. The v0.6 native
tensor and autograd runtime remains the VM's tensor backend.

For a complete guide, see [Flang Lite: Complete Tutorial](FLANG_LITE_TUTORIAL.md).

## Build and run

You need a C11 compiler, GNU Make, and Python 3 (for the tests).

```sh
make release
./build/flang examples/hello.fl
make run
make test
make debug
make clean
```

On Windows, the executable is `build/flang.exe`.
Debug builds use AddressSanitizer and UndefinedBehaviorSanitizer where the host
toolchain provides them; the Windows target uses portable `-O0 -g` flags.

## VM pipeline and CLI

The default execution pipeline is:

```text
lexer -> parser -> resolver -> type checker -> IR -> optimizer -> bytecode -> VM
```

The VM uses integer slots for module variables, function parameters, and local
variables. Its bytecode has typed integer, string, array, and tensor operations;
known tensor methods are lowered to fixed native method IDs. Constant folding,
dead-code cleanup, and redundant-jump cleanup run before bytecode generation.

```sh
./build/flang program.fl
./build/flang --interp program.fl
./build/flang --vm --dump-ir program.fl
./build/flang --dump-bytecode program.fl
./build/flang --no-opt program.fl
./build/flang --time program.fl
```

`--interp` retains the v0.6 AST evaluator for comparison and debugging. User
module bytecode and index-assignment lowering remain transitional: those
constructs automatically use the compatibility interpreter, while arithmetic,
control flow, functions, arrays, `for`, strings, native tensor imports,
properties, tensor operations, and autograd execute on the VM.

## Language

A Flang program can combine modules, loops, arrays, and tensors:

```flang
values = [1, 2, 3]
for value in values:
    print(value)

x = tensor([[1, 2], [3, 4]])
print(shape(x))
```

Version 0.7 supports all earlier language features, plus:

- Double-quoted strings with `\n`, `\t`, `\"`, and `\\` escapes
- Mixed and nested arrays with indexing and index assignment
- `len()`, two-argument `range()`, and `for` over arrays and ranges
- Multiline array and function-call expressions
- Contiguous f64 tensors built from rectangular integer arrays
- `shape()`, `rank()`, `zeros()`, and `ones()`
- Checked tensor indexing and same-shape elementwise arithmetic
- Scalar/tensor arithmetic in either operand order
- `.shape`, `.rank`, and `.len` tensor properties
- `.len` on arrays and strings
- Pre-execution name, scope, import, and visibility resolution
- Static operator, assignment, call, return, condition, and known-shape checks
- Inferred declarations such as `error = pred - target` inside functions
- Homogeneous array element inference such as `array[int]`
- Source-line, caret, and actionable-hint diagnostics
- Native `tensor` module imports and qualified module calls
- Tensor `.sum()`, `.mean()`, `.backward()`, and `.zero_grad()` methods
- Tensor `.grad` and `.requires_grad` properties
- Autograd for `+`, `-`, `*`, `/`, `sum`, and `mean`
- Typed IR and readable IR dumps
- Constant folding and bytecode compilation
- Slot-based VM locals and globals
- Specialized tensor bytecode operations
- VM/interpreter selection and compile/execution timing

Blocks use leading spaces. Tabs are rejected when used for indentation, and
blank and comment-only lines do not affect indentation. Every module has its
own global environment; each function call creates a local environment.
Assignments update the nearest existing binding. Typed declarations inside a
function create locals, allowing safe shadowing of module variables. An
unannotated assignment creates an inferred variable in the current scope when
no existing binding is found.

Function calls validate argument counts and parameter types. Non-void functions
must return a value of their declared type, while void functions cannot return a
value. Arithmetic, comparisons, boolean logic, and conditions retain v0.2's
strict runtime checks. Declarations support `int`, `bool`, `str`, `list`, and
`tensor`; v0.5 checks declarations and later assignments before evaluation.

## Pipeline and diagnostics

Every program now passes through:

```text
lexer -> parser -> resolver -> type checker -> evaluator
```

The evaluator runs only after both semantic passes succeed. The resolver checks
names, duplicate declarations, scopes, modules, imports, and public visibility
without executing module code. The type checker tracks inferred and annotated
types, function signatures, data properties, and statically known tensor
shapes. Diagnostics share the format:

```text
file.fl:line:column: type error: message

    source line
    ^^^^^

hint: an actionable suggestion
```

## Native tensors and autograd

Arrays use deep-copy value semantics when assigned or passed to functions:

```flang
items = [1, 2, 3]
copy = items
copy[0] = 9
print(items) # [1, 2, 3]
```

Tensor literals accept nested integer arrays but store data as doubles. Operations
run as contiguous native C loops and support equal tensor shapes or one tensor
and one integer scalar. The evaluator dispatches once per operation—there is no
per-element dynamic `Value` dispatch.

Import the native module in either style:

```flang
from tensor import tensor, zeros, ones, arange

x = tensor([1, 2, 3], true)
y = x * x + 2
loss = y.sum()
loss.backward()
print(x.grad) # [2, 4, 6]
```

```flang
import tensor
x = tensor.tensor([1, 2, 3])
```

The second `tensor()` argument is optional and controls `requires_grad`.
Autograd nodes are allocated only when an operand requires gradients. Backward
uses an iterative reverse-topological traversal, gradients accumulate across
calls, and `zero_grad()` resets a tensor's accumulated gradient. `backward()`
requires a scalar result, normally produced with `sum()` or `mean()`.

Properties are available without function calls:

```flang
x = tensor([[1, 2], [3, 4]])
print(x.shape) # [2, 2]
print(x.rank)  # 2
print(x.len)   # 4
print(x.requires_grad)
```

## Modules

Imports first search for `<module>.fl` relative to the entry script, then the
project `stdlib/` directory, then the native module registry. A local module can
therefore intentionally shadow a standard module.
Modules are cached, so duplicate imports execute a module only once. Circular
imports are detected and rejected. Declarations are private by default:

```flang
# math.fl
pub fn square(x: int) -> int:
    return x * x

fn helper(x: int) -> int:
    return x + 1
```

```flang
# main.fl
import math
print(math.square(5))
```

The interpreter reports lexer, parser, resolve, type, and runtime errors with
source line and column information. Errors include invalid indentation,
undefined variables, incorrect operator types, non-boolean conditions, division
by zero, and integer overflow.

The [v0.7 VM examples](examples/v07_vm/) cover IR-friendly arithmetic, slots,
control flow, functions, arrays, modules, tensors, and a loop benchmark.
The [v0.6 autograd examples](examples/v06_autograd/) cover native imports,
reductions, gradients, accumulation, and diagnostics. The
[performance examples](examples/v06_perf/) exercise the grad and no-grad paths.
Earlier [v0.5 examples](examples/v05_stability/) remain supported.

Broadcasting, matmul, neural-network layers, optimizers, GPU execution, SIMD,
views/strides, higher-order gradients, bytecode, LLVM, and JIT compilation
remain outside the v0.6 scope.
