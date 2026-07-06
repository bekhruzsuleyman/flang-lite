#!/usr/bin/env python3
"""End-to-end tests for the Flang Lite command-line interpreter."""

from __future__ import annotations

import pathlib
import subprocess
import sys
import tempfile
import unittest


if len(sys.argv) > 1:
    EXECUTABLE = pathlib.Path(sys.argv.pop(1)).resolve()
else:
    suffix = ".exe" if sys.platform == "win32" else ""
    EXECUTABLE = (pathlib.Path(__file__).parents[1] / "build" / f"flang{suffix}").resolve()


def run_project(
    files: dict[str, str], entry: str = "main.fl", args: list[str] | None = None
) -> subprocess.CompletedProcess[str]:
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        for name, source in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(source.encode("utf-8"))
        return subprocess.run(
            [str(EXECUTABLE), *(args or []), str(root / entry)],
            text=True,
            capture_output=True,
            check=False,
        )


def run_flang(source: str) -> subprocess.CompletedProcess[str]:
    return run_project({"main.fl": source})


class FlangTests(unittest.TestCase):
    def assert_program(self, source: str, expected: str) -> None:
        result = run_flang(source)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, expected)
        self.assertEqual(result.stderr, "")

    def assert_error(self, source: str, category: str, message: str) -> None:
        result = run_flang(source)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn(category, result.stderr)
        self.assertIn(message, result.stderr)

    def test_success_criterion(self) -> None:
        self.assert_program("x = 5\ny = x * 2 + 1\nprint(y)\n", "11\n")

    def test_precedence_parentheses_and_unary(self) -> None:
        self.assert_program(
            "print(2 + 3 * 4)\nprint((2 + 3) * 4)\nprint(-2 * -3)\n",
            "14\n20\n6\n",
        )

    def test_annotations_comments_blanks_and_reassignment(self) -> None:
        self.assert_program(
            "# annotations are accepted in v0.1\n\nvalue: int = 7 # inline\n"
            "value = value / 2\nprint(value)\n",
            "3\n",
        )

    def test_crlf_input(self) -> None:
        self.assert_program("x = 4\r\nprint(x)\r\n", "4\n")

    def test_empty_and_comment_only_programs(self) -> None:
        self.assert_program("", "")
        self.assert_program("# nothing to execute", "")

    def test_final_newline_is_optional(self) -> None:
        self.assert_program("value = 9\nprint(value)", "9\n")

    def test_eof_closes_all_nested_blocks(self) -> None:
        self.assert_program(
            "if true:\n    if true:\n        print(1)",
            "1\n",
        )

    def test_unknown_variable(self) -> None:
        self.assert_error("print(missing)\n", "resolve error", "Unknown variable")

    def test_division_by_zero(self) -> None:
        self.assert_error("print(10 / 0)\n", "runtime error", "Division by zero")

    def test_unexpected_character(self) -> None:
        self.assert_error("x = @\n", "lexer error", "Unexpected character")

    def test_expected_expression(self) -> None:
        self.assert_error("x =\n", "parser error", "Expected expression")

    def test_statements_must_be_newline_separated(self) -> None:
        self.assert_error("x = 1 print(x)\n", "parser error", "Expected newline")

    def test_integer_literal_range(self) -> None:
        self.assert_error(
            "print(999999999999999999999999)\n",
            "lexer error",
            "out of range",
        )

    def test_runtime_integer_overflow(self) -> None:
        self.assert_error(
            "print(9223372036854775807 + 1)\n",
            "runtime error",
            "Integer overflow",
        )

    def test_boolean_literals_and_logic(self) -> None:
        self.assert_program(
            "print(true)\nprint(false)\nprint(not false)\n"
            "print(true and true)\nprint(false or true)\n",
            "true\nfalse\ntrue\ntrue\ntrue\n",
        )

    def test_comparisons(self) -> None:
        self.assert_program(
            "x = 5\nprint(x == 5)\nprint(x != 3)\nprint(x < 10)\n"
            "print(x <= 5)\nprint(x > 1)\nprint(x >= 5)\n",
            "true\ntrue\ntrue\ntrue\ntrue\ntrue\n",
        )

    def test_boolean_precedence(self) -> None:
        self.assert_program(
            "print(true or false and false)\n"
            "print(not false and 1 + 2 * 3 == 7)\n",
            "true\ntrue\n",
        )

    def test_if_else_and_optional_else(self) -> None:
        self.assert_program(
            "x = 7\nif x > 10:\n    print(100)\nelse:\n    print(200)\n"
            "if x > 0:\n    print(x)\n",
            "200\n7\n",
        )

    def test_while_assignment_and_nested_blocks(self) -> None:
        self.assert_program(
            "i = 0\nwhile i < 3:\n    if i == 1:\n        print(100)\n"
            "    else:\n        print(i)\n    i = i + 1\n",
            "0\n100\n2\n",
        )

    def test_blank_and_comment_lines_do_not_change_indentation(self) -> None:
        self.assert_program(
            "if true:\n    # inside the block\n\n    print(1)\nprint(2)\n",
            "1\n2\n",
        )

    def test_block_declaration_uses_global_scope(self) -> None:
        self.assert_program("if true:\n    answer: int = 42\nprint(answer)\n", "42\n")

    def test_expression_statement(self) -> None:
        self.assert_program("1 + 2\ntrue\n", "")

    def test_condition_must_be_boolean(self) -> None:
        self.assert_error("if 1:\n    print(1)\n", "type error", "condition must be bool")

    def test_operator_type_errors(self) -> None:
        self.assert_error("print(true + 1)\n", "type error", "operator '+' with bool and int")
        self.assert_error("print(5 and true)\n", "type error", "operator 'and' with int and bool")
        self.assert_error("print(not 1)\n", "type error", "Operator 'not' cannot be used with int")

    def test_missing_or_empty_indented_block(self) -> None:
        self.assert_error(
            "if true:\nprint(1)\n",
            "parser error",
            "Expected indented block",
        )
        self.assert_error(
            "if true:\n    # no statement\n",
            "parser error",
            "Expected indented block",
        )

    def test_indentation_errors(self) -> None:
        self.assert_error(
            "if true:\n\tprint(1)\n",
            "lexer error",
            "Tabs are not allowed",
        )
        self.assert_error(
            "if true:\n    print(1)\n  print(2)\n",
            "lexer error",
            "Invalid indentation level",
        )

    def test_missing_control_flow_colon(self) -> None:
        self.assert_error(
            "if true\n    print(1)\n",
            "parser error",
            "Expected ':' after if condition",
        )

    def test_functions_calls_and_void_functions(self) -> None:
        self.assert_program(
            "fn add(a: int, b: int) -> int:\n    return a + b\n\n"
            "fn log(x: int):\n    print(x)\n\n"
            "print(add(2, 3))\nlog(7)\n",
            "5\n7\n",
        )
        self.assert_program(
            "fn is_positive(x: int) -> bool:\n    return x > 0\n\n"
            "fn explicit_void() -> void:\n    return\n\n"
            "print(is_positive(3))\nexplicit_void()\n",
            "true\n",
        )

    def test_function_call_precedence(self) -> None:
        self.assert_program(
            "fn square(x: int) -> int:\n    return x * x\n\n"
            "print(square(2 + 3) * 4)\n",
            "100\n",
        )

    def test_local_scope_and_outer_assignment(self) -> None:
        self.assert_program(
            "x: int = 10\nfn local() -> int:\n    x: int = 3\n"
            "    return x\nfn update():\n    x = 12\n\n"
            "print(local())\nprint(x)\nupdate()\nprint(x)\n",
            "3\n10\n12\n",
        )

    def test_recursive_function_and_return_propagation(self) -> None:
        self.assert_program(
            "fn fact(n: int) -> int:\n    if n == 0:\n        return 1\n"
            "    else:\n        return n * fact(n - 1)\n\nprint(fact(5))\n",
            "120\n",
        )

    def test_function_argument_errors(self) -> None:
        function = "fn add(a: int, b: int) -> int:\n    return a + b\n\n"
        self.assert_error(
            function + "print(add(1))\n",
            "type error",
            "expects 2 arguments, got 1",
        )
        self.assert_error(
            function + "print(add(true, 1))\n",
            "type error",
            "argument 1 to be int, got bool",
        )

    def test_function_return_errors(self) -> None:
        self.assert_error(
            "fn missing() -> int:\n    value: int = 1\n\nprint(missing())\n",
            "type error",
            "has no return statement",
        )
        self.assert_error(
            "fn wrong() -> bool:\n    return 1\n\nprint(wrong())\n",
            "type error",
            "returns bool but got int",
        )
        self.assert_error(
            "fn nope():\n    return 1\n\nnope()\n",
            "type error",
            "returns void but got int",
        )

    def test_invalid_return_and_undefined_local_assignment(self) -> None:
        self.assert_error(
            "return 1\n",
            "resolve error",
            "Cannot return outside of function",
        )
        self.assert_program(
            "fn local():\n    inferred = 1\n    print(inferred)\n\nlocal()\n",
            "1\n",
        )
        self.assert_error(
            "missing()\n",
            "resolve error",
            "Unknown function 'missing'",
        )

    def test_modules_and_public_variables(self) -> None:
        result = run_project(
            {
                "main.fl": "import math\nfrom config import VERSION\n"
                "print(math.square(5))\nprint(math.cube(3))\nprint(VERSION)\n",
                "math.fl": "pub fn square(x: int) -> int:\n    return x * x\n\n"
                "pub fn cube(x: int) -> int:\n    return x * x * x\n\n"
                "fn hidden(x: int) -> int:\n    return x + 999\n",
                "config.fl": "pub VERSION: int = 3\n",
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "25\n27\n3\n")

    def test_multiple_selective_imports_and_alias(self) -> None:
        module = (
            "pub fn square(x: int) -> int:\n    return x * x\n\n"
            "pub fn cube(x: int) -> int:\n    return x * x * x\n"
        )
        result = run_project(
            {
                "main.fl": "from math import square, cube\nimport math as m\n"
                "print(square(4))\nprint(cube(3))\nprint(m.square(2))\n",
                "math.fl": module,
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "16\n27\n4\n")

    def test_private_module_access_errors(self) -> None:
        module = "fn hidden() -> int:\n    return 99\n"
        member = run_project(
            {"main.fl": "import math\nprint(math.hidden())\n", "math.fl": module}
        )
        self.assertNotEqual(member.returncode, 0)
        self.assertIn("Cannot access private symbol 'hidden' from module 'math'", member.stderr)
        selective = run_project(
            {"main.fl": "from math import hidden\n", "math.fl": module}
        )
        self.assertNotEqual(selective.returncode, 0)
        self.assertIn("Cannot import private symbol", selective.stderr)

    def test_missing_module_and_symbol_errors(self) -> None:
        missing_module = run_project({"main.fl": "import nowhere\n"})
        self.assertNotEqual(missing_module.returncode, 0)
        self.assertIn("Module 'nowhere' not found", missing_module.stderr)
        missing_symbol = run_project(
            {"main.fl": "import math\nprint(math.nope)\n", "math.fl": "pub X: int = 1\n"}
        )
        self.assertNotEqual(missing_symbol.returncode, 0)
        self.assertIn("Symbol 'nope' not found", missing_symbol.stderr)

    def test_duplicate_and_circular_imports(self) -> None:
        duplicate = run_project(
            {
                "main.fl": "import once\nimport once\nprint(once.VALUE)\n",
                "once.fl": "print(9)\npub VALUE: int = 1\n",
            }
        )
        self.assertEqual(duplicate.returncode, 0, duplicate.stderr)
        self.assertEqual(duplicate.stdout, "9\n1\n")
        circular = run_project(
            {
                "main.fl": "import a\n",
                "a.fl": "import b\n",
                "b.fl": "import a\n",
            }
        )
        self.assertNotEqual(circular.returncode, 0)
        self.assertIn("Circular import detected", circular.stderr)

    def test_strings_escapes_comments_and_indexing(self) -> None:
        self.assert_program(
            'name: str = "Flang"\nprint(name)\nprint("hello " + "world")\n'
            'print(len(name))\nprint(name[0])\nprint("# not comment")\n'
            'print("line\\nnext")\nprint("tab\\tquote: \\" slash: \\\\")\n',
            "Flang\nhello world\n5\nF\n# not comment\nline\nnext\n"
            'tab\tquote: " slash: \\\n',
        )

    def test_string_lexer_errors(self) -> None:
        self.assert_error(
            'print("unterminated)\n',
            "lexer error",
            "Unterminated string literal",
        )
        self.assert_error(
            'print("bad\\q")\n',
            "lexer error",
            "Invalid escape sequence",
        )

    def test_arrays_len_index_and_deep_copy_assignment(self) -> None:
        self.assert_program(
            "fn changed(xs: list) -> list:\n    xs[0][0] = 8\n    return xs\n\n"
            "a: list = [[1, 2], [3, 4]]\nb = a\nb[0][0] = 9\nb[1][1] = 7\n"
            "c = changed(a)\nprint(a)\nprint(b)\nprint(c)\n"
            "print(len(b))\nprint(b[1][1])\n",
            "[[1, 2], [3, 4]]\n[[9, 2], [3, 7]]\n[[8, 2], [3, 4]]\n2\n7\n",
        )

    def test_index_errors_and_invalid_assignment_target(self) -> None:
        self.assert_error("print([1][true])\n", "type error", "Index must be int")
        self.assert_error("print([1][2])\n", "runtime error", "Array index out of bounds")
        self.assert_error('print("x"[2])\n', "runtime error", "String index out of bounds")
        self.assert_error("print(1[0])\n", "type error", "not indexable")
        self.assert_error("1 + 2 = 3\n", "parser error", "Invalid assignment target")

    def test_for_array_range_and_loop_scope(self) -> None:
        self.assert_program(
            "sum = 0\nfor n in [1, 2, 3]:\n    sum = sum + n\n"
            "for i in range(0, 3):\n    print(i)\nprint(sum)\n",
            "0\n1\n2\n6\n",
        )
        self.assert_error(
            "for x in 1:\n    print(x)\n",
            "type error",
            "must be list or range",
        )

    def test_data_type_annotations_and_function_types(self) -> None:
        self.assert_program(
            "fn first(xs: list) -> int:\n    return xs[0]\n\n"
            "fn greet(name: str) -> str:\n    return \"hi \" + name\n\n"
            "items: list = [8]\nprint(first(items))\nprint(greet(\"flang\"))\n",
            "8\nhi flang\n",
        )
        self.assert_error(
            'name: str = 1\n',
            "type error",
            "Cannot initialize variable 'name' of type str with value of type int",
        )

    def test_len_and_range_errors(self) -> None:
        self.assert_error("print(len(1))\n", "type error", "len() cannot be used with int")
        self.assert_error("range(0)\n", "type error", "expects 2 arguments")
        self.assert_error("range(false, 2)\n", "type error", "expects int and int")

    def test_tensor_creation_shape_rank_and_indexing(self) -> None:
        self.assert_program(
            "m: tensor = tensor([\n    [1, 2],\n    [3, 4]\n])\n"
            "print(m)\nprint(shape(m))\nprint(rank(m))\n"
            "print(m[0][1])\nprint(m[1][0])\n",
            "[[1, 2], [3, 4]]\n[2, 2]\n2\n2\n3\n",
        )

    def test_tensor_and_scalar_arithmetic(self) -> None:
        self.assert_program(
            "a = tensor([1, 2, 3])\nb = tensor([4, 5, 6])\n"
            "print(a + b)\nprint(a - b)\nprint(a * b)\nprint(b / a)\n"
            "print(a + 10)\nprint(10 + a)\nprint(10 - a)\nprint(12 / a)\n",
            "[5, 7, 9]\n[-3, -3, -3]\n[4, 10, 18]\n[4, 2.5, 2]\n"
            "[11, 12, 13]\n[11, 12, 13]\n[9, 8, 7]\n[12, 6, 4]\n",
        )

    def test_tensor_factories(self) -> None:
        self.assert_program(
            "z = zeros([2, 3])\no = ones([2, 3])\n"
            "print(shape(z))\nprint(rank(o))\nprint(z)\nprint(o)\n",
            "[2, 3]\n2\n[[0, 0, 0], [0, 0, 0]]\n"
            "[[1, 1, 1], [1, 1, 1]]\n",
        )

    def test_tensor_validation_errors(self) -> None:
        self.assert_error(
            "print(tensor([[1, 2], [3]]))\n",
            "type error",
            "Ragged tensor literal",
        )
        self.assert_error(
            "print(tensor([true, false]))\n",
            "type error",
            "Tensor elements must be int",
        )
        self.assert_error(
            "print(tensor([1, 2]) + tensor([1]))\n",
            "type error",
            "Tensor shape mismatch",
        )
        self.assert_error(
            "print(tensor([1, 2]) / tensor([1, 0]))\n",
            "runtime error",
            "Division by zero",
        )
        self.assert_error(
            "print(shape([1]))\n",
            "type error",
            "shape() expects tensor",
        )
        self.assert_error(
            "print(zeros([2, -1]))\n",
            "runtime error",
            "non-negative ints",
        )

    def test_v05_data_properties(self) -> None:
        self.assert_program(
            "x = tensor([[1, 2], [3, 4]])\nnums = [1, 2, 3, 4]\n"
            'name = "flang"\nprint(x.shape)\nprint(x.rank)\nprint(x.len)\n'
            "print(nums.len)\nprint(name.len)\nprint(tensor([1, 2, 3]).shape)\n",
            "[2, 2]\n2\n4\n4\n5\n[3]\n",
        )

    def test_v05_property_chains_and_tensor_shape(self) -> None:
        self.assert_program(
            "a = tensor([1, 2, 3])\nb = tensor([4, 5, 6])\n"
            "print(a + b)\nprint((a * b).shape)\n",
            "[5, 7, 9]\n[3]\n",
        )

    def test_v05_invalid_properties_are_type_errors(self) -> None:
        self.assert_error(
            "x = 5\nprint(x.len)\n",
            "type error",
            "Type int has no property 'len'",
        )
        self.assert_error(
            "x = tensor([1, 2, 3])\nprint(x.foo)\n",
            "type error",
            "Type tensor has no property 'foo'",
        )
        self.assert_error(
            "x = [1]\nprint(x.shape)\n",
            "type error",
            "Type list has no property 'shape'",
        )

    def test_v05_assignment_type_stability(self) -> None:
        self.assert_error(
            'x: int = 5\nx = "hello"\n',
            "type error",
            "Cannot assign str to variable 'x' of type int",
        )
        self.assert_error(
            'x = 5\nx = "hello"\n',
            "type error",
            "Cannot assign str to variable 'x' of type int",
        )

    def test_v05_equality_types(self) -> None:
        self.assert_program(
            'print(true == false)\nprint("a" == "a")\nprint("a" != "b")\n',
            "false\ntrue\ntrue\n",
        )
        self.assert_error(
            'print("a" == 1)\n',
            "type error",
            "operator '==' with str and int",
        )

    def test_v05_duplicate_and_scope_resolution(self) -> None:
        self.assert_error(
            "x: int = 1\nx: int = 2\n",
            "resolve error",
            "Duplicate declaration of 'x'",
        )
        self.assert_error(
            "fn f(a: int, a: int) -> int:\n    return a\n",
            "resolve error",
            "Duplicate declaration of 'a'",
        )
        self.assert_error(
            "fn f(a: int) -> int:\n    return a\nprint(a)\n",
            "resolve error",
            "Unknown variable 'a'",
        )

    def test_v05_static_tensor_shape_mismatch(self) -> None:
        result = run_flang(
            "a = tensor([1, 2, 3])\nb = tensor([1, 2])\nprint(a + b)\n"
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("type error", result.stderr)
        self.assertIn("Tensor shape mismatch for '+'", result.stderr)
        self.assertIn("[3] vs [2]", result.stderr)

    def test_v05_visibility_fails_before_execution(self) -> None:
        result = run_project(
            {
                "main.fl": "import math\nprint(111)\nprint(math.hidden(1))\n",
                "math.fl": "print(999)\nfn hidden(x: int) -> int:\n    return x\n",
            }
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")
        self.assertIn("resolve error", result.stderr)
        self.assertIn("Cannot access private symbol 'hidden'", result.stderr)

    def test_v05_diagnostic_location_and_unknown_function(self) -> None:
        result = run_flang("x = 1\nprint(missing(x))\n")
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stderr, r"main\.fl:2:7: resolve error:")
        self.assertIn("Unknown function 'missing'", result.stderr)

    def test_v05_inferred_function_locals(self) -> None:
        self.assert_program(
            "fn predict(w: int, b: int, xs: tensor) -> tensor:\n"
            "    return xs * w + b\n\n"
            "fn evaluate(pred: tensor, ys: tensor):\n"
            "    error = pred - ys\n"
            "    loss = error * error\n"
            "    print(loss)\n\n"
            "xs = tensor([1, 2, 3, 4])\n"
            "ys = tensor([3, 5, 7, 9])\n"
            "pred = predict(2, 1, xs)\n"
            "evaluate(pred, ys)\n"
            "print(pred)\nprint(pred.shape)\nprint(pred.rank)\n",
            "[0, 0, 0, 0]\n[3, 5, 7, 9]\n[4]\n1\n",
        )

    def test_v05_inferred_local_does_not_escape(self) -> None:
        self.assert_error(
            "fn create():\n    local_value = 3\n\ncreate()\nprint(local_value)\n",
            "resolve error",
            "Unknown variable 'local_value'",
        )

    def test_v05_homogeneous_array_inference(self) -> None:
        self.assert_error(
            'items = [1, "x"]\n',
            "type error",
            "Array elements must have the same type",
        )
        self.assert_error(
            'items = [1, 2]\nitems = ["x", "y"]\n',
            "type error",
            "Cannot assign array[str] to variable 'items' of type array[int]",
        )
        self.assert_error(
            'items = [1, 2]\nitems[0] = "x"\n',
            "type error",
            "Cannot assign str to indexed value of type int",
        )

    def test_v05_friendly_diagnostic_caret_and_hint(self) -> None:
        result = run_flang('x: int = "hello"\n')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('x: int = "hello"', result.stderr)
        self.assertIn("^^^^^^^", result.stderr)
        self.assertIn("hint: change the annotation", result.stderr)

    def test_v06_native_tensor_module_and_properties(self) -> None:
        self.assert_program(
            "from tensor import tensor, zeros, ones, arange\n"
            "x = tensor([1, 2, 3], true)\n"
            "print(x)\nprint(x.shape)\nprint(x.rank)\nprint(x.len)\n"
            "print(x.requires_grad)\nprint(zeros([2]))\nprint(ones([2]))\n"
            "print(arange(2, 6))\n",
            "[1, 2, 3]\n[3]\n1\n3\ntrue\n[0, 0]\n[1, 1]\n[2, 3, 4, 5]\n",
        )

    def test_v06_qualified_native_module_and_reductions(self) -> None:
        self.assert_program(
            "import tensor\n"
            "x = tensor.tensor([[1, 2], [3, 4]])\n"
            "print(x.sum())\nprint(x.mean())\n",
            "10\n2.5\n",
        )

    def test_v06_basic_autograd(self) -> None:
        self.assert_program(
            "from tensor import tensor\n"
            "x = tensor([1, 2, 3], true)\n"
            "loss = (x * x + 2).sum()\nloss.backward()\n"
            "print(loss)\nprint(x.grad)\n",
            "20\n[2, 4, 6]\n",
        )

    def test_v06_autograd_scalar_and_shared_operand_rules(self) -> None:
        self.assert_program(
            "from tensor import tensor\n"
            "a = tensor([2, 4], true)\n"
            "((a + a) / 2).sum().backward()\nprint(a.grad)\n"
            "b = tensor([2, 4], true)\n"
            "(8 / b).sum().backward()\nprint(b.grad)\n",
            "[1, 1]\n[-2, -0.5]\n",
        )

    def test_v06_mean_gradient_through_function(self) -> None:
        self.assert_program(
            "from tensor import tensor\n"
            "fn loss(x: tensor) -> tensor:\n"
            "    return (x * 4).mean()\n\n"
            "x = tensor([1, 2, 3, 4], true)\n"
            "result = loss(x)\nresult.backward()\nprint(x.grad)\n",
            "[1, 1, 1, 1]\n",
        )

    def test_v06_gradient_accumulation_and_zero_grad(self) -> None:
        self.assert_program(
            "from tensor import tensor\n"
            "x = tensor([1, 2, 3], true)\n"
            "loss = (x * x).sum()\n"
            "loss.backward()\nloss.backward()\nprint(x.grad)\n"
            "x.zero_grad()\nprint(x.grad)\n",
            "[4, 8, 12]\n[0, 0, 0]\n",
        )

    def test_v06_backward_errors_are_friendly(self) -> None:
        non_scalar = run_flang(
            "from tensor import tensor\n"
            "x = tensor([1, 2, 3], true)\nx.backward()\n"
        )
        self.assertNotEqual(non_scalar.returncode, 0)
        self.assertIn("backward() requires scalar tensor; got shape [3]",
                      non_scalar.stderr)
        self.assertIn("hint: use tensor.sum().backward()", non_scalar.stderr)
        no_graph = run_flang(
            "from tensor import tensor\n"
            "x = tensor([1])\nx.sum().backward()\n"
        )
        self.assertNotEqual(no_graph.returncode, 0)
        self.assertIn("requires a tensor with a grad graph", no_graph.stderr)

    def test_v06_grad_and_method_validation_errors(self) -> None:
        self.assert_error(
            "from tensor import tensor\nx = tensor([1], 1)\n",
            "type error",
            "requires requires_grad to be bool",
        )
        self.assert_error(
            "from tensor import tensor\nx = tensor([1])\nx.missing()\n",
            "type error",
            "Tensor has no method 'missing'",
        )
        unavailable = run_flang(
            "from tensor import tensor\nx = tensor([1], true)\nprint(x.grad)\n"
        )
        self.assertNotEqual(unavailable.returncode, 0)
        self.assertIn("gradient is not available", unavailable.stderr)

    def test_v06_native_module_symbol_and_shape_errors(self) -> None:
        self.assert_error(
            "from tensor import missing\n",
            "resolve error",
            "Symbol 'missing' not found in module 'tensor'",
        )
        self.assert_error(
            "from tensor import tensor\n"
            "a = tensor([1, 2, 3])\nb = tensor([1, 2])\nprint(a + b)\n",
            "type error",
            "Tensor shape mismatch",
        )

    def test_v06_local_module_shadows_native_module(self) -> None:
        result = run_project(
            {
                "main.fl": "from tensor import tensor\nprint(tensor(3))\n",
                "tensor.fl": "pub fn tensor(x: int) -> int:\n    return x + 10\n",
            }
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "13\n")

    def test_v07_vm_is_default_and_interp_remains_available(self) -> None:
        source = "x = 2 + 3 * 4\nprint(x)\n"
        vm = run_project({"main.fl": source}, args=["--time"])
        self.assertEqual(vm.returncode, 0, vm.stderr)
        self.assertEqual(vm.stdout, "14\n")
        self.assertIn("backend: vm", vm.stderr)
        interp = run_project({"main.fl": source}, args=["--interp"])
        self.assertEqual(interp.returncode, 0, interp.stderr)
        self.assertEqual(interp.stdout, "14\n")

    def test_v07_ir_dump_and_constant_folding(self) -> None:
        source = "x = 2 + 3 * 4\nprint(x)\n"
        optimized = run_project(
            {"main.fl": source}, args=["--dump-ir"]
        )
        self.assertEqual(optimized.returncode, 0, optimized.stderr)
        self.assertIn("IR function __main__", optimized.stdout)
        self.assertIn("CONST                14", optimized.stdout)
        unoptimized = run_project(
            {"main.fl": source}, args=["--dump-ir", "--no-opt"]
        )
        self.assertEqual(unoptimized.returncode, 0, unoptimized.stderr)
        self.assertIn("MUL_INT", unoptimized.stdout)
        self.assertIn("ADD_INT", unoptimized.stdout)

    def test_v07_bytecode_dump_uses_slots_and_typed_ops(self) -> None:
        result = run_project(
            {"main.fl": "x = 5\ny = x + 2\nprint(y)\n"},
            args=["--dump-bytecode"],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OP_STORE_GLOBAL", result.stdout)
        self.assertIn("OP_LOAD_GLOBAL", result.stdout)
        self.assertIn("OP_ADD_INT", result.stdout)
        self.assertTrue(result.stdout.endswith("7\n"))

    def test_v07_tensor_specialized_bytecode(self) -> None:
        result = run_project(
            {
                "main.fl": "from tensor import tensor\n"
                "a = tensor([1, 2])\nb = a * 3 + 1\nprint(b.shape)\n"
            },
            args=["--dump-bytecode"],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("OP_TENSOR_MUL", result.stdout)
        self.assertIn("OP_TENSOR_ADD", result.stdout)
        self.assertIn("OP_GET_PROPERTY", result.stdout)
        self.assertTrue(result.stdout.endswith("[2]\n"))

    def test_v07_vm_runtime_diagnostic_keeps_source_span(self) -> None:
        result = run_project(
            {"main.fl": "x = 0\nprint(10 / x)\n"}, args=["--vm"]
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertRegex(result.stderr, r"main\.fl:2:10: runtime error")
        self.assertIn("Division by zero", result.stderr)
        self.assertIn("print(10 / x)", result.stderr)

    def test_v07_unsupported_construct_uses_interpreter_fallback(self) -> None:
        result = run_project(
            {
                "main.fl": "import helper\nprint(helper.value())\n",
                "helper.fl": "pub fn value() -> int:\n    return 7\n",
            },
            args=["--time"],
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "7\n")
        self.assertIn("backend: interpreter fallback", result.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
