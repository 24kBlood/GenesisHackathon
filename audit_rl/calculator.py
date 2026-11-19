"""Safe AST-based Decimal arithmetic calculator for the financial audit RL environment.

Evaluates arithmetic expressions using ``ast.parse(..., mode="eval")`` without ever
calling ``eval()``. All numeric literals are converted to ``decimal.Decimal`` via
``Decimal(str(val))`` to avoid binary floating-point inaccuracies.
"""

from __future__ import annotations

import ast
from decimal import Decimal, DivisionByZero, InvalidOperation, Overflow, ROUND_HALF_EVEN
from typing import Callable

MAX_EXPRESSION_LENGTH = 256
MAX_AST_DEPTH = 64
MAX_ROUND_DIGITS = 28


class CalculatorError(ValueError):
    """Raised when an expression is invalid, unsafe, or cannot be evaluated."""


def _safe_round(args: list[Decimal]) -> Decimal:
    if len(args) == 1:
        try:
            return args[0].quantize(Decimal("1"), rounding=ROUND_HALF_EVEN)
        except (InvalidOperation, Overflow) as exc:
            raise CalculatorError(f"Rounding failed: {exc}") from exc
    if len(args) == 2:
        value, digits_dec = args
        if digits_dec != digits_dec.to_integral_value():
            raise CalculatorError("round() second argument must be an integer")
        digits = int(digits_dec)
        if abs(digits) > MAX_ROUND_DIGITS:
            raise CalculatorError(
                f"round() precision must be between -{MAX_ROUND_DIGITS} and {MAX_ROUND_DIGITS}"
            )
        quantizer = Decimal(1).scaleb(-digits)
        try:
            return value.quantize(quantizer, rounding=ROUND_HALF_EVEN)
        except (InvalidOperation, Overflow) as exc:
            raise CalculatorError(f"Rounding failed: {exc}") from exc
    raise CalculatorError("round() takes 1 or 2 arguments")


def _safe_abs(args: list[Decimal]) -> Decimal:
    if len(args) != 1:
        raise CalculatorError("abs() takes exactly 1 argument")
    return abs(args[0])


def _safe_min(args: list[Decimal]) -> Decimal:
    if len(args) < 1:
        raise CalculatorError("min() requires at least 1 argument")
    return min(args)


def _safe_max(args: list[Decimal]) -> Decimal:
    if len(args) < 1:
        raise CalculatorError("max() requires at least 1 argument")
    return max(args)


_SAFE_FUNCTIONS: dict[str, Callable[[list[Decimal]], Decimal]] = {
    "abs": _safe_abs,
    "min": _safe_min,
    "max": _safe_max,
    "round": _safe_round,
}


def _eval_node(node: ast.AST, depth: int) -> Decimal:
    if depth > MAX_AST_DEPTH:
        raise CalculatorError("Expression nesting depth exceeds limit")

    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CalculatorError(f"Unsupported literal type: {type(value).__name__}")
        try:
            dec = Decimal(str(value))
        except InvalidOperation as exc:
            raise CalculatorError(f"Invalid numeric literal: {value!r}") from exc
        if dec.is_nan() or dec.is_infinite():
            raise CalculatorError("Non-finite numeric literal is not allowed")
        return dec

    if isinstance(node, ast.UnaryOp):
        operand = _eval_node(node.operand, depth + 1)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise CalculatorError(f"Unsupported unary operator: {type(node.op).__name__}")

    if isinstance(node, ast.BinOp):
        left = _eval_node(node.left, depth + 1)
        right = _eval_node(node.right, depth + 1)
        op = node.op
        try:
            if isinstance(op, ast.Add):
                result = left + right
            elif isinstance(op, ast.Sub):
                result = left - right
            elif isinstance(op, ast.Mult):
                result = left * right
            elif isinstance(op, ast.Div):
                if right == 0:
                    raise CalculatorError("Division by zero")
                result = left / right
            elif isinstance(op, ast.FloorDiv):
                if right == 0:
                    raise CalculatorError("Division by zero")
                result = left // right
            elif isinstance(op, ast.Mod):
                if right == 0:
                    raise CalculatorError("Division by zero")
                result = left % right
            else:
                raise CalculatorError(f"Unsupported binary operator: {type(op).__name__}")
        except CalculatorError:
            raise
        except (DivisionByZero, InvalidOperation, Overflow, ArithmeticError) as exc:
            raise CalculatorError(f"Arithmetic error: {exc}") from exc

        if result.is_nan() or result.is_infinite():
            raise CalculatorError("Arithmetic result is not finite")
        return result

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise CalculatorError("Only direct calls to whitelisted functions are allowed")
        func_name = node.func.id
        if func_name not in _SAFE_FUNCTIONS:
            raise CalculatorError(f"Unsupported function: {func_name!r}")
        if node.keywords:
            raise CalculatorError("Keyword arguments are not supported")
        eval_args: list[Decimal] = []
        for arg in node.args:
            if isinstance(arg, ast.Starred):
                raise CalculatorError("Starred arguments are not supported")
            eval_args.append(_eval_node(arg, depth + 1))
        result = _SAFE_FUNCTIONS[func_name](eval_args)
        if result.is_nan() or result.is_infinite():
            raise CalculatorError("Function result is not finite")
        return result

    raise CalculatorError(f"Unsupported expression element: {type(node).__name__}")


def _format_decimal(value: Decimal) -> str:
    if value == 0:
        return "0"
    normalized = value.normalize()
    text = format(normalized, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    if text in ("", "-0"):
        return "0"
    return text


def safe_calculate(expression: str) -> str:
    """Safely evaluate a decimal arithmetic expression and return a normalized string."""
    if not isinstance(expression, str):
        raise CalculatorError("Expression must be a string")
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise CalculatorError(
            f"Expression length {len(expression)} exceeds maximum of {MAX_EXPRESSION_LENGTH} characters"
        )
    stripped = expression.strip()
    if not stripped:
        raise CalculatorError("Expression must not be empty")

    try:
        parsed = ast.parse(stripped, mode="eval")
    except SyntaxError as exc:
        raise CalculatorError(f"Invalid syntax: {exc.msg}") from exc

    result = _eval_node(parsed.body, depth=0)
    return _format_decimal(result)
