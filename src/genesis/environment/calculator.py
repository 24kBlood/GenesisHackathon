"""Limited decimal arithmetic for the public audit tool."""

import re
from decimal import Context, Decimal, DecimalException, DivisionByZero, InvalidOperation, Overflow, ROUND_HALF_EVEN, Subnormal


MAX_LENGTH = 256
MAX_DEPTH = 16
MAX_OPERATIONS = 64
MAX_LITERAL_DIGITS = 34
MAX_RESULT_LENGTH = 96
_TOKEN = re.compile(r"[ \t\r\n]*([0-9]+(?:\.[0-9]*)?|\.[0-9]+|[()+*/-])")
_CONTEXT = Context(
    prec=34, rounding=ROUND_HALF_EVEN, Emin=-48, Emax=47, clamp=0,
    traps=[DivisionByZero, InvalidOperation, Overflow, Subnormal],
)


class _Parser:
    def __init__(self, tokens: list[str]):
        self.tokens = iter(tokens)
        self.token = next(self.tokens, "")
        self.operations = 0
        self.context = _CONTEXT.copy()

    def pop(self) -> str:
        token = self.token
        self.token = next(self.tokens, "")
        return token

    def count_operation(self) -> None:
        self.operations += 1
        if self.operations > MAX_OPERATIONS:
            raise OverflowError("too many operations")

    def expression(self, depth: int = 0) -> Decimal:
        value = self.term(depth)
        while self.token in ("+", "-"):
            operator = self.pop()
            self.count_operation()
            rhs = self.term(depth)
            value = self.context.add(value, rhs) if operator == "+" else self.context.subtract(value, rhs)
        return value

    def term(self, depth: int) -> Decimal:
        value = self.factor(depth)
        while self.token in ("*", "/"):
            operator = self.pop()
            self.count_operation()
            rhs = self.factor(depth)
            if operator == "*":
                value = self.context.multiply(value, rhs)
            else:
                if rhs.is_zero():
                    raise ZeroDivisionError("division by zero")
                value = self.context.divide(value, rhs)
        return value

    def factor(self, depth: int) -> Decimal:
        if depth > MAX_DEPTH:
            raise OverflowError("expression too deep")
        token = self.pop()
        if token in ("+", "-"):
            self.count_operation()
            value = self.factor(depth + 1)
            return value.copy_negate() if token == "-" else value
        if token == "(":
            value = self.expression(depth + 1)
            if self.pop() != ")":
                raise ValueError("missing closing parenthesis")
            return value
        if not token or token in ")*/":
            raise ValueError("expected number")
        if len(token.replace(".", "")) > MAX_LITERAL_DIGITS:
            raise OverflowError("too many decimal digits")
        return self.context.create_decimal(token)


def calculate(expression: str) -> str:
    """Return a finite, fixed-point decimal string or raise a bounded error type."""
    if len(expression) > MAX_LENGTH:
        raise OverflowError("expression too long")
    expression = expression.rstrip(" \t\r\n")
    tokens = []
    index = 0
    while index < len(expression):
        match = _TOKEN.match(expression, index)
        if match is None:
            raise ValueError("invalid character")
        tokens.append(match[1])
        index = match.end()
    parser = _Parser(tokens)
    try:
        result = parser.expression()
    except DivisionByZero:
        raise ZeroDivisionError("division by zero") from None
    except DecimalException:
        raise OverflowError("decimal result outside limits") from None
    if parser.token:
        raise ValueError("unexpected trailing token")
    if result.is_zero():
        return "0"
    rendered = format(result, "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    if len(rendered) > MAX_RESULT_LENGTH:
        raise OverflowError("result too long")
    return rendered
