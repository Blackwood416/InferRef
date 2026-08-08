"""Relation expression language for contract shape/dtype invariants.

Contract Schema v0.1 section 9. The language is deliberately tiny and is
parsed with a recursive-descent parser into a restricted AST; evaluation is a
plain AST walk and never uses ``eval``/``exec``.

Grammar::

    expr        := or_expr
    or_expr     := and_expr ("or" and_expr)*
    and_expr    := not_expr ("and" not_expr)*
    not_expr    := "not" not_expr | comparison
    comparison  := operand ("==" | "!=") operand
    operand     := path | length | integer | string
    path        := NAME ("." attr)? ("[" index "]")*
    attr        := "shape" | "dtype" | "rank" | "numel"
    length      := "len" "(" NAME "." "shape" ")"
    index       := integer
    integer     := "-"? DIGIT+
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class RelationError(ValueError):
    """Base class for relation parse and role errors."""


class RelationSyntaxError(RelationError):
    """The expression does not conform to the section 9.1 grammar."""


class RelationRoleError(RelationError):
    """The expression references an undeclared or non-tensor role."""


# AST node shapes:
#   ("and", left, right) / ("or", left, right)
#   ("not", child)
#   ("eq", left, right) / ("neq", left, right)
#   ("path", name, attr | None, indexes tuple)
#   ("len", name)
#   ("int", value)
#   ("str", value)

_ATTRS = frozenset({"shape", "dtype", "rank", "numel"})


class _Token:
    __slots__ = ("kind", "position", "value")

    def __init__(self, kind: str, value: str, position: int) -> None:
        self.kind = kind
        self.value = value
        self.position = position

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_Token({self.kind!r}, {self.value!r}, {self.position})"


def _tokenize(expression: str) -> list[_Token]:
    tokens: list[_Token] = []
    index = 0
    length = len(expression)
    while index < length:
        char = expression[index]
        if char.isspace():
            index += 1
            continue
        start = index
        if char == "(":
            tokens.append(_Token("LPAREN", char, index))
            index += 1
        elif char == ")":
            tokens.append(_Token("RPAREN", char, index))
            index += 1
        elif char == "[":
            tokens.append(_Token("LBRACKET", char, index))
            index += 1
        elif char == "]":
            tokens.append(_Token("RBRACKET", char, index))
            index += 1
        elif char == ".":
            tokens.append(_Token("DOT", char, index))
            index += 1
        elif char == "=":
            if index + 1 < length and expression[index + 1] == "=":
                tokens.append(_Token("EQ", "==", index))
                index += 2
            else:
                raise RelationSyntaxError(
                    f"invalid character '=' at position {index}"
                )
        elif char == "!":
            if index + 1 < length and expression[index + 1] == "=":
                tokens.append(_Token("NEQ", "!=", index))
                index += 2
            else:
                raise RelationSyntaxError(
                    f"invalid character '!' at position {index}"
                )
        elif char == '"':
            index += 1
            while index < length and expression[index] != '"':
                index += 1
            if index >= length:
                raise RelationSyntaxError(
                    f"unterminated string literal at position {start}"
                )
            tokens.append(_Token("STRING", expression[start + 1 : index], start))
            index += 1
        elif char.isdigit() or (char == "-" and index + 1 < length and expression[index + 1].isdigit()):
            index += 1
            if char == "-":
                index += 1
            while index < length and expression[index].isdigit():
                index += 1
            tokens.append(_Token("INTEGER", expression[start:index], start))
        elif char.isalpha() or char == "_":
            index += 1
            while index < length and (expression[index].isalnum() or expression[index] == "_"):
                index += 1
            tokens.append(_Token("NAME", expression[start:index], start))
        else:
            raise RelationSyntaxError(
                f"unexpected character {char!r} at position {index}"
            )
    tokens.append(_Token("EOF", "", length))
    return tokens


class _Parser:
    def __init__(self, expression: str) -> None:
        self.expression = expression
        self.tokens = _tokenize(expression)
        self.position = 0

    def _peek(self) -> _Token:
        return self.tokens[self.position]

    def _next(self) -> _Token:
        token = self.tokens[self.position]
        self.position += 1
        return token

    def _expect(self, kind: str) -> _Token:
        token = self._next()
        if token.kind != kind:
            raise RelationSyntaxError(
                f"expected {kind} at position {token.position}, got "
                f"{token.value if token.kind != 'EOF' else 'end of expression'}"
            )
        return token

    def _accept_name(self, name: str) -> bool:
        token = self._peek()
        if token.kind == "NAME" and token.value == name:
            self.position += 1
            return True
        return False

    def parse(self) -> tuple[Any, ...]:
        node = self._parse_or()
        if self._peek().kind != "EOF":
            token = self._peek()
            raise RelationSyntaxError(
                f"unexpected {token.value!r} at position {token.position}"
            )
        return node

    def _parse_or(self) -> tuple[Any, ...]:
        node = self._parse_and()
        while self._accept_name("or"):
            node = ("or", node, self._parse_and())
        return node

    def _parse_and(self) -> tuple[Any, ...]:
        node = self._parse_not()
        while self._accept_name("and"):
            node = ("and", node, self._parse_not())
        return node

    def _parse_not(self) -> tuple[Any, ...]:
        if self._accept_name("not"):
            return ("not", self._parse_not())
        if self._peek().kind == "LPAREN":
            self._next()
            node = self._parse_or()
            self._expect("RPAREN")
            return node
        return self._parse_comparison()

    def _parse_comparison(self) -> tuple[Any, ...]:
        left = self._parse_operand()
        token = self._peek()
        if token.kind == "EQ":
            self._next()
            return ("eq", left, self._parse_operand())
        if token.kind == "NEQ":
            self._next()
            return ("neq", left, self._parse_operand())
        raise RelationSyntaxError(
            f"expected '==' or '!=' at position {token.position}"
        )

    def _parse_operand(self) -> tuple[Any, ...]:
        token = self._peek()
        if token.kind == "INTEGER":
            self._next()
            return ("int", int(token.value))
        if token.kind == "STRING":
            self._next()
            return ("str", token.value)
        if token.kind != "NAME":
            raise RelationSyntaxError(
                f"expected a role, length, integer or string at position {token.position}"
            )
        if token.value == "len" and self.tokens[self.position + 1].kind == "LPAREN":
            self._next()  # len
            self._next()  # (
            name = self._expect("NAME")
            self._expect("DOT")
            attr = self._expect("NAME")
            if attr.value != "shape":
                raise RelationSyntaxError(
                    f"len() only supports NAME.shape at position {attr.position}"
                )
            self._expect("RPAREN")
            return ("len", name.value)
        self._next()
        node: tuple[Any, ...] = ("path", token.value, None, ())
        if self._peek().kind == "DOT":
            self._next()
            attr = self._expect("NAME")
            if attr.value not in _ATTRS:
                raise RelationSyntaxError(
                    f"unsupported attribute {attr.value!r} at position {attr.position}; "
                    f"supported: {', '.join(sorted(_ATTRS))}"
                )
            node = ("path", token.value, attr.value, ())
        indexes: list[int] = []
        while self._peek().kind == "LBRACKET":
            self._next()
            integer = self._expect("INTEGER")
            indexes.append(int(integer.value))
            self._expect("RBRACKET")
        if indexes:
            node = ("path", token.value, node[2], tuple(indexes))
        return node


def parse(expression: str) -> tuple[Any, ...]:
    """Parse one relation expression into an AST tuple.

    Raises :class:`RelationSyntaxError` when the expression does not parse.
    """

    if not isinstance(expression, str):
        raise RelationSyntaxError(f"relation must be a string, got {type(expression).__name__}")
    try:
        return _Parser(expression).parse()
    except RelationSyntaxError:
        raise
    except Exception as exc:  # pragma: no cover - defensive parser safety
        raise RelationSyntaxError(f"cannot parse relation: {exc}") from exc


def relation_roles(expression: str) -> tuple[str, ...]:
    """Return every role name referenced by a parsed expression."""

    ast = parse(expression)
    found: set[str] = set()

    def walk(node: tuple[Any, ...]) -> None:
        kind = node[0]
        if kind in {"path", "len"}:
            found.add(node[1])
        else:
            for child in node[1:]:
                if isinstance(child, tuple):
                    walk(child)

    walk(ast)
    return tuple(sorted(found))


def render_operand(node: tuple[Any, ...]) -> str:
    """Render an operand node back to its source-like text."""

    kind = node[0]
    if kind == "int":
        return str(node[1])
    if kind == "str":
        return repr(node[1])
    if kind == "len":
        return f"len({node[1]}.shape)"
    if kind == "path":
        text = node[1]
        if node[2] is not None:
            text += f".{node[2]}"
        for index in node[3]:
            text += f"[{index}]"
        return text
    raise ValueError(f"not an operand node: {node!r}")  # pragma: no cover


def _format_value(value: Any) -> str:
    if isinstance(value, list):
        return "[" + ",".join(str(dimension) for dimension in value) + "]"
    if isinstance(value, str):
        return repr(value)
    return str(value)


def _resolve_path(
    node: tuple[Any, ...],
    roles: Mapping[str, Mapping[str, Any]],
) -> Any:
    name = node[1]
    record = roles.get(name)
    if not isinstance(record, dict):
        raise KeyError(f"role {name!r} is not bound")
    attr = node[2]
    if attr is None:
        if node[3]:
            raise KeyError(f"role {name!r} cannot be indexed directly")
        return record
    if attr == "dtype":
        value = record.get("dtype")
        if not isinstance(value, str) or not value:
            raise KeyError(f"{name}.dtype is not declared")
        return value
    shape = record.get("shape")
    if not isinstance(shape, list) or not all(
        isinstance(dim, int) and not isinstance(dim, bool) for dim in shape
    ):
        raise KeyError(f"{name}.shape is not a valid tensor shape")
    if attr == "shape":
        if not node[3]:
            return list(shape)
        index = node[3][0]
        try:
            return shape[index]
        except IndexError:
            raise KeyError(f"{name}.shape[{index}] is out of range") from None
    if attr == "rank":
        return len(shape)
    if attr == "numel":
        total = 1
        for dimension in shape:
            total *= dimension
        return total
    raise KeyError(f"unsupported attribute {attr!r}")  # pragma: no cover


def _evaluate(
    node: tuple[Any, ...],
    roles: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, tuple[Any, ...] | None]:
    """Evaluate an AST node. Returns ``(value, failing_comparison)``."""

    kind = node[0]
    if kind in {"int", "str"}:
        return bool(node[1]), None
    if kind == "path":
        try:
            value = _resolve_path(node, roles)
        except KeyError as exc:
            raise RelationEvaluationError(str(exc)) from exc
        return bool(value), None
    if kind == "len":
        record = roles.get(node[1])
        if not isinstance(record, dict):
            raise RelationEvaluationError(f"role {node[1]!r} is not bound")
        shape = record.get("shape")
        if not isinstance(shape, list) or not all(
            isinstance(dim, int) and not isinstance(dim, bool) for dim in shape
        ):
            raise RelationEvaluationError(f"{node[1]}.shape is not a valid tensor shape")
        return bool(len(shape)), None
    if kind in {"eq", "neq"}:
        left = _resolve_operand(node[1], roles)
        right = _resolve_operand(node[2], roles)
        result = left == right if kind == "eq" else left != right
        if result:
            return True, None
        return False, (node[1], left, node[2], right)
    if kind == "not":
        child_value, _ = _evaluate(node[1], roles)
        return not child_value, None
    if kind == "and":
        left_value, left_detail = _evaluate(node[1], roles)
        if not left_value:
            return False, left_detail
        return _evaluate(node[2], roles)
    if kind == "or":
        left_value, _ = _evaluate(node[1], roles)
        if left_value:
            return True, None
        return _evaluate(node[2], roles)
    raise RelationEvaluationError(f"unknown relation node {kind!r}")  # pragma: no cover


def _resolve_operand(
    node: tuple[Any, ...],
    roles: Mapping[str, Mapping[str, Any]],
) -> Any:
    kind = node[0]
    if kind == "int":
        return node[1]
    if kind == "str":
        return node[1]
    if kind == "path":
        try:
            return _resolve_path(node, roles)
        except KeyError as exc:
            raise RelationEvaluationError(str(exc)) from exc
    if kind == "len":
        record = roles.get(node[1])
        if not isinstance(record, dict):
            raise RelationEvaluationError(f"role {node[1]!r} is not bound")
        shape = record.get("shape")
        if not isinstance(shape, list) or not all(
            isinstance(dim, int) and not isinstance(dim, bool) for dim in shape
        ):
            raise RelationEvaluationError(f"{node[1]}.shape is not a valid tensor shape")
        return len(shape)
    raise RelationEvaluationError(f"not an operand node: {node!r}")  # pragma: no cover


class RelationEvaluationError(RelationError):
    """A role record could not be resolved while evaluating a relation."""


def evaluate(
    expression: str,
    roles: Mapping[str, Mapping[str, Any]],
) -> tuple[bool, str | None]:
    """Evaluate one relation against a role-name -> tensor-record mapping.

    Returns ``(holds, failure_message)``. ``failure_message`` is deterministic
    and includes the expression plus both operands of the failing comparison,
    e.g.::

        relation 'y.shape == x.shape' failed (y.shape=[2,3,5], x.shape=[2,3,6])

    Raises :class:`RelationSyntaxError` for malformed expressions and
    :class:`RelationEvaluationError` when a referenced role cannot be resolved.
    """

    ast = parse(expression)
    holds, detail = _evaluate(ast, roles)
    if holds:
        return True, None
    if detail is None:
        return False, f"relation {expression!r} failed"
    left_node, left_value, right_node, right_value = detail
    message = (
        f"relation {expression!r} failed ({render_operand(left_node)}="
        f"{_format_value(left_value)}, {render_operand(right_node)}="
        f"{_format_value(right_value)})"
    )
    return False, message
