"""
formula_parser.py — Restricted DSL Parser for Alpha Formulas

Parses formula strings like:
  rank(sub(div(close_trade_px, vwap), 1))
  ts_zscore(close_trade_px, 20)
  rank(volume) + rank(dvolume)

Into an AST that can be validated and executed.
"""
import re
from dataclasses import dataclass, field
from typing import List, Optional, Union


# ── AST Nodes ──
@dataclass
class NumberNode:
    value: float
    pos: int = 0

@dataclass
class FieldNode:
    name: str
    pos: int = 0

@dataclass
class FuncCallNode:
    name: str
    args: List  # List of AST nodes
    pos: int = 0

@dataclass
class BinaryOpNode:
    op: str   # +, -, *, /
    left: object
    right: object
    pos: int = 0

@dataclass
class UnaryOpNode:
    op: str  # -
    operand: object
    pos: int = 0


class ParseError(Exception):
    def __init__(self, msg, pos=None):
        self.pos = pos
        super().__init__(f"Parse error at position {pos}: {msg}" if pos else f"Parse error: {msg}")


class Tokenizer:
    """Tokenize a formula string into tokens."""

    TOKEN_SPEC = [
        ('NUMBER',   r'-?\d+\.?\d*'),
        ('IDENT',    r'[a-zA-Z_][a-zA-Z0-9_]*'),
        ('LPAREN',   r'\('),
        ('RPAREN',   r'\)'),
        ('COMMA',    r','),
        ('PLUS',     r'\+'),
        ('MINUS',    r'-'),
        ('STAR',     r'\*'),
        ('SLASH',    r'/'),
        ('SKIP',     r'[ \t\n]+'),
    ]

    def __init__(self, text):
        self.text = text
        self.tokens = []
        self.pos = 0
        self._tokenize()

    def _tokenize(self):
        pattern = '|'.join(f'(?P<{name}>{pat})' for name, pat in self.TOKEN_SPEC)
        regex = re.compile(pattern)
        pos = 0
        for m in regex.finditer(self.text):
            if m.start() != pos:
                raise ParseError(f"Unexpected character '{self.text[pos]}'", pos)
            kind = m.lastgroup
            value = m.group()
            pos = m.end()
            if kind == 'SKIP':
                continue
            self.tokens.append((kind, value, m.start()))
        if pos != len(self.text):
            raise ParseError(f"Unexpected character '{self.text[pos]}'", pos)

    def peek(self, offset=0):
        idx = self.pos + offset
        if idx >= len(self.tokens):
            return None
        return self.tokens[idx]

    def consume(self, expected_kind=None):
        if self.pos >= len(self.tokens):
            raise ParseError("Unexpected end of formula")
        tok = self.tokens[self.pos]
        if expected_kind and tok[0] != expected_kind:
            raise ParseError(f"Expected {expected_kind}, got {tok[0]} '{tok[1]}'", tok[2])
        self.pos += 1
        return tok

    def at_end(self):
        return self.pos >= len(self.tokens)


class FormulaParser:
    """
    Recursive descent parser for the restricted DSL.
    
    Grammar:
        expr     → term (('+' | '-') term)*
        term     → unary (('*' | '/') unary)*
        unary    → '-' unary | primary
        primary  → NUMBER | func_call | IDENT | '(' expr ')'
        func_call → IDENT '(' arg_list ')'
        arg_list  → expr (',' expr)*
    """

    def __init__(self, formula_text):
        self.text = formula_text
        self.tokenizer = Tokenizer(formula_text)

    def parse(self):
        """Parse the formula and return an AST node."""
        if not self.text.strip():
            raise ParseError("Empty formula")
        ast = self._expr()
        if not self.tokenizer.at_end():
            tok = self.tokenizer.peek()
            raise ParseError(f"Unexpected token '{tok[1]}'", tok[2])
        return ast

    def _expr(self):
        left = self._term()
        while True:
            tok = self.tokenizer.peek()
            if tok and tok[0] in ('PLUS', 'MINUS'):
                op_tok = self.tokenizer.consume()
                right = self._term()
                left = BinaryOpNode(op=op_tok[1], left=left, right=right, pos=op_tok[2])
            else:
                break
        return left

    def _term(self):
        left = self._unary()
        while True:
            tok = self.tokenizer.peek()
            if tok and tok[0] in ('STAR', 'SLASH'):
                op_tok = self.tokenizer.consume()
                right = self._unary()
                left = BinaryOpNode(op=op_tok[1], left=left, right=right, pos=op_tok[2])
            else:
                break
        return left

    def _unary(self):
        tok = self.tokenizer.peek()
        if tok and tok[0] == 'MINUS':
            op_tok = self.tokenizer.consume()
            operand = self._unary()
            return UnaryOpNode(op='-', operand=operand, pos=op_tok[2])
        return self._primary()

    def _primary(self):
        tok = self.tokenizer.peek()
        if tok is None:
            raise ParseError("Unexpected end of formula")

        # Number literal
        if tok[0] == 'NUMBER':
            num_tok = self.tokenizer.consume()
            return NumberNode(value=float(num_tok[1]), pos=num_tok[2])

        # Identifier or function call
        if tok[0] == 'IDENT':
            # Check if next token is '(' → function call
            next_tok = self.tokenizer.peek(1)
            if next_tok and next_tok[0] == 'LPAREN':
                return self._func_call()
            else:
                ident_tok = self.tokenizer.consume()
                return FieldNode(name=ident_tok[1], pos=ident_tok[2])

        # Parenthesized expression
        if tok[0] == 'LPAREN':
            self.tokenizer.consume('LPAREN')
            expr = self._expr()
            self.tokenizer.consume('RPAREN')
            return expr

        raise ParseError(f"Unexpected token '{tok[1]}'", tok[2])

    def _func_call(self):
        name_tok = self.tokenizer.consume('IDENT')
        self.tokenizer.consume('LPAREN')
        args = []
        if self.tokenizer.peek() and self.tokenizer.peek()[0] != 'RPAREN':
            args.append(self._expr())
            while self.tokenizer.peek() and self.tokenizer.peek()[0] == 'COMMA':
                self.tokenizer.consume('COMMA')
                args.append(self._expr())
        self.tokenizer.consume('RPAREN')
        return FuncCallNode(name=name_tok[1], args=args, pos=name_tok[2])


def parse_formula(formula_text):
    """Parse a formula string into an AST. Raises ParseError on failure."""
    parser = FormulaParser(formula_text)
    return parser.parse()


def ast_to_string(node):
    """Convert AST back to a formula string."""
    if isinstance(node, NumberNode):
        v = node.value
        return str(int(v)) if v == int(v) else str(v)
    elif isinstance(node, FieldNode):
        return node.name
    elif isinstance(node, FuncCallNode):
        args = ', '.join(ast_to_string(a) for a in node.args)
        return f"{node.name}({args})"
    elif isinstance(node, BinaryOpNode):
        return f"({ast_to_string(node.left)} {node.op} {ast_to_string(node.right)})"
    elif isinstance(node, UnaryOpNode):
        return f"(-{ast_to_string(node.operand)})"
    return str(node)


def collect_fields(node):
    """Collect all field references from an AST."""
    fields = set()
    if isinstance(node, FieldNode):
        fields.add(node.name)
    elif isinstance(node, FuncCallNode):
        for arg in node.args:
            fields |= collect_fields(arg)
    elif isinstance(node, BinaryOpNode):
        fields |= collect_fields(node.left)
        fields |= collect_fields(node.right)
    elif isinstance(node, UnaryOpNode):
        fields |= collect_fields(node.operand)
    return fields


def collect_operators(node):
    """Collect all operator/function names from an AST."""
    ops = set()
    if isinstance(node, FuncCallNode):
        ops.add(node.name)
        for arg in node.args:
            ops |= collect_operators(arg)
    elif isinstance(node, BinaryOpNode):
        ops |= collect_operators(node.left)
        ops |= collect_operators(node.right)
    elif isinstance(node, UnaryOpNode):
        ops |= collect_operators(node.operand)
    return ops
