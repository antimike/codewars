"""
The purpose of this kata is to write a program that can do some algebra.

Write a function expand that takes in an expression with a single, one character variable, and expands it. The expression is in the form (ax+b)^n where a and b are integers which may be positive or negative, x is any single character variable, and n is a natural number. If a = 1, no coefficient will be placed in front of the variable. If a = -1, a "-" will be placed in front of the variable.

The expanded form should be returned as a string in the form ax^b+cx^d+ex^f... where a, c, and e are the coefficients of the term, x is the original one character variable that was passed in the original expression and b, d, and f, are the powers that x is being raised to in each term and are in decreasing order.

If the coefficient of a term is zero, the term should not be included. If the coefficient of a term is one, the coefficient should not be included. If the coefficient of a term is -1, only the "-" should be included. If the power of the term is 0, only the coefficient should be included. If the power of the term is 1, the caret and power should be excluded.
"""
import re
from math import factorial


def binom(m, n):
    return factorial(m) // (factorial(n) * factorial(m - n))


def format_polynomial(coeffs, var):
    terms = []
    for e, c in enumerate(coeffs):
        if c == 0:
            terms.append("")
            continue
        elif e == 0:
            terms.append(f"{c:+}")
            continue
        else:
            term = f"{c:+}"
        if c == 1 or c == -1:
            term = term[0]
        term += f"{var}"
        if e > 1:
            term += f"^{e}"
        terms.append(term)
    return "".join(reversed(terms)).removeprefix("+")


def _parse_expr(expr):
    pattern = re.compile(r"\((-?\d*)(\w)([-+]\d+)?\)\^(\d+)")
    a, var, b, e = pattern.match(expr).groups()
    if a in ("", "-"):
        a = int(f"{a}1")
    else:
        a = int(a)
    b = 0 if b is None else int(b)
    e = int(e)
    return a, var, b, e


def expand(expr):
    a, var, b, n = _parse_expr(expr)
    if a == 0:
        return str(b**n)
    elif b == 0:
        c = a**n
        if c == 1 or c == -1:
            return f"{c:+}"[0].removeprefix("+") + f"{var}^{n}"
        return f"{c}{var}^{n}"
    coeffs = [a**j * b ** (n - j) * binom(n, j) for j in range(n + 1)]
    return format_polynomial(coeffs, var)
