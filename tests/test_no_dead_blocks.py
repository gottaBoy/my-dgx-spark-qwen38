"""Catch code that compiles, passes every test, and can never run.

A duplicated paste left an entire second function body after `return` inside
another function. Python reads it as unreachable statements and a bare string
expression, so it compiles clean, imports clean, and no behaviour test can see it.
The only reliable detector is a syntactic one, so this is a syntactic check.
"""

from __future__ import annotations

import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TERMINATORS = (ast.Return, ast.Raise, ast.Continue, ast.Break)


def unreachable_statements(tree: ast.AST) -> list[tuple[int, str]]:
    """Statements that follow a return/raise/break/continue in the same block."""
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if not isinstance(block, list) or not block:
                continue
            for index, statement in enumerate(block[:-1]):
                if isinstance(statement, TERMINATORS):
                    dead = block[index + 1]
                    label = ast.unparse(dead).splitlines()[0][:70]
                    found.append((getattr(dead, "lineno", 0), label))
    return found


class TestNoUnreachableCode(unittest.TestCase):
    def _sources(self):
        for path in sorted([*(ROOT / "lib" / "qwen38").glob("*.py"),
                            *(ROOT / "tests").glob("*.py"),
                            ROOT / "bin" / "qwen38"]):
            if path.is_file():
                yield path

    def test_no_statement_follows_a_return(self):
        offenders = []
        for path in self._sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for lineno, label in unreachable_statements(tree):
                offenders.append(f"{path.relative_to(ROOT)}:{lineno}: unreachable {label!r}")
        self.assertEqual(offenders, [], "\n" + "\n".join(offenders))

    def test_the_detector_actually_detects(self):
        # A guard that cannot fail is worse than no guard: it reads as coverage.
        sample = ast.parse("def f():\n    return 1\n    print('never')\n")
        self.assertEqual(len(unreachable_statements(sample)), 1)
        clean = ast.parse("def f():\n    if x:\n        return 1\n    return 2\n")
        self.assertEqual(unreachable_statements(clean), [])

    def test_no_duplicate_function_definitions(self):
        """Two `def` with one name in a file means the first is silently shadowed."""
        offenders = []
        for path in self._sources():
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            seen: dict[str, int] = {}
            for node in tree.body:               # module level only
                name = getattr(node, "name", None)
                if name:
                    if name in seen:
                        offenders.append(
                            f"{path.relative_to(ROOT)}:{node.lineno}: {name} "
                            f"already defined at line {seen[name]}")
                    seen[name] = node.lineno
        self.assertEqual(offenders, [], "\n" + "\n".join(offenders))


if __name__ == "__main__":
    unittest.main()
