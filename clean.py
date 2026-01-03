#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import ast
import tokenize
from pathlib import Path


# =========================
# ✅ 硬编码：输入文件夹（你来填）
# =========================
IN_DIR = Path(r"C:\Users\Ze\Desktop\Data_and_code\Data_code").expanduser().resolve()

# 输出文件夹：默认在输入目录旁边新建一个 *_nocomments
OUT_DIR = IN_DIR.parent / f"{IN_DIR.name}_nocomments"


class RemoveDocstrings(ast.NodeTransformer):
    """Remove module/class/function/async-function docstrings and keep code valid."""

    @staticmethod
    def _strip_docstring_from_body(body):
        if not body:
            return body, False
        first = body[0]
        is_doc = (
            isinstance(first, ast.Expr)
            and isinstance(getattr(first, "value", None), ast.Constant)
            and isinstance(first.value.value, str)
        )
        if is_doc:
            return body[1:], True
        return body, False

    @staticmethod
    def _ensure_non_empty_suite(body):
        return body if body else [ast.Pass()]

    def visit_Module(self, node: ast.Module):
        self.generic_visit(node)
        node.body, _ = self._strip_docstring_from_body(node.body)
        node.body = self._ensure_non_empty_suite(node.body)
        return node

    def visit_ClassDef(self, node: ast.ClassDef):
        self.generic_visit(node)
        node.body, _ = self._strip_docstring_from_body(node.body)
        node.body = self._ensure_non_empty_suite(node.body)
        return node

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.generic_visit(node)
        node.body, _ = self._strip_docstring_from_body(node.body)
        node.body = self._ensure_non_empty_suite(node.body)
        return node

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.generic_visit(node)
        node.body, _ = self._strip_docstring_from_body(node.body)
        node.body = self._ensure_non_empty_suite(node.body)
        return node


def remove_hash_comments_keep_layout(src: str) -> str:
    """Remove `# ...` comments using tokenize while keeping code structure."""
    out_tokens = []
    tokgen = tokenize.generate_tokens(iter(src.splitlines(keepends=True)).__next__)
    for tok in tokgen:
        if tok.type == tokenize.COMMENT:
            continue
        out_tokens.append(tok)
    return tokenize.untokenize(out_tokens)


def strip_comments_from_py(py_path: Path) -> str:
    """Read a .py file, remove # comments + docstrings, return cleaned code (utf-8)."""
    # Respect PEP263 encoding
    with tokenize.open(str(py_path)) as f:
        src = f.read()

    no_hash = remove_hash_comments_keep_layout(src)

    tree = ast.parse(no_hash)
    tree = RemoveDocstrings().visit(tree)
    ast.fix_missing_locations(tree)

    # Python 3.9+ recommended
    try:
        cleaned = ast.unparse(tree) + "\n"
    except AttributeError:
        # fallback for old Python
        try:
            import astor  # type: ignore
        except ImportError as e:
            raise RuntimeError(
                "Python < 3.9 lacks ast.unparse. Please upgrade or `pip install astor`."
            ) from e
        cleaned = astor.to_source(tree)

    return cleaned


def should_skip(path: Path) -> bool:
    """Skip common junk/virtualenv/cache folders."""
    parts = {p.lower() for p in path.parts}
    skip_names = {
        ".git", "__pycache__", ".venv", "venv", "env",
        ".mypy_cache", ".pytest_cache", ".ruff_cache",
        "site-packages", "dist", "build",
    }
    return any(name in parts for name in skip_names)


def main():
    if not IN_DIR.exists() or not IN_DIR.is_dir():
        raise NotADirectoryError(f"IN_DIR not found or not a directory: {IN_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    py_files = [p for p in IN_DIR.rglob("*.py") if not should_skip(p)]
    print(f"Found {len(py_files)} .py files under: {IN_DIR}")
    print(f"Output folder: {OUT_DIR}\n")

    ok = 0
    failed = 0

    for src_path in py_files:
        rel = src_path.relative_to(IN_DIR)
        dst_path = OUT_DIR / rel
        dst_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            cleaned = strip_comments_from_py(src_path)
            dst_path.write_text(cleaned, encoding="utf-8")
            ok += 1
        except Exception as e:
            failed += 1
            print(f"[FAIL] {src_path}\n  -> {type(e).__name__}: {e}")

    print("\n==================================================")
    print(f"Done. OK: {ok}, Failed: {failed}")
    print("==================================================")
    if failed == 0:
        print("All files processed successfully.")


if __name__ == "__main__":
    main()
