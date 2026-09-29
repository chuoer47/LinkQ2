"""Comment hygiene: prose must not state anything the code can outlive.

Scans comments and docstrings only, never code: a path in a string literal is
a fact the runtime uses, the same path in a comment is a claim about the repo.
"""
from __future__ import annotations

import ast
import io
import re
import subprocess
import tokenize
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

MILESTONE = re.compile(r"\b[Mm]\d{1,2}\b|\b[Rr]\d{1,2}\b|design-[a-z]|§|\bTODO\b")
LONG_HASH = re.compile(r"\b[0-9a-f]{7,40}\b")
DIR_REF = re.compile(r"\b(?:docs|notes|archive|benchmarks/\d[\w.-]*)/\S*|\bREADME\.md\b")
ARTIFACT = re.compile(r"\bresults/[\w./-]+\.(?:txt|json|pt|md|log)\b")
KNOB = re.compile(r"^([A-Z][A-Z0-9_]{1,})\s{2,}(.*)$")
GETENV = re.compile(r'os\.environ\.get\(\s*["\x27]([A-Z0-9_]+)["\x27]'
                    r'(?:\s*,\s*["\x27]([^"\x27]*)["\x27])?')


@pytest.fixture(scope="module")
def tracked():
    out = subprocess.check_output(["git", "ls-files", "*.py"], cwd=REPO,
                                  text=True).split()
    return [(rel, REPO / rel) for rel in out]


def comments(path: Path):
    with path.open(encoding="utf-8") as fh:
        src = fh.read()
    return [tok.string.lstrip("#").strip()
            for tok in tokenize.generate_tokens(io.StringIO(src).readline)
            if tok.type == tokenize.COMMENT]


def docstrings(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            doc = ast.get_docstring(node, clean=False)
            if doc:
                out.append(doc)
    return out


def prose(path: Path):
    return comments(path) + docstrings(path)


def test_no_milestone_or_plan_references(tracked):
    bad = []
    for rel, path in tracked:
        for text in prose(path):
            for rx, why in ((MILESTONE, "milestone/plan id"),
                            (LONG_HASH, "commit hash")):
                m = rx.search(text)
                if m:
                    bad.append(f"{rel}: {why}: {m.group(0)!r} in {text[:70]!r}")
    assert not bad, "\n".join(bad)


def test_no_references_to_untracked_directories(tracked):
    bad = []
    for rel, path in tracked:
        for text in prose(path):
            for m in DIR_REF.finditer(text):
                tok = m.group(0).rstrip(".,;:)")
                if not (REPO / tok).exists():
                    bad.append(f"{rel}: dead path {tok!r} in {text[:70]!r}")
    assert not bad, "\n".join(bad)


def test_results_paths_in_prose_must_be_used_by_code(tracked):
    bad = []
    for rel, path in tracked:
        code = "".join(line for line in path.read_text(encoding="utf-8")
                       .splitlines(keepends=True) if not line.lstrip().startswith("#"))
        for text in prose(path):
            for m in ARTIFACT.finditer(text):
                if m.group(0) not in code:
                    bad.append(f"{rel}: {m.group(0)!r} cited but not written by this file")
    assert not bad, "\n".join(bad)


def env_rows(doc: str):
    """Yield (NAME, documented default) from a module docstring's Env table."""
    lines = doc.splitlines()
    start = next((i for i, l in enumerate(lines)
                  if re.fullmatch(r"\s*(Env|Usage|Options):", l)), None)
    if start is None:
        return
    rows, name, parts = [], None, []
    for ln in lines[start + 1:]:
        m = KNOB.match(ln)
        if m:
            if name:
                rows.append((name, " ".join(parts)))
            name, parts = m.group(1), [m.group(2)]
        elif name and ln.strip():
            parts.append(ln.strip())
    if name:
        rows.append((name, " ".join(parts)))
    for name, text in rows:
        m = re.search(r"\(([^()]*)\)", text)
        if not m:
            continue
        first = re.split(r";| — ", m.group(1).strip())[0].strip()
        val = re.sub(r"^default\s+", "", first).strip().strip('"')
        if val:
            yield name, val


def test_documented_env_defaults_match_the_code(tracked):
    bad = []
    for rel, path in tracked:
        src = path.read_text(encoding="utf-8")
        code = {k: (v or "") for k, v in GETENV.findall(src)}
        doc = ast.get_docstring(ast.parse(src)) or ""
        for name, val in env_rows(doc):
            if name in code and code[name] and code[name] != val:
                bad.append(f'{rel}: {name} docstring says {val!r}, '
                           f'code says {code[name]!r}')
    assert not bad, "\n".join(bad)
