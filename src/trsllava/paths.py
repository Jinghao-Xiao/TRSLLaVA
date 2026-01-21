from __future__ import annotations

from pathlib import Path


def to_posix_rel(path: Path, root: Path) -> str:
    path_r = path.resolve()
    root_r = root.resolve()
    try:
        rel = path_r.relative_to(root_r)
    except Exception as e:
        raise ValueError(f"Path {path_r} is not under root {root_r}; would create absolute path") from e
    return rel.as_posix()


def resolve_under_root(path_str: str, root: Path) -> Path:
    p = Path(path_str.strip()).expanduser()
    if p.is_absolute():
        return p
    return (root / p).resolve()
