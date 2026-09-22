"""Explicit runtime paths, including intermediate symlink targets."""
import hashlib
import os
from pathlib import Path


def checked_path(value, must_exist=True):
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError("Runtime paths must be absolute: " + str(path))
    # Inspect every hop: resolve() alone can hide an intermediate forbidden mount.
    for _ in range(64):
        if any(part in ("hdd2", "roboticsystem2") for part in path.parts):
            raise ValueError("Forbidden old-project runtime path: " + str(path))
        prefix = Path(path.anchor)
        for index, part in enumerate(path.parts[1:], 1):
            prefix = prefix / part
            if prefix.is_symlink():
                target = Path(os.readlink(prefix))
                if not target.is_absolute():
                    target = prefix.parent / target
                # Check targets before normalizing '..'.
                if any(p in ("hdd2", "roboticsystem2") for p in target.parts):
                    raise ValueError("Forbidden symlink target: " + str(target))
                path = Path(os.path.abspath(target.joinpath(*path.parts[index + 1:])))
                break
        else:
            path = path.resolve(strict=must_exist)
            if must_exist and not path.exists():
                raise FileNotFoundError(path)
            return path
    raise ValueError("Symlink loop or more than 64 hops: " + str(value))


def sha256_file(path):
    digest = hashlib.sha256()
    with checked_path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
