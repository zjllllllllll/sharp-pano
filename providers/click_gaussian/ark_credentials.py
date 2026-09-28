"""Resolve the Volcengine Ark API key without logging or job persistence."""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path
from typing import Mapping


ARK_API_KEY_ENV = "ARK_API_KEY"
PROJECT_ARK_CREDENTIAL_PATH = (
    Path(__file__).resolve().parents[2] / 'secrets' / 'ark-key.txt'
)
MAX_CREDENTIAL_BYTES = 16 * 1024


class ArkCredentialError(RuntimeError):
    """The Ark credential is missing or unsafe to consume."""


def _validate(value: object, source: str) -> str:
    if not isinstance(value, str):
        raise ArkCredentialError(f"Ark API Key from {source} must be text")
    if "\n" in value or "\r" in value:
        raise ArkCredentialError(f"Ark API Key from {source} must be one line")
    if not value or value != value.strip():
        raise ArkCredentialError(
            f"Ark API Key from {source} is empty or has surrounding whitespace"
        )
    return value


def _read_private_project_file(path: Path) -> str:
    metadata = path.lstat()
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ArkCredentialError("ark_api_key.txt must be a regular non-symlink file")
    if hasattr(os, "geteuid") and int(metadata.st_uid) != int(os.geteuid()):
        raise ArkCredentialError("ark_api_key.txt must be owned by the current user")
    if stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ArkCredentialError("ark_api_key.txt must use chmod 600")
    if int(metadata.st_size) > MAX_CREDENTIAL_BYTES:
        raise ArkCredentialError("ark_api_key.txt exceeds the size limit")
    lines = path.read_text(encoding="utf-8").splitlines()
    values = [line.strip() for line in lines if line.strip() and not line.lstrip().startswith("#")]
    if len(values) != 1:
        raise ArkCredentialError(
            "ark_api_key.txt must contain exactly one non-comment API Key line"
        )
    return _validate(values[0], str(path))


def resolve_ark_api_key(
    explicit: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    credential_path: str | os.PathLike | None = None,
) -> str:
    """Resolve explicit value, then environment, then project key file."""

    if explicit is not None:
        return _validate(explicit, "the explicit argument")
    environment = os.environ if environ is None else environ
    if ARK_API_KEY_ENV in environment:
        return _validate(environment[ARK_API_KEY_ENV], ARK_API_KEY_ENV)
    path = (
        PROJECT_ARK_CREDENTIAL_PATH
        if credential_path is None
        else Path(credential_path).expanduser()
    )
    try:
        return _read_private_project_file(path)
    except FileNotFoundError:
        raise ArkCredentialError(
            f"no Ark API Key: write it to {path} or set {ARK_API_KEY_ENV}"
        ) from None


def ark_api_key_available(
    explicit: str | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    credential_path: str | os.PathLike | None = None,
) -> bool:
    try:
        resolve_ark_api_key(
            explicit,
            environ=environ,
            credential_path=credential_path,
        )
    except (ArkCredentialError, OSError, UnicodeError):
        return False
    return True


def save_project_ark_api_key(
    value: str,
    *,
    credential_path: str | os.PathLike | None = None,
) -> Path:
    """Atomically save one Ark key as a private project credential.

    This is the write-side counterpart of :func:`resolve_ark_api_key`, used by
    the Viewer password field.  The key is never returned or logged.  A
    same-directory temporary file plus ``os.replace`` prevents a partial key
    file if the process exits while saving.
    """

    key = _validate(value, "the Viewer password field")
    path = (
        PROJECT_ARK_CREDENTIAL_PATH
        if credential_path is None
        else Path(credential_path).expanduser()
    )
    parent = path.parent.resolve(strict=True)
    if not parent.is_dir():
        raise ArkCredentialError(f"credential parent is not a directory: {parent}")
    path = parent / path.name
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        metadata = None
    if metadata is not None:
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
            raise ArkCredentialError(
                "ark_api_key.txt must be a regular non-symlink file"
            )
        if hasattr(os, "geteuid") and int(metadata.st_uid) != int(os.geteuid()):
            raise ArkCredentialError(
                "ark_api_key.txt must be owned by the current user"
            )

    temporary_name = None
    try:
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".ark_api_key.", suffix=".tmp", dir=str(parent)
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                handle.write(key)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
        except Exception:
            try:
                os.close(descriptor)
            except OSError:
                pass
            raise
        os.replace(temporary_name, path)
        temporary_name = None
        path.chmod(0o600)
        # Read it back through the strict consumer before reporting success.
        _read_private_project_file(path)
        return path
    finally:
        if temporary_name is not None:
            try:
                Path(temporary_name).unlink()
            except FileNotFoundError:
                pass


__all__ = [
    "ARK_API_KEY_ENV",
    "PROJECT_ARK_CREDENTIAL_PATH",
    "ArkCredentialError",
    "ark_api_key_available",
    "resolve_ark_api_key",
    "save_project_ark_api_key",
]
