#!/usr/bin/env python3
"""Fail on personal paths, private network addresses, or tracked private artifacts.

Gitleaks separately checks secrets. This checker prints only filenames/reasons.
It checks Git-index content so it also sees files accidentally staged despite ignores.
"""

import re
import subprocess
import sys
from pathlib import PurePosixPath
from urllib.parse import urlsplit

PERSONAL_PATH = re.compile(r"/(?:home|Users)/[A-Za-z0-9_.-]+/")
WINDOWS_HOME = re.compile(r"[A-Za-z]:[\\/]Users[\\/][A-Za-z0-9_.-]+[\\/]")
PRIVATE_IPV4 = re.compile(
    r"(?<![\d.])(?:10(?:\.\d{1,3}){3}|192\.168(?:\.\d{1,3}){2}"
    r"|172\.(?:1[6-9]|2\d|3[01])(?:\.\d{1,3}){2})(?![\d.])"
)
PRIVATE_SUFFIXES = {".pt", ".pth", ".onnx", ".mp4", ".avi", ".pem", ".key", ".bag", ".mcap"}
PRIVATE_ROOTS = {"data", "weights", "outputs", "runs", ".venv"}


def main():
    files = subprocess.check_output(["git", "ls-files", "-z"]).decode().split("\0")
    failures = []
    for filename in filter(None, files):
        path = PurePosixPath(filename)
        if (path.suffix in PRIVATE_SUFFIXES or path.parts[0] in PRIVATE_ROOTS
                or path.name == ".env" or (path.name.startswith(".env.") and path.name != ".env.example")
                or path.name.startswith(("id_rsa", "id_ed25519"))):
            failures.append((filename, "private artifact staged"))
            continue
        content = subprocess.check_output(["git", "show", f":{filename}"])
        if b"\0" in content:
            failures.append((filename, "binary file staged; public source must be text"))
            continue
        text = content.decode("utf-8", errors="replace")
        if PERSONAL_PATH.search(text) or WINDOWS_HOME.search(text):
            failures.append((filename, "personal absolute path"))
        # A CUDA dependency's four-component version can look like a private IP.
        # In the generated lockfile inspect URL hosts rather than version strings.
        address_text = text
        if path.name == "uv.lock":
            address_text = "\n".join(
                urlsplit(url).hostname or ""
                for url in re.findall(r'https?://[^"\s]+', text)
            )
        if PRIVATE_IPV4.search(address_text):
            failures.append((filename, "private network address"))
    for filename, reason in failures:
        print(f"{filename}: {reason}", file=sys.stderr)
    if failures:
        return 1
    print("Public-source check passed (tracked/index content).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
