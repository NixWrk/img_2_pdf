"""Verify that the interpreter producing an x64 Windows artifact is itself win-amd64."""

from __future__ import annotations

import platform
import struct
import sysconfig


def main() -> int:
    system = platform.system()
    bits = struct.calcsize("P") * 8
    target = sysconfig.get_platform().lower()
    if system != "Windows" or bits != 64 or target != "win-amd64":
        raise SystemExit(
            f"Windows x64 build requires win-amd64 Python; got "
            f"system={system}, bits={bits}, platform={target}"
        )
    print(f"Build runtime OK: {platform.python_version()} ({target}, {bits}-bit)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
