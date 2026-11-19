"""Module entrypoint for `python3 -m audit_rl`."""

from __future__ import annotations

import sys

from audit_rl.cli import main

if __name__ == "__main__":
    sys.exit(main())
