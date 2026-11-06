"""Enable ``python -m quarry`` as an alias for the ``quarry`` console script."""

from __future__ import annotations

from .cli import main

if __name__ == "__main__":
    raise SystemExit(main())
