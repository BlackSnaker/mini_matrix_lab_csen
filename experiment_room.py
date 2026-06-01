from __future__ import annotations

import sys

from training_room import main


DEFAULT_ARGS = [
    "--best-agent",
    "--white-room",
    "--conversation-only",
]


if __name__ == "__main__":
    raise SystemExit(main(DEFAULT_ARGS + list(sys.argv[1:])))
