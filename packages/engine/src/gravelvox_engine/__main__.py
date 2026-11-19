"""Allow `python -m gravelvox_engine`."""

import sys

from gravelvox_engine.cli import main

if __name__ == "__main__":
    sys.exit(main())
