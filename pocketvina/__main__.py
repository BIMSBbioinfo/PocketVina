"""``python -m pocketvina``: the command line (``pocketvina.cli``)."""
from .cli import main

if __name__ == "__main__":  # also keeps spawned worker processes from running the command again
    main()
