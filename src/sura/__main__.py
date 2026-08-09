"""Allow ``python -m sura`` to invoke the project CLI."""

from .cli import main

raise SystemExit(main())
