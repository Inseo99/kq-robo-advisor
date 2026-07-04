"""Package entrypoint for running the legacy-compatible local app."""

from __future__ import annotations


def main() -> None:
    """Run the current local server entrypoint.

    The top-level ``server.py`` file remains the compatibility entrypoint for
    team members. Importing it here gives us ``python -m kq_tool`` while the
    remaining legacy app is gradually thinned into package modules.
    """

    import server

    server.main()


if __name__ == "__main__":
    main()
