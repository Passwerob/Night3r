#!/usr/bin/env python3
"""Night3R training entry point.

This wrapper keeps the historical train_event_dark3r.py implementation intact
for checkpoint compatibility while exposing the cleaned project name.
"""

from train_event_dark3r import main


if __name__ == "__main__":
    main()
