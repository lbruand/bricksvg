#!/usr/bin/env python3
"""Thin wrapper so ``scripts/mermaid2svg.py`` works without a package install."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ldr2svg.mermaid2svg import main
main()
