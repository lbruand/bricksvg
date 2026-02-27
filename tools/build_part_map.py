#!/usr/bin/env python3
"""Generate ldr2svg/part_map_data.py from the LDraw parts library.

Usage:
    uv run python tools/build_part_map.py [ldraw_parts_dir]

Default ldraw_parts_dir: /usr/share/ldraw/parts

In LDraw part descriptions "A x B":
  - A is the number of studs along the Z axis  → PartDef.length
  - B is the number of studs along the X axis  → PartDef.width
Verified from geometry: Brick 2x4 spans X=±40 (4 studs), Z=±20 (2 studs).
"""
import re
import sys
from pathlib import Path

LDRAW_PARTS = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/usr/share/ldraw/parts")
OUTPUT = Path(__file__).parent.parent / "ldr2svg" / "part_map_data.py"

# (base_type, is_round) → (block_type, height_expr)
_TYPES: dict[tuple[str, bool], tuple[str, str]] = {
    ("Brick", False): ("brick",      "1.0"),
    ("Brick", True):  ("round",      "1.0"),
    ("Plate", False): ("brick",      "1/3"),
    ("Plate", True):  ("round",      "1/3"),
    ("Tile",  False): ("tile",       "1/3"),
    ("Tile",  True):  ("round-tile", "1/3"),
}

DESC_RE = re.compile(
    r"^(Brick|Plate|Tile)\s+(\d+)\s+x\s+(\d+)(\s+Round\b)?",
    re.IGNORECASE,
)


def parse_dat(path: Path) -> tuple[str, int, int, str, str] | None:
    """Return (part_id, width, length, height_expr, block_type) or None."""
    try:
        first_line = path.read_text(errors="replace").split("\n")[0]
    except OSError:
        return None
    if not first_line.startswith("0 "):
        return None
    desc = first_line[2:].strip()
    # Skip moved/aliased parts
    if desc.startswith("~") or desc.startswith("="):
        return None
    m = DESC_RE.match(desc)
    if not m:
        return None
    base = m.group(1).capitalize()
    a, b = int(m.group(2)), int(m.group(3))
    is_round = m.group(4) is not None
    block_type, height_expr = _TYPES[(base, is_round)]
    # "A x B": A = Z studs (length), B = X studs (width)
    return path.stem.lower(), b, a, height_expr, block_type


def main() -> None:
    entries: dict[str, tuple[int, int, str, str]] = {}
    for dat in sorted(LDRAW_PARTS.glob("*.dat")):
        result = parse_dat(dat)
        if result:
            part_id, width, length, height_expr, block_type = result
            entries[part_id] = (width, length, height_expr, block_type)

    lines = [
        '"""part_map_data.py — auto-generated; do not edit by hand.',
        "",
        f"Generated from {LDRAW_PARTS}",
        "Run: uv run python tools/build_part_map.py",
        '"""',
        "",
        "# (width_studs, length_studs, height_fraction, block_type)",
        "PART_MAP_DATA: dict[str, tuple[int, int, float, str]] = {",
    ]
    for part_id, (w, ln, h, bt) in sorted(entries.items()):
        lines.append(f'    "{part_id}": ({w}, {ln}, {h}, "{bt}"),')
    lines.append("}")
    lines.append("")

    OUTPUT.write_text("\n".join(lines))
    print(f"Wrote {len(entries)} parts to {OUTPUT}")


if __name__ == "__main__":
    main()
