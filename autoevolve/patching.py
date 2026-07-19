from __future__ import annotations

from dataclasses import dataclass


SEARCH_MARKER = "<<<<<<< SEARCH"
DIVIDER_MARKER = "======="
REPLACE_MARKER = ">>>>>>> REPLACE"
EVOLVE_START = "# EVOLVE-BLOCK-START"
EVOLVE_END = "# EVOLVE-BLOCK-END"


class PatchError(ValueError):
    pass


@dataclass(frozen=True)
class PatchBlock:
    search: str
    replacement: str


def parse_patch(response: str) -> list[PatchBlock]:
    lines = response.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[PatchBlock] = []
    index = 0

    while index < len(lines):
        if lines[index].strip() != SEARCH_MARKER:
            index += 1
            continue
        index += 1
        search_lines: list[str] = []
        while index < len(lines) and lines[index].strip() != DIVIDER_MARKER:
            search_lines.append(lines[index])
            index += 1
        if index >= len(lines):
            raise PatchError("SEARCH block is missing the ======= divider")
        index += 1
        replacement_lines: list[str] = []
        while index < len(lines) and lines[index].strip() != REPLACE_MARKER:
            replacement_lines.append(lines[index])
            index += 1
        if index >= len(lines):
            raise PatchError("REPLACE block is missing the >>>>>>> REPLACE marker")
        index += 1
        search = "\n".join(search_lines)
        replacement = "\n".join(replacement_lines)
        if not search:
            raise PatchError("SEARCH text cannot be empty")
        blocks.append(PatchBlock(search=search, replacement=replacement))

    if not blocks:
        raise PatchError("Model response contained no SEARCH/REPLACE blocks")
    return blocks


def _evolve_regions(code: str) -> list[tuple[int, int]]:
    regions: list[tuple[int, int]] = []
    cursor = 0
    while True:
        start_marker = code.find(EVOLVE_START, cursor)
        if start_marker < 0:
            break
        content_start = code.find("\n", start_marker)
        if content_start < 0:
            raise PatchError("EVOLVE-BLOCK-START must be followed by a newline")
        content_start += 1
        end_marker = code.find(EVOLVE_END, content_start)
        if end_marker < 0:
            raise PatchError("EVOLVE-BLOCK-START has no matching EVOLVE-BLOCK-END")
        regions.append((content_start, end_marker))
        cursor = end_marker + len(EVOLVE_END)
    return regions


def _inside_region(start: int, end: int, regions: list[tuple[int, int]]) -> bool:
    return any(start >= region_start and end <= region_end for region_start, region_end in regions)


def evolve_content(code: str) -> str:
    """Return only mutable regions, or the whole file when no regions are declared."""
    normalized = code.replace("\r\n", "\n").replace("\r", "\n")
    regions = _evolve_regions(normalized)
    if not regions:
        return normalized
    return "\n\n".join(normalized[start:end] for start, end in regions)


def apply_patch(code: str, response: str, *, validate_python: bool = True) -> str:
    current = code.replace("\r\n", "\n").replace("\r", "\n")
    blocks = parse_patch(response)

    for block_number, block in enumerate(blocks, 1):
        if EVOLVE_START in block.replacement or EVOLVE_END in block.replacement:
            raise PatchError(f"Block {block_number} cannot add or modify EVOLVE markers")
        matches = current.count(block.search)
        if matches == 0:
            raise PatchError(f"Block {block_number} SEARCH text was not found exactly")
        if matches > 1:
            raise PatchError(f"Block {block_number} SEARCH text is ambiguous ({matches} matches)")
        start = current.index(block.search)
        end = start + len(block.search)
        regions = _evolve_regions(current)
        if regions and not _inside_region(start, end, regions):
            raise PatchError(f"Block {block_number} modifies code outside an EVOLVE block")
        current = current[:start] + block.replacement + current[end:]

    if current == code:
        raise PatchError("Patch made no change")
    if current.count(EVOLVE_START) != code.count(EVOLVE_START) or current.count(EVOLVE_END) != code.count(EVOLVE_END):
        raise PatchError("Patch changed the EVOLVE block boundaries")
    if validate_python:
        try:
            compile(current, "<evolved-program>", "exec")
        except SyntaxError as exc:
            raise PatchError(f"Patched program is invalid Python: {exc.msg} at line {exc.lineno}") from exc
    return current
