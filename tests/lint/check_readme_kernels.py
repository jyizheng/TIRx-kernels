#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""Check that each README architecture list matches the kernel registry.

Every supported CUDA architecture has its own section. Each section must list
exactly the public kernels whose ``KERNEL_META["runtime_cuda_archs"]`` contains
that architecture, and every link must open the registered module.
"""

from __future__ import annotations

import re
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
README = REPO_ROOT / "README.md"
ARCHITECTURES = ("sm_110a", "sm_100a", "sm_103a", "sm_107a")

_ARCH_HEADING = re.compile(r"^### `(sm_[0-9]+[af]?)` \([^)]+\)$", re.MULTILINE)
_LINK = re.compile(r"^- \[`([^`]+)`\]\((tirx_kernels/[^)]+\.py)\)$", re.MULTILINE)


def _registry() -> dict[str, tuple[str, tuple[str, ...]]]:
    sys.path.insert(0, str(REPO_ROOT))
    from tirx_kernels import registry

    index, diagnostics = registry._build_kernel_index(registry._source_snapshot())
    if diagnostics:
        raise SystemExit("\n".join(diagnostics))
    return {
        name: (str(record.source_path.relative_to(REPO_ROOT)), tuple(record.runtime_cuda_archs))
        for name, record in index.items()
    }


def main() -> int:
    kernels = _registry()
    text = README.read_text()
    errors: list[str] = []

    _, kernel_marker, remainder = text.partition("## Kernels\n")
    kernel_text, performance_marker, _ = remainder.partition("\n## Performance")
    if not kernel_marker or not performance_marker:
        errors.append("README.md must contain Kernels and Performance sections")
        headings = []
    else:
        headings = list(_ARCH_HEADING.finditer(kernel_text))

    listed_archs = tuple(match.group(1) for match in headings)
    if listed_archs != ARCHITECTURES:
        errors.append(f"architecture sections must be {ARCHITECTURES}, found {listed_archs}")

    registry_archs = {arch for _, archs in kernels.values() for arch in archs}
    for arch in sorted(registry_archs - set(ARCHITECTURES)):
        errors.append(f"registry architecture {arch} has no README section")

    parsed_link_count = 0
    for index, heading in enumerate(headings):
        arch = heading.group(1)
        body_end = headings[index + 1].start() if index + 1 < len(headings) else len(kernel_text)
        body = kernel_text[heading.end() : body_end]
        links = _LINK.findall(body)
        parsed_link_count += len(links)

        counts = Counter(name for name, _ in links)
        for name, count in sorted(counts.items()):
            if count > 1:
                errors.append(f"{arch}: {name} is linked {count} times")

        found = dict(links)
        expected = {name: path for name, (path, archs) in kernels.items() if arch in archs}
        for name in sorted(set(expected) - set(found)):
            errors.append(f"{arch}: {name} is not linked")
        for name in sorted(set(found) - set(expected)):
            errors.append(f"{arch}: {name} is linked but not supported")
        for name in sorted(set(expected) & set(found)):
            if found[name] != expected[name]:
                errors.append(f"{arch}: {name} links {found[name]}, module is {expected[name]}")

    all_kernel_links = re.findall(r"\[`[^`]+`\]\(tirx_kernels/[^)]+\.py\)", kernel_text)
    if len(all_kernel_links) != parsed_link_count:
        errors.append("kernel links must be plain bullets inside architecture sections")

    for error in errors:
        print(error, file=sys.stderr)
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
