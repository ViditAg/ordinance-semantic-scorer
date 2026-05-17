"""
Named chunking profiles for consistent scoring across PDFs.

**Design**

* **Balanced** — default house standard (chunk size + overlap) for routine runs.
* **Fine** / **Coarse** — optional alternatives when document length or layout
  suggests smaller or larger windows, without ad-hoc slider tuning.
* **Custom** — advanced; same sliders as before, for one-off experiments.

Optional **calibration** (grid sweep via ``scripts/calibrate.py``) is occasional;
production chunking uses ``DEFAULT_CHUNK_*`` so comparable ordinances share the
same segmentation policy.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# House defaults — single source of truth for "recommended" segmentation.
# Smaller windows improve citation locality vs. a single 2k blob that mixes articles.
DEFAULT_CHUNK_SIZE = 750
DEFAULT_CHUNK_OVERLAP = 120

# Default calibration grid (``scripts/calibrate.py`` when flags omitted).
# Spans finer windows around ``DEFAULT_CHUNK_SIZE`` (750) so sweeps match production
# segmentation policy; overlaps stay strictly below the smallest chunk size (500).
SWEEP_CHUNK_SIZE_MIN_DEFAULT = 500
SWEEP_CHUNK_SIZE_MAX_DEFAULT = 1500
SWEEP_CHUNK_SIZE_STEP_DEFAULT = 250
SWEEP_OVERLAP_MIN_DEFAULT = 50
SWEEP_OVERLAP_MAX_DEFAULT = 150
SWEEP_OVERLAP_STEP_DEFAULT = 25

PresetKey = str

_PRESETS: Dict[PresetKey, Tuple[int, int, str, str]] = {
    "balanced": (
        DEFAULT_CHUNK_SIZE,
        DEFAULT_CHUNK_OVERLAP,
        "Balanced (recommended)",
        "Default for cross-document comparisons; good for typical ordinance length.",
    ),
    "fine": (
        500,
        75,
        "Fine-grained",
        "Shorter chunks than the default — tighter citations; more embedding calls.",
    ),
    "coarse": (
        1800,
        180,
        "Coarse segments",
        "Fewer, longer chunks — fewer embedding calls; watch for ideas split across cuts.",
    ),
}

PRESET_ORDER: List[PresetKey] = ["balanced", "fine", "coarse", "custom"]


def preset_label(preset_key: str) -> str:
    """Sidebar ``format_func`` — human-readable row for ``selectbox``."""
    if preset_key == "custom":
        return "Custom (manual sliders)"
    return _PRESETS[preset_key][2]


def resolve_chunking(
    preset_key: str,
    custom_chunk_size: int,
    custom_overlap: int,
) -> Tuple[int, int]:
    """
    Return ``(chunk_size, overlap)`` for the chosen profile.

    For ``custom``, returns the slider values unchanged (caller must enforce
    ``overlap < chunk_size`` via Streamlit bounds).
    """
    if preset_key == "custom":
        return int(custom_chunk_size), int(custom_overlap)
    size, ov, _, _ = _PRESETS[preset_key]
    return int(size), int(ov)


def describe_preset(preset_key: str) -> str:
    """One-line rationale for captions / tooltips."""
    if preset_key == "custom":
        return "Set chunk size and overlap manually in the sidebar."
    return _PRESETS[preset_key][3]
