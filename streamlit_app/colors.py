"""Shared color tokens for the ED Model Explorer UI.

Two muted color layers distinguish status and model comparisons without
turning the interface into a multi-color dashboard:

- ``STATUS_STYLE`` — what a number *means* (good / warning / serious) in the
  wait-time gauge.
- ``MODEL_BASELINE`` / ``MODEL_MAIN`` / ``SCENARIO_ALT`` — *which* model or
  scenario a bar or line represents. Reused by every wait-time comparison
  chart.
"""

STATUS_STYLE = {
    "good": {"icon": "↓", "bg": "#6f8797", "fg": "#ffffff"},
    "warning": {"icon": "→", "bg": "#5f7485", "fg": "#ffffff"},
    "serious": {"icon": "↑", "bg": "#4f6272", "fg": "#ffffff"},
}

MODEL_BASELINE = "#58748f"  # Model 1 / access-only — the research baseline
MODEL_MAIN = "#565f7f"  # Model 2 / main estimate
SCENARIO_ALT = "#7a6c7d"  # what-if scenario — a hypothetical, not the estimate
