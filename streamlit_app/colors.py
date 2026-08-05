"""Shared color tokens for the ED Model Explorer UI.

Two independent color layers, kept deliberately non-overlapping so they never
get confused when shown together (e.g. the wait-time gauge sits right above
its model-comparison charts on the same page):

- ``STATUS_STYLE`` — what a number *means* (good / warning / serious). Reused
  by the wait-time gauge and the admission risk badges, so the same color
  always means the same thing everywhere in the app.
- ``MODEL_BASELINE`` / ``MODEL_MAIN`` / ``SCENARIO_ALT`` — *which* model or
  scenario a bar or line represents. Reused by every wait-time comparison
  chart. Chosen from a different hue family than the status colors (blue,
  violet, magenta vs. green, amber, salmon) so the two layers never collide.
"""

STATUS_STYLE = {
    "good": {"icon": "↓", "bg": "#0ca30c", "fg": "#ffffff"},
    "warning": {"icon": "→", "bg": "#fab219", "fg": "#1a1a19"},
    "serious": {"icon": "↑", "bg": "#ec835a", "fg": "#1a1a19"},
}

MODEL_BASELINE = "#2a78d6"  # Model 1 / access-only — the research baseline
MODEL_MAIN = "#4a3aa7"  # Model 2 / main estimate — the number to quote
SCENARIO_ALT = "#e87ba4"  # what-if scenario — a hypothetical, not the estimate
