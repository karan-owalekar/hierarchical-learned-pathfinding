"""Dark-mode visual theme: colors, fonts, spacing."""

from __future__ import annotations

import pygame

# ---------------------------------------------------------------------------
# Colors
# ---------------------------------------------------------------------------

BG_DARK = (18, 18, 22)
BG_TOOLBAR = (28, 28, 34)
BG_STATUS = (24, 24, 30)
BG_BUTTON = (50, 50, 60)
BG_BUTTON_HOVER = (70, 70, 82)
BG_ACCENT = (56, 132, 244)
BG_ACCENT_HOVER = (80, 150, 255)
BG_DROPDOWN = (40, 40, 50)
BG_DROPDOWN_HOVER = (60, 60, 74)
BG_OVERLAY = (0, 0, 0, 180)

TEXT_PRIMARY = (230, 230, 235)
TEXT_SECONDARY = (160, 160, 170)
TEXT_DISABLED = (90, 90, 100)

BORDER = (70, 70, 80)
BORDER_LIGHT = (100, 100, 112)

# Grid cell colors
CELL_EMPTY = (38, 38, 46)
CELL_OBSTACLE = (12, 12, 16)
CELL_START = (46, 204, 113)
CELL_GOAL = (231, 76, 60)
CELL_PATH = (56, 132, 244)
CELL_EXPLORED = (60, 60, 90)
CELL_FRONTIER = (100, 80, 140)
CELL_CORRIDOR = (80, 120, 60)

# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------

TOOLBAR_HEIGHT = 72
STATUS_HEIGHT = 36
BUTTON_HEIGHT = 34
BUTTON_RADIUS = 6
CELL_RADIUS = 2
OVERLAY_RADIUS = 14
CELL_GAP = 1
MIN_CELL_SIZE = 2
PADDING = 12
ELEMENT_SPACING = 10

# ---------------------------------------------------------------------------
# Fonts (initialized lazily after pygame.init)
# ---------------------------------------------------------------------------

_fonts: dict[str, pygame.font.Font] = {}


def init_fonts() -> None:
    _fonts["title"] = pygame.font.SysFont(None, 36)
    _fonts["normal"] = pygame.font.SysFont(None, 32)
    _fonts["small"] = pygame.font.SysFont(None, 26)


def font(name: str) -> pygame.font.Font:
    return _fonts[name]
