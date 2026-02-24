"""Custom-drawn UI components: Button, Dropdown, StatusBar, InfoOverlay."""

from __future__ import annotations

from typing import Callable, Optional

import pygame

from ui.theme import (
    BG_ACCENT,
    BG_ACCENT_HOVER,
    BG_BUTTON,
    BG_BUTTON_HOVER,
    BG_DROPDOWN,
    BG_DROPDOWN_HOVER,
    BG_OVERLAY,
    BG_STATUS,
    BORDER,
    BORDER_LIGHT,
    BUTTON_HEIGHT,
    BUTTON_RADIUS,
    OVERLAY_RADIUS,
    TEXT_DISABLED,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    font,
)


# ---------------------------------------------------------------------------
# Button
# ---------------------------------------------------------------------------

class Button:
    def __init__(
        self,
        text: str,
        x: int = 0,
        y: int = 0,
        width: int = 100,
        accent: bool = False,
        square: bool = False,
        on_click: Optional[Callable[[], None]] = None,
    ) -> None:
        self.text = text
        self.rect = pygame.Rect(x, y, width, BUTTON_HEIGHT)
        self.accent = accent
        self.square = square
        self.on_click = on_click
        self.hovered = False
        self.enabled = True

    def layout(self, x: int, y: int, width: Optional[int] = None) -> int:
        if width is not None:
            self.rect.width = width
        if self.square:
            self.rect.width = BUTTON_HEIGHT
        self.rect.x = x
        self.rect.y = y
        return self.rect.right

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.on_click:
                    self.on_click()
                return True
        return False

    def update(self, mouse_pos: tuple[int, int]) -> None:
        self.hovered = self.enabled and self.rect.collidepoint(mouse_pos)

    def draw(self, surface: pygame.Surface) -> None:
        if self.accent:
            bg = BG_ACCENT_HOVER if self.hovered else BG_ACCENT
        else:
            bg = BG_BUTTON_HOVER if self.hovered else BG_BUTTON

        if not self.enabled:
            bg = BG_BUTTON

        pygame.draw.rect(surface, bg, self.rect, border_radius=BUTTON_RADIUS)

        color = TEXT_PRIMARY if self.enabled else TEXT_DISABLED
        txt = font("normal").render(self.text, True, color)
        tx = self.rect.x + (self.rect.width - txt.get_width()) // 2
        ty = self.rect.y + (self.rect.height - txt.get_height()) // 2
        surface.blit(txt, (tx, ty))


# ---------------------------------------------------------------------------
# ToggleButton
# ---------------------------------------------------------------------------

class ToggleButton:
    def __init__(
        self,
        text_on: str = "Visualize ON",
        text_off: str = "Visualize OFF",
        active: bool = True,
        width: int = 140,
    ) -> None:
        self.text_on = text_on
        self.text_off = text_off
        self.active = active
        self.rect = pygame.Rect(0, 0, width, BUTTON_HEIGHT)
        self.hovered = False

    def layout(self, x: int, y: int, width: Optional[int] = None) -> int:
        if width is not None:
            self.rect.width = width
        self.rect.x = x
        self.rect.y = y
        return self.rect.right

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.active = not self.active
                return True
        return False

    def update(self, mouse_pos: tuple[int, int]) -> None:
        self.hovered = self.rect.collidepoint(mouse_pos)

    def draw(self, surface: pygame.Surface) -> None:
        bg = BG_ACCENT_HOVER if self.active and self.hovered else (
            BG_ACCENT if self.active else (
                BG_BUTTON_HOVER if self.hovered else BG_BUTTON
            )
        )
        pygame.draw.rect(surface, bg, self.rect, border_radius=BUTTON_RADIUS)
        text = self.text_on if self.active else self.text_off
        txt = font("normal").render(text, True, TEXT_PRIMARY)
        tx = self.rect.x + (self.rect.width - txt.get_width()) // 2
        ty = self.rect.y + (self.rect.height - txt.get_height()) // 2
        surface.blit(txt, (tx, ty))


# ---------------------------------------------------------------------------
# Dropdown
# ---------------------------------------------------------------------------

class Dropdown:
    def __init__(
        self,
        options: list[str],
        selected: int = 0,
        x: int = 0,
        y: int = 0,
        width: int = 140,
        label: str = "",
        on_change: Optional[Callable[[int, str], None]] = None,
        disabled_indices: Optional[set[int]] = None,
    ) -> None:
        self.options = options
        self.selected = selected
        self.label = label
        self.on_change = on_change
        self.disabled_indices = disabled_indices or set()
        self.rect = pygame.Rect(x, y, width, BUTTON_HEIGHT)
        self.open = False
        self.hovered_option = -1

    @property
    def value(self) -> str:
        return self.options[self.selected]

    def layout(self, x: int, y: int, width: Optional[int] = None) -> int:
        if width is not None:
            self.rect.width = width
        self.rect.x = x
        self.rect.y = y
        return self.rect.right

    def handle_event(self, event: pygame.event.Event) -> bool:
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.open:
                for i, opt_rect in enumerate(self._option_rects()):
                    if opt_rect.collidepoint(event.pos) and i not in self.disabled_indices:
                        self.selected = i
                        self.open = False
                        if self.on_change:
                            self.on_change(i, self.options[i])
                        return True
                self.open = False
                return True
            elif self.rect.collidepoint(event.pos):
                self.open = True
                return True
        return False

    def update(self, mouse_pos: tuple[int, int]) -> None:
        self.hovered_option = -1
        if self.open:
            for i, r in enumerate(self._option_rects()):
                if r.collidepoint(mouse_pos):
                    self.hovered_option = i

    def draw(self, surface: pygame.Surface) -> None:
        # Label
        if self.label:
            lbl = font("small").render(self.label, True, TEXT_SECONDARY)
            surface.blit(lbl, (self.rect.x, self.rect.y - 18))

        # Trigger box
        pygame.draw.rect(surface, BG_DROPDOWN, self.rect, border_radius=BUTTON_RADIUS)
        pygame.draw.rect(surface, BORDER, self.rect, width=1, border_radius=BUTTON_RADIUS)

        txt = font("normal").render(self.options[self.selected], True, TEXT_PRIMARY)
        tx = self.rect.x + 8
        ty = self.rect.y + (self.rect.height - txt.get_height()) // 2
        surface.blit(txt, (tx, ty))

        # Draw a small dropdown triangle
        tri_x = self.rect.right - 16
        tri_y = self.rect.centery
        pygame.draw.polygon(surface, TEXT_SECONDARY, [
            (tri_x - 4, tri_y - 3),
            (tri_x + 4, tri_y - 3),
            (tri_x, tri_y + 3),
        ])

        # Expanded options
        if self.open:
            for i, r in enumerate(self._option_rects()):
                bg = BG_DROPDOWN_HOVER if i == self.hovered_option else BG_DROPDOWN
                if i in self.disabled_indices:
                    bg = BG_DROPDOWN
                pygame.draw.rect(surface, bg, r)
                pygame.draw.rect(surface, BORDER, r, width=1)
                color = TEXT_DISABLED if i in self.disabled_indices else TEXT_PRIMARY
                ot = font("normal").render(self.options[i], True, color)
                surface.blit(ot, (r.x + 8, r.y + (r.height - ot.get_height()) // 2))

    def _option_rects(self) -> list[pygame.Rect]:
        rects: list[pygame.Rect] = []
        for i in range(len(self.options)):
            r = pygame.Rect(self.rect.x, self.rect.bottom + i * BUTTON_HEIGHT,
                            self.rect.width, BUTTON_HEIGHT)
            rects.append(r)
        return rects

    @property
    def consumes_click(self) -> bool:
        return self.open


# ---------------------------------------------------------------------------
# StatusBar
# ---------------------------------------------------------------------------

class StatusBar:
    def __init__(self) -> None:
        self.message = "Click to draw obstacles. Press S/G to set start/goal."
        self.rect = pygame.Rect(0, 0, 0, 0)

    def layout(self, x: int, y: int, w: int, h: int) -> None:
        self.rect = pygame.Rect(x, y, w, h)

    def set_message(self, msg: str) -> None:
        self.message = msg

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, BG_STATUS, self.rect)
        txt = font("small").render(self.message, True, TEXT_SECONDARY)
        surface.blit(txt, (self.rect.x + 12, self.rect.y + (self.rect.height - txt.get_height()) // 2))


# ---------------------------------------------------------------------------
# InfoOverlay
# ---------------------------------------------------------------------------

class InfoOverlay:
    SHORTCUTS = [
        ("Left click / drag", "Draw obstacles"),
        ("Right click / drag", "Erase obstacles"),
        ("S + click", "Set start"),
        ("G + click", "Set goal"),
        ("Esc", "Close this overlay"),
    ]

    def __init__(self) -> None:
        self.visible = False

    def toggle(self) -> None:
        self.visible = not self.visible

    def handle_event(self, event: pygame.event.Event) -> bool:
        if not self.visible:
            return False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.visible = False
            return True
        if event.type == pygame.MOUSEBUTTONDOWN:
            self.visible = False
            return True
        return False

    def draw(self, surface: pygame.Surface) -> None:
        if not self.visible:
            return

        w, h = surface.get_size()
        overlay = pygame.Surface((w, h), pygame.SRCALPHA)
        overlay.fill(BG_OVERLAY)
        surface.blit(overlay, (0, 0))

        pw, ph = min(500, w - 40), min(380, h - 40)
        px = (w - pw) // 2
        py = (h - ph) // 2
        panel = pygame.Rect(px, py, pw, ph)
        pygame.draw.rect(surface, BG_DROPDOWN, panel, border_radius=OVERLAY_RADIUS)
        pygame.draw.rect(surface, BORDER_LIGHT, panel, width=1, border_radius=OVERLAY_RADIUS)

        title = font("title").render("How to Use", True, TEXT_PRIMARY)
        surface.blit(title, (px + 24, py + 20))

        y = py + 70
        for key, desc in self.SHORTCUTS:
            kt = font("normal").render(key, True, TEXT_PRIMARY)
            dt = font("small").render(desc, True, TEXT_SECONDARY)
            surface.blit(kt, (px + 30, y))
            surface.blit(dt, (px + 240, y + 2))
            y += 38

        hint = font("small").render("Click anywhere or press Esc to close", True, TEXT_DISABLED)
        surface.blit(hint, (px + (pw - hint.get_width()) // 2, py + ph - 40))
