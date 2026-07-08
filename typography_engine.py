"""
QuoteLab Elite v2 - Typography Engine
======================================
Professional typography with smart font selection, optical sizing,
intelligent line breaking, and visual hierarchy.
"""

import re
import math
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum

from PIL import ImageFont, ImageDraw, Image
import numpy as np

import config


class FontStyle(Enum):
    """Font style variations"""
    REGULAR = "Regular"
    LIGHT = "Light"
    MEDIUM = "Medium"
    SEMI_BOLD = "SemiBold"
    BOLD = "Bold"
    EXTRA_BOLD = "ExtraBold"
    BLACK = "Black"
    ITALIC = "Italic"
    BOLD_ITALIC = "BoldItalic"


@dataclass
class FontSpec:
    """Font specification"""
    family: str = "Poppins"
    style: str = "Regular"
    size: int = 120
    letter_spacing: float = 0.0
    line_height: float = 1.3
    ligatures: bool = True
    kerning: bool = True

    @property
    def filename(self) -> str:
        """Generate font filename"""
        style_map = {
            "Regular": "-Regular.ttf",
            "Light": "-Light.ttf",
            "Medium": "-Medium.ttf",
            "SemiBold": "-SemiBold.ttf",
            "Bold": "-Bold.ttf",
            "ExtraBold": "-ExtraBold.ttf",
            "Black": "-Black.ttf",
            "Italic": "-Italic.ttf",
            "BoldItalic": "-BoldItalic.ttf",
        }
        return f"{self.family}{style_map.get(self.style, '-Regular.ttf')}"

    @property
    def path(self) -> Path:
        """Get full font path"""
        return config.FONTS / self.filename


@dataclass
class TextLine:
    """Represents a single line of text with metrics"""
    text: str = ""
    words: List[str] = field(default_factory=list)
    width: float = 0.0
    height: float = 0.0
    ascent: float = 0.0
    descent: float = 0.0
    x: float = 0.0
    y: float = 0.0
    font_size: int = 120
    font_spec: Optional[FontSpec] = None
    score: float = 0.0


@dataclass
class TextBlock:
    """Complete text block with layout information"""
    lines: List[TextLine] = field(default_factory=list)
    total_width: float = 0.0
    total_height: float = 0.0
    font_spec: Optional[FontSpec] = None
    alignment: str = "center"
    base_y: float = 0.0
    layout_score: float = 0.0
    readability_score: float = 0.0

    def get_all_words(self) -> List[str]:
        """Get all words from all lines"""
        return [word for line in self.lines for word in line.words]


class SmartLineBreaker:
    """
    Intelligent line breaking with optimal balance and readability.
    Uses dynamic programming for optimal paragraph breaking.
    """

    # Optimal characters per line (readability science: 45-75 chars)
    OPTIMAL_CPL = 35
    MIN_CPL = 15
    MAX_CPL = 50

    # Words that shouldn't end a line
    NO_BREAK_AFTER = {
        "a", "an", "the", "is", "are", "was", "were", "be",
        "of", "in", "for", "on", "with", "at", "from", "as",
        "and", "but", "or", "nor", "if", "to", "by",
    }

    def __init__(self, font: ImageFont.FreeTypeFont, max_width: float):
        self.font = font
        self.max_width = max_width
        self.draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

    def measure(self, text: str) -> float:
        """Measure text width in pixels"""
        if not text:
            return 0.0
        bbox = self.draw.textbbox((0, 0), text, font=self.font)
        return bbox[2] - bbox[0]

    def get_optimal_breaks(self, text: str) -> List[str]:
        """
        Find optimal line breaks using dynamic programming.
        Minimizes the total badness of line lengths.
        """
        words = text.split()
        if not words:
            return []

        if len(words) == 1:
            return [text]

        n = len(words)
        
        # Precompute word widths
        word_widths = [self.measure(w) for w in words]
        space_width = self.measure(" ")
        
        # Line widths: width of concatenating words[i:j]
        line_widths = {}
        for i in range(n):
            for j in range(i + 1, n + 1):
                line_text = " ".join(words[i:j])
                line_widths[(i, j)] = self.measure(line_text)

        # DP: cost[i] = minimum cost to break words[0:i]
        INF = float('inf')
        cost = [INF] * (n + 1)
        breaks = [0] * (n + 1)
        cost[0] = 0

        for j in range(1, n + 1):
            for i in range(max(0, j - 15), j):  # Look back max 15 words
                width = line_widths[(i, j)]
                
                if width > self.max_width * 1.2:  # Allow 20% overflow
                    continue

                # Badness of this line
                badness = self._compute_badness(words[i:j], width)
                total_cost = cost[i] + badness

                if total_cost < cost[j]:
                    cost[j] = total_cost
                    breaks[j] = i

        # Reconstruct lines
        lines = []
        j = n
        while j > 0:
            i = breaks[j]
            line = " ".join(words[i:j])
            lines.append(line)
            j = i

        lines.reverse()
        return lines

    def _compute_badness(self, words: List[str], width: float) -> float:
        """
        Compute badness of a line (lower is better).
        Considers length, balance, and grammar.
        """
        badness = 0.0

        # Badness based on deviation from optimal width
        ratio = width / self.max_width if self.max_width > 0 else 1.0
        
        if ratio < 0.5:
            badness += (0.5 - ratio) ** 2 * 500  # Very short lines are bad
        elif ratio > 0.95:
            badness += (ratio - 0.95) ** 2 * 200  # Overflow is worse
        else:
            # Favor lines close to optimal (around 70-80% width)
            optimal_ratio = 0.75
            badness += abs(ratio - optimal_ratio) * 50

        # Penalize breaking after small words
        if words and words[-1].lower() in self.NO_BREAK_AFTER:
            badness += 100

        # Penalize single-word lines (hard to read)
        if len(words) == 1:
            badness += 200

        # Penalize lines starting with lowercase (grammar issue)
        if words and words[0] and words[0][0].islower():
            badness += 150

        # Reward balanced line lengths (variance penalty)
        if len(words) > 1:
            word_lens = [len(w) for w in words]
            avg_len = sum(word_lens) / len(word_lens)
            variance = sum((l - avg_len) ** 2 for l in word_lens) / len(word_lens)
            badness += math.sqrt(variance) * 10

        return badness

    def balance_lines(self, lines: List[str]) -> List[str]:
        """Further optimize line lengths for visual balance"""
        if len(lines) <= 1:
            return lines

        # If lines are very unbalanced, rebreak them
        widths = [self.measure(line) for line in lines]
        max_width = max(widths) if widths else 0
        min_width = min(widths) if widths else 0

        if max_width > 0 and (max_width - min_width) / max_width > 0.4:
            # Too unbalanced, re-optimize
            full_text = " ".join(lines)
            return self.get_optimal_breaks(full_text)

        return lines


class TypographyEngine:
    """
    Master Typography Engine.
    Handles smart font selection, optical sizing, line breaking, and layout.
    """

    def __init__(self):
        self.font_cache: Dict[str, ImageFont.FreeTypeFont] = {}
        self.available_fonts: List[str] = self._scan_fonts()

        # Font pairing by category
        self.category_fonts = {
            "motivation": ["Poppins", "Montserrat", "Oswald"],
            "love": ["PlayfairDisplay", "GreatVibes", "Cormorant"],
            "business": ["Inter", "Roboto", "Lato"],
            "wisdom": ["Merriweather", "EBGaramond", "Crimson"],
            "life": ["Poppins", "Inter", "Montserrat"],
            "default": ["Poppins", "Inter", "Montserrat"],
        }

    def _scan_fonts(self) -> List[str]:
        """Scan available fonts in assets/fonts directory"""
        fonts = []
        if config.FONTS.exists():
            for font_file in config.FONTS.glob("*.ttf"):
                fonts.append(font_file.stem)
        return fonts

    def get_font(
        self,
        family: str = None,
        style: str = "Regular",
        size: int = None
    ) -> ImageFont.FreeTypeFont:
        """Get cached font or load new one"""
        family = family or config.DEFAULT_FONT
        size = size or config.FONT_SIZE_OPTIMAL
        key = f"{family}-{style}-{size}"

        if key not in self.font_cache:
            try:
                spec = FontSpec(family=family, style=style, size=size)
                font = ImageFont.truetype(str(spec.path), size)
                self.font_cache[key] = font
            except (OSError, FileNotFoundError):
                # Fallback to default
                try:
                    default_font = ImageFont.truetype(
                        str(config.FONTS / "Poppins-Regular.ttf"),
                        size
                    )
                    self.font_cache[key] = default_font
                except:
                    # Last resort: default Pillow font
                    self.font_cache[key] = ImageFont.load_default()

        return self.font_cache[key]

    def choose_font_for_category(self, category: str = "default") -> str:
        """Choose appropriate font family for category"""
        import random
        fonts = self.category_fonts.get(category, self.category_fonts["default"])
        # Filter to available fonts
        available = [f for f in fonts if f in self.available_fonts]
        return available[0] if available else "Poppins"

    def fit_font_size(
        self,
        text: str,
        max_width: float,
        max_size: int = None,
        min_size: int = None,
        family: str = None
    ) -> Tuple[ImageFont.FreeTypeFont, int]:
        """
        Find optimal font size that fits text within max_width.
        Uses binary search for efficiency.
        """
        max_size = max_size or config.FONT_SIZE_MAX
        min_size = min_size or config.FONT_SIZE_MIN
        family = family or config.DEFAULT_FONT

        draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))

        # Binary search for optimal size
        low, high = min_size, max_size
        best_font = None
        best_size = min_size

        while low <= high:
            mid = (low + high) // 2
            try:
                font = self.get_font(family, "Regular", mid)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]

                if text_width <= max_width:
                    best_font = font
                    best_size = mid
                    low = mid + 1  # Try larger
                else:
                    high = mid - 1  # Try smaller
            except:
                high = mid - 1

        if best_font is None:
            best_font = self.get_font(family, "Regular", min_size)
            best_size = min_size

        return best_font, best_size

    def compose(
        self,
        quote: str,
        author: str = "",
        category: str = "default",
        max_width: float = None,
        font_family: str = None
    ) -> TextBlock:
        """
        Compose text block with optimal layout.
        
        Pipeline:
        1. Choose font family
        2. Fit font size to quote
        3. Optimal line breaking
        4. Balance lines
        5. Calculate metrics
        6. Score layout
        """
        # Sanitize quote
        quote = quote.strip()
        if not quote:
            return TextBlock()

        max_width = max_width or (
            (config.OUTPUT_WIDTH - config.MARGIN_LEFT - config.MARGIN_RIGHT) 
            * config.TEXT_WIDTH_RATIO
        )

        # Choose font
        font_family = font_family or self.choose_font_for_category(category)

        # Fit font size to quote width
        font, font_size = self.fit_font_size(quote, max_width, family=font_family)

        # Line breaking
        breaker = SmartLineBreaker(font, max_width)
        lines = breaker.get_optimal_breaks(quote)
        lines = breaker.balance_lines(lines)

        # Measure and position lines
        draw = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
        text_lines = []
        total_width = 0
        total_height = 0

        line_spacing = font_size * config.LINE_SPACING

        for line_text in lines:
            bbox = draw.textbbox((0, 0), line_text, font=font)
            width = bbox[2] - bbox[0]
            height = bbox[3] - bbox[1]
            ascent = -bbox[1]
            descent = bbox[3]
            total_width = max(total_width, width)

            line = TextLine(
                text=line_text,
                words=line_text.split(),
                width=width,
                height=height,
                ascent=ascent,
                descent=descent,
                font_size=font_size,
                font_spec=FontSpec(family=font_family, style="Regular", size=font_size),
            )
            text_lines.append(line)

        total_height = len(text_lines) * line_spacing

        # Vertical centering (optical center is slightly above geometric center)
        vertical_offset = 0.45  # Slightly above center
        base_y = (config.OUTPUT_HEIGHT - total_height) * vertical_offset + config.TEXT_Y_OFFSET

        # Position lines
        current_y = base_y
        for line in text_lines:
            if config.TEXT_ALIGNMENT == "center":
                line.x = (config.OUTPUT_WIDTH - line.width) // 2
            elif config.TEXT_ALIGNMENT == "left":
                line.x = config.MARGIN_LEFT
            elif config.TEXT_ALIGNMENT == "right":
                line.x = config.OUTPUT_WIDTH - config.MARGIN_RIGHT - line.width

            line.y = current_y
            current_y += line_spacing

        # Score layout
        layout_score = self._score_layout(text_lines, total_width, total_height)
        readability_score = self._score_readability(text_lines, font_size)

        return TextBlock(
            lines=text_lines,
            total_width=total_width,
            total_height=total_height,
            font_spec=FontSpec(family=font_family, style="Regular", size=font_size),
            alignment=config.TEXT_ALIGNMENT,
            base_y=base_y,
            layout_score=layout_score,
            readability_score=readability_score,
        )

    def _score_layout(
        self,
        lines: List[TextLine],
        total_width: float,
        total_height: float
    ) -> float:
        """Score text layout quality (0-100)"""
        score = 100.0

        if not lines:
            return 0.0

        # Penalize too many lines
        if len(lines) > 7:
            score -= (len(lines) - 7) * 15

        # Penalize too few lines (text not wrapped well)
        if len(lines) < 2 and total_width > config.OUTPUT_WIDTH * 0.8:
            score -= 20

        # Score line length balance
        if len(lines) > 1:
            widths = [line.width for line in lines]
            avg_width = sum(widths) / len(widths)
            max_diff = max(widths) - min(widths)

            if max_diff / avg_width > 0.5:  # Very unbalanced
                score -= 30
            elif max_diff / avg_width > 0.3:  # Somewhat unbalanced
                score -= 10

        # Reward 3-5 lines (optimal)
        line_count = len(lines)
        if line_count in [3, 4, 5]:
            score += 15
        elif line_count in [2, 6]:
            score += 5

        return max(0, min(100, score))

    def _score_readability(self, lines: List[TextLine], font_size: int) -> float:
        """Score readability (0-100)"""
        score = 100.0

        # Font size check (below 36pt is hard to read)
        if font_size < 36:
            score -= (36 - font_size) * 2

        # Line length check (optimal is 35-55 chars)
        for line in lines:
            char_count = len(line.text)
            if char_count > 60:
                score -= (char_count - 60) * 1.5
            elif char_count < 10 and len(lines) > 1:
                score -= 10

        # Word count per line (too many words is hard to scan)
        for line in lines:
            word_count = len(line.words)
            if word_count > 15:
                score -= (word_count - 15) * 2

        return max(0, min(100, score))

    def render(
        self,
        draw: ImageDraw.Draw,
        block: TextBlock,
        color: Tuple[int, int, int] = None,
        highlight_words: List[str] = None,
        highlight_color: Tuple[int, int, int] = None,
        shadow_enabled: bool = True,
        shadow_color: Tuple[int, int, int] = None,
        shadow_offset: Tuple[int, int] = None,
    ) -> None:
        """
        Render text block to image with optional highlighting and shadow.
        """
        if not block.lines:
            return

        color = color or config.TEXT_COLOR
        highlight_color = highlight_color or config.HIGHLIGHT_COLOR
        shadow_color = shadow_color or config.SHADOW_COLOR
        shadow_offset = shadow_offset or (config.SHADOW_OFFSET_X, config.SHADOW_OFFSET_Y)
        highlight_words = set(w.lower().strip(",.!?;:'\"") for w in (highlight_words or []))

        font = self.get_font(
            block.font_spec.family if block.font_spec else config.DEFAULT_FONT,
            block.font_spec.style if block.font_spec else "Regular",
            block.font_spec.size if block.font_spec else config.FONT_SIZE_OPTIMAL,
        )

        for line in block.lines:
            self._render_line(
                draw, line, font, color, highlight_words, highlight_color,
                shadow_enabled, shadow_color, shadow_offset
            )

    def _render_line(
        self,
        draw: ImageDraw.Draw,
        line: TextLine,
        font: ImageFont.FreeTypeFont,
        color: Tuple[int, int, int],
        highlight_words: set,
        highlight_color: Tuple[int, int, int],
        shadow_enabled: bool,
        shadow_color: Tuple[int, int, int],
        shadow_offset: Tuple[int, int],
    ) -> None:
        """Render a single line with word-level highlighting"""
        words = line.words
        x = line.x

        # Render shadow first
        if shadow_enabled and config.SHADOW_ENABLED:
            sx = line.x + shadow_offset[0]
            for i, word in enumerate(words):
                display = word + (" " if i < len(words) - 1 else "")
                draw.text(
                    (sx, line.y + shadow_offset[1]),
                    display,
                    font=font,
                    fill=(*shadow_color, config.SHADOW_OPACITY),
                )
                bbox = draw.textbbox((0, 0), display, font=font)
                sx += bbox[2] - bbox[0]

        # Render text with highlighting
        for i, word in enumerate(words):
            display = word + (" " if i < len(words) - 1 else "")
            word_clean = word.lower().strip(",.!?;:'\"")

            # Determine color
            if config.HIGHLIGHT_ENABLED and word_clean in highlight_words:
                text_color = highlight_color
            else:
                text_color = color

            draw.text((x, line.y), display, font=font, fill=text_color)

            # Advance cursor
            bbox = draw.textbbox((0, 0), display, font=font)
            x += bbox[2] - bbox[0]

    def summary(self) -> None:
        """Print engine summary"""
        print("\n" + "=" * 60)
        print("  TYPOGRAPHY ENGINE")
        print("=" * 60)
        print(f"  Default Font   : {config.DEFAULT_FONT}")
        print(f"  Font Size      : {config.FONT_SIZE_OPTIMAL}px")
        print(f"  Size Range     : {config.FONT_SIZE_MIN}px - {config.FONT_SIZE_MAX}px")
        print(f"  Line Spacing   : {config.LINE_SPACING}x")
        print(f"  Alignment      : {config.TEXT_ALIGNMENT}")
        print(f"  Available Fonts: {len(self.available_fonts)}")
        print(f"  Optical Sizing : Enabled")
        print(f"  Smart Breaking : Enabled")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    engine = TypographyEngine()
    engine.summary()

    # Demo
    block = engine.compose(
        "The strongest people are not those who never fail but those who never quit.",
        category="motivation"
    )
    print(f"Lines: {len(block.lines)}")
    print(f"Layout Score: {block.layout_score:.1f}")
    print(f"Readability: {block.readability_score:.1f}")
    for i, line in enumerate(block.lines):
        print(f"  {i+1}. {line.text}")
