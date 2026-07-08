"""
QuoteLab Elite v2 - Effects Engine
===================================
Professional visual effects: bloom, shadows, grain, dithering, particles.
"""

import random
import math
from typing import List, Tuple, Optional
from dataclasses import dataclass

from PIL import Image, ImageDraw, ImageFilter
import numpy as np

import config


@dataclass
class BloomSettings:
    """Bloom effect settings"""
    enabled: bool = True
    small_radius: int = 4
    medium_radius: int = 12
    large_radius: int = 28
    small_opacity: float = 0.45
    medium_opacity: float = 0.25
    large_opacity: float = 0.12
    tint: Tuple[int, int, int] = (255, 255, 255)


@dataclass
class ShadowSettings:
    """Shadow effect settings"""
    enabled: bool = True
    color: Tuple[int, int, int] = (0, 0, 0)
    opacity: int = 100
    offset_x: int = 12
    offset_y: int = 14
    blur: int = 20


class BloomEngine:
    """
    Multi-pass bloom engine for cinematic glow.
    Uses screen blending for additive light effect.
    """

    def __init__(self, settings: BloomSettings = None):
        self.settings = settings or BloomSettings()

    def render(self, text_layer: Image.Image) -> Image.Image:
        """
        Render bloom from text layer.
        Creates three blur passes for smooth, artistic glow.
        """
        if not self.settings.enabled or text_layer.mode != "RGBA":
            return Image.new("RGBA", text_layer.size, (0, 0, 0, 0))

        # Generate three blur passes
        small = text_layer.filter(ImageFilter.GaussianBlur(self.settings.small_radius))
        medium = text_layer.filter(ImageFilter.GaussianBlur(self.settings.medium_radius))
        large = text_layer.filter(ImageFilter.GaussianBlur(self.settings.large_radius))

        # Apply opacity to each pass
        small_alpha = small.getchannel("A").point(
            lambda x: int(x * self.settings.small_opacity)
        )
        medium_alpha = medium.getchannel("A").point(
            lambda x: int(x * self.settings.medium_opacity)
        )
        large_alpha = large.getchannel("A").point(
            lambda x: int(x * self.settings.large_opacity)
        )

        small.putalpha(small_alpha)
        medium.putalpha(medium_alpha)
        large.putalpha(large_alpha)

        # Composite passes (largest first for proper layering)
        bloom = Image.new("RGBA", text_layer.size, (0, 0, 0, 0))
        bloom = Image.alpha_composite(bloom, large)
        bloom = Image.alpha_composite(bloom, medium)
        bloom = Image.alpha_composite(bloom, small)

        # Apply tint if not white
        if self.settings.tint != (255, 255, 255):
            bloom = self._apply_tint(bloom)

        return bloom

    def _apply_tint(self, bloom: Image.Image) -> Image.Image:
        """Apply color tint to bloom"""
        r, g, b, a = bloom.split()
        r = r.point(lambda x: int(x * self.settings.tint[0] / 255))
        g = g.point(lambda x: int(x * self.settings.tint[1] / 255))
        b = b.point(lambda x: int(x * self.settings.tint[2] / 255))
        return Image.merge("RGBA", [r, g, b, a])


class ShadowEngine:
    """
    Soft shadow engine with Gaussian blur.
    """

    def __init__(self, settings: ShadowSettings = None):
        self.settings = settings or ShadowSettings()

    def render(self, text_layer: Image.Image) -> Image.Image:
        """Render soft shadow"""
        if not self.settings.enabled or text_layer.mode != "RGBA":
            return Image.new("RGBA", text_layer.size, (0, 0, 0, 0))

        # Extract alpha channel
        alpha = text_layer.getchannel("A")

        # Create shadow with color
        shadow_r = Image.new("L", text_layer.size, self.settings.color[0])
        shadow_g = Image.new("L", text_layer.size, self.settings.color[1])
        shadow_b = Image.new("L", text_layer.size, self.settings.color[2])

        shadow = Image.merge("RGBA", [shadow_r, shadow_g, shadow_b, alpha])

        # Apply opacity
        shadow_alpha = shadow.getchannel("A").point(
            lambda x: int(x * self.settings.opacity / 255)
        )
        shadow.putalpha(shadow_alpha)

        # Blur
        shadow = shadow.filter(ImageFilter.GaussianBlur(self.settings.blur))

        # Offset
        from PIL import ImageChops
        shadow = ImageChops.offset(
            shadow,
            self.settings.offset_x,
            self.settings.offset_y
        )

        return shadow


class GrainEngine:
    """
    Film grain and dithering engine.
    Adds subtle texture to eliminate banding and add realism.
    """

    def __init__(self, width: int = None, height: int = None):
        self.width = width or config.RENDER_WIDTH
        self.height = height or config.RENDER_HEIGHT
        self.opacity = config.GRAIN_OPACITY
        self.grain_type = config.GRAIN_TYPE

    def render(self) -> Image.Image:
        """Render grain overlay layer"""
        if not config.GRAIN_ENABLED:
            return Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))

        if self.grain_type == "film":
            grain = self._film_grain()
        elif self.grain_type == "digital":
            grain = self._digital_noise()
        elif self.grain_type == "gradient_dither":
            grain = self._gradient_dither()
        else:
            grain = self._film_grain()

        return grain

    def _film_grain(self) -> Image.Image:
        """Simulate analog film grain"""
        noise = np.random.normal(128, self.opacity, (self.height, self.width))
        noise = np.clip(noise, config.GRAIN_MIN, config.GRAIN_MAX).astype(np.uint8)
        gray = Image.fromarray(noise, "L")
        return Image.merge("RGBA", [gray, gray, gray, gray])

    def _digital_noise(self) -> Image.Image:
        """Simulate digital sensor noise"""
        noise = np.random.randint(
            config.GRAIN_MIN, config.GRAIN_MAX + 1,
            (self.height, self.width),
            dtype=np.uint8
        )
        gray = Image.fromarray(noise, "L")
        return Image.merge("RGBA", [gray, gray, gray, gray])

    def _gradient_dither(self) -> Image.Image:
        """
        Subtle dithering specifically for gradients.
        Eliminates banding without adding visible noise.
        """
        # Very subtle noise (1-3 levels)
        noise = np.random.randint(-3, 4, (self.height, self.width), dtype=np.int16)
        noise = np.clip(128 + noise, 0, 255).astype(np.uint8)
        gray = Image.fromarray(noise, "L")
        return Image.merge("RGBA", [gray, gray, gray, np.full((self.height, self.width), 5, dtype=np.uint8)])

    def render_temporal(self, frame: int) -> Image.Image:
        """Render temporal grain for video (changes per frame)"""
        if not config.TEMPORAL_GRAIN:
            return self.render()

        # Seed based on frame
        random.seed(frame * 12345)
        np.random.seed(frame * 12345)

        return self.render()


class ParticleSystem:
    """
    Particle effects: dust, snow, rain, fireflies, sparks, bokeh, etc.
    """

    def __init__(self, width: int = None, height: int = None):
        self.width = width or config.RENDER_WIDTH
        self.height = height or config.RENDER_HEIGHT

    def render(self, frame: int = 0) -> Image.Image:
        """Render particles for a frame"""
        if not config.PARTICLES_ENABLED:
            return Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))

        overlay = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        # Seed for reproducibility
        random.seed(frame * 7 + 12345)

        particle_type = config.PARTICLE_TYPE

        if particle_type == "dust":
            self._render_dust(draw)
        elif particle_type == "snow":
            self._render_snow(draw, frame)
        elif particle_type == "rain":
            self._render_rain(draw, frame)
        elif particle_type == "fireflies":
            self._render_fireflies(draw, frame)
        elif particle_type == "sparks":
            self._render_sparks(draw, frame)
        elif particle_type == "bokeh":
            self._render_bokeh(draw)
        elif particle_type == "stars":
            self._render_stars(draw)
        else:
            self._render_dust(draw)

        return overlay

    def _render_dust(self, draw: ImageDraw.Draw) -> None:
        """Floating dust particles"""
        for _ in range(config.PARTICLE_COUNT):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            size = random.uniform(1, config.PARTICLE_SIZE * 2)
            alpha = int(random.uniform(20, 80) * config.PARTICLE_OPACITY)
            draw.ellipse(
                (x - size, y - size, x + size, y + size),
                fill=(*config.PARTICLE_COLOR, alpha)
            )

    def _render_snow(self, draw: ImageDraw.Draw, frame: int) -> None:
        """Falling snow"""
        for i in range(config.PARTICLE_COUNT):
            x = (random.randint(0, self.width) + frame * config.PARTICLE_SPEED * 0.5 + i * 100) % self.width
            y = (random.randint(0, self.height) + frame * config.PARTICLE_SPEED * 2 + i * 50) % self.height
            size = random.uniform(1, config.PARTICLE_SIZE * 1.5)
            alpha = int(random.uniform(100, 200) * config.PARTICLE_OPACITY)
            draw.ellipse(
                (x - size, y - size, x + size, y + size),
                fill=(255, 255, 255, alpha)
            )

    def _render_rain(self, draw: ImageDraw.Draw, frame: int) -> None:
        """Falling rain streaks"""
        for i in range(config.PARTICLE_COUNT):
            x = (random.randint(0, self.width) + i * 37) % self.width
            y = (random.randint(0, self.height) + frame * config.PARTICLE_SPEED * 8 + i * 73) % self.height
            length = random.uniform(5, 15) * config.PARTICLE_SIZE
            alpha = int(random.uniform(50, 120) * config.PARTICLE_OPACITY)
            draw.line(
                [(x, y), (x - 2, y + length)],
                fill=(200, 210, 230, alpha),
                width=1
            )

    def _render_fireflies(self, draw: ImageDraw.Draw, frame: int) -> None:
        """Glowing fireflies"""
        for i in range(config.PARTICLE_COUNT):
            t = frame * 0.05 + i * 2.5
            x = (self.width * 0.3 + math.sin(t) * self.width * 0.2 + i * 100) % self.width
            y = (self.height * 0.5 + math.cos(t * 0.7) * self.height * 0.2 + i * 80) % self.height
            size = random.uniform(2, 4) * config.PARTICLE_SIZE
            pulse = (math.sin(t * 3) + 1) * 0.5
            alpha = int(200 * pulse * config.PARTICLE_OPACITY)
            draw.ellipse(
                (x - size, y - size, x + size, y + size),
                fill=(255, 230, 100, alpha)
            )

    def _render_sparks(self, draw: ImageDraw.Draw, frame: int) -> None:
        """Rising sparks"""
        for i in range(config.PARTICLE_COUNT):
            x = (self.width // 2 + random.gauss(0, self.width * 0.1) + i * 50) % self.width
            y = (self.height - frame * config.PARTICLE_SPEED * 5 - i * 100) % self.height
            size = random.uniform(1, 3) * config.PARTICLE_SIZE
            alpha = int(random.uniform(150, 255) * config.PARTICLE_OPACITY)
            draw.ellipse(
                (x - size, y - size, x + size, y + size),
                fill=(255, 180, 50, alpha)
            )

    def _render_bokeh(self, draw: ImageDraw.Draw) -> None:
        """Bokeh circles"""
        for i in range(config.PARTICLE_COUNT // 3):
            x = random.randint(0, self.width)
            y = random.randint(0, self.height)
            size = random.uniform(10, 40) * config.PARTICLE_SIZE
            alpha = int(random.uniform(10, 40) * config.PARTICLE_OPACITY)
            color = random.choice([
                (255, 200, 150), (200, 220, 255),
                (255, 255, 200), (220, 180, 255),
            ])
            draw.ellipse(
                (x - size, y - size, x + size, y + size),
                fill=(*color, alpha)
            )

    def _render_stars(self, draw: ImageDraw.Draw) -> None:
        """Twinkling stars"""
        for i in range(config.PARTICLE_COUNT):
            x = (i * 137 + 50) % self.width
            y = (i * 93 + 30) % (self.height // 2)
            size = random.uniform(0.5, 2) * config.PARTICLE_SIZE
            alpha = int(random.uniform(80, 200) * config.PARTICLE_OPACITY)
            draw.ellipse(
                (x - size, y - size, x + size, y + size),
                fill=(255, 255, 240, alpha)
            )


class EffectsEngine:
    """
    Master Effects Engine.
    Orchestrates all visual effects.
    """

    def __init__(self):
        self.bloom = BloomEngine()
        self.shadow = ShadowEngine()
        self.grain = GrainEngine()
        self.particles = ParticleSystem()

    def apply_all(
        self,
        background: Image.Image,
        text_layer: Image.Image,
        frame: int = 0
    ) -> Image.Image:
        """
        Apply all effects in correct order.
        Returns final composited image.
        """
        result = background.convert("RGBA")

        # Apply shadow first (behind text)
        if config.SHADOW_ENABLED:
            shadow = self.shadow.render(text_layer)
            result = Image.alpha_composite(result, shadow)

        # Apply bloom (light effect)
        if config.BLOOM_ENABLED:
            bloom = self.bloom.render(text_layer)
            result = Image.alpha_composite(result, bloom)

        # Apply text
        result = Image.alpha_composite(result, text_layer)

        # Apply grain (subtle, on top)
        if config.GRAIN_ENABLED:
            grain = self.grain.render_temporal(frame)
            result = Image.alpha_composite(result, grain)

        # Apply particles (on top)
        if config.PARTICLES_ENABLED:
            particles = self.particles.render(frame)
            result = Image.alpha_composite(result, particles)

        return result

    def summary(self) -> None:
        """Print effects engine summary"""
        print("\n" + "=" * 60)
        print("  EFFECTS ENGINE")
        print("=" * 60)
        print(f"  Bloom      : {config.BLOOM_ENABLED}")
        print(f"  Shadow     : {config.SHADOW_ENABLED}")
        print(f"  Grain      : {config.GRAIN_ENABLED}")
        print(f"  Particles  : {config.PARTICLES_ENABLED}")
        print(f"  Particle Type : {config.PARTICLE_TYPE}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    engine = EffectsEngine()
    engine.summary()
