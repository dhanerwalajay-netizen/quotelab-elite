"""
QuoteLab Elite v2 - Background Engine
=====================================
Professional background generation with gradients, lighting,
procedural effects, and vignetting.
"""

import math
from typing import Tuple, List, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

from PIL import Image, ImageDraw, ImageFilter
import numpy as np

import config


class GradientType(Enum):
    """Gradient types"""
    LINEAR = "linear"
    RADIAL = "radial"
    ANGULAR = "angular"
    MESH = "mesh"


@dataclass
class GradientStop:
    """Color stop in a gradient"""
    position: float  # 0.0 to 1.0
    color: Tuple[int, int, int]


@dataclass
class LightSource:
    """Point light source"""
    x: float  # 0.0 to 1.0 (normalized)
    y: float
    intensity: float  # 0.0 to 1.0
    radius: float  # 0.0 to 1.0
    color: Tuple[int, int, int]
    falloff: str = "smoothstep"  # linear, smoothstep, gaussian


class GradientEngine:
    """
    Professional gradient engine with multi-stop support.
    Uses mathematical formulas instead of interpolation for smooth results.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def linear(
        self,
        stops: List[Tuple[float, Tuple[int, int, int]]],
        angle: float = 180.0
    ) -> Image.Image:
        """
        Create linear gradient at angle.
        
        Args:
            stops: List of (position, color) tuples
            angle: Direction in degrees (0=right, 90=down, 180=left, 270=up)
        """
        # Convert to radians
        rad = math.radians(angle)
        cos_a = math.cos(rad)
        sin_a = math.sin(rad)

        # Create float32 array for precision
        gradient = np.zeros((self.height, self.width, 3), dtype=np.float32)

        # Calculate gradient direction
        max_dist = math.sqrt(self.width ** 2 + self.height ** 2)

        for y in range(self.height):
            for x in range(self.width):
                # Project onto gradient direction
                px = (x - self.width / 2) * cos_a + (y - self.height / 2) * sin_a
                # Normalize to 0-1
                t = (px / max_dist + 0.5)
                t = max(0.0, min(1.0, t))

                # Interpolate color
                color = self._interpolate_stops(t, stops)
                gradient[y, x] = color

        return self._array_to_image(gradient)

    def radial(
        self,
        stops: List[Tuple[float, Tuple[int, int, int]]],
        center: Tuple[float, float] = (0.5, 0.5),
        radius: float = 0.7
    ) -> Image.Image:
        """
        Create radial gradient from center.
        
        Args:
            stops: List of (position, color) tuples
            center: Center position (normalized 0-1)
            radius: Gradient radius (normalized, 0.5 = half image)
        """
        gradient = np.zeros((self.height, self.width, 3), dtype=np.float32)

        center_x = center[0] * self.width
        center_y = center[1] * self.height
        max_radius = radius * math.sqrt(self.width ** 2 + self.height ** 2)

        for y in range(self.height):
            for x in range(self.width):
                # Distance from center
                dx = x - center_x
                dy = y - center_y
                dist = math.sqrt(dx ** 2 + dy ** 2)

                # Normalize to 0-1
                t = dist / max_radius if max_radius > 0 else 0
                t = max(0.0, min(1.0, t))

                # Interpolate color
                color = self._interpolate_stops(t, stops)
                gradient[y, x] = color

        return self._array_to_image(gradient)

    def angular(
        self,
        stops: List[Tuple[float, Tuple[int, int, int]]],
        center: Tuple[float, float] = (0.5, 0.5)
    ) -> Image.Image:
        """
        Create angular/sweep gradient around center.
        
        Args:
            stops: List of (position, color) tuples
            center: Center position (normalized 0-1)
        """
        gradient = np.zeros((self.height, self.width, 3), dtype=np.float32)

        center_x = center[0] * self.width
        center_y = center[1] * self.height

        for y in range(self.height):
            for x in range(self.width):
                # Angle from center
                dx = x - center_x
                dy = y - center_y
                angle = math.atan2(dy, dx)

                # Normalize to 0-1
                t = (angle + math.pi) / (2 * math.pi)

                # Interpolate color
                color = self._interpolate_stops(t, stops)
                gradient[y, x] = color

        return self._array_to_image(gradient)

    def _interpolate_stops(
        self,
        t: float,
        stops: List[Tuple[float, Tuple[int, int, int]]]
    ) -> np.ndarray:
        """Interpolate color at position t along gradient stops"""
        if not stops:
            return np.array([0, 0, 0], dtype=np.float32)

        # Find surrounding stops
        for i in range(len(stops) - 1):
            t1, color1 = stops[i]
            t2, color2 = stops[i + 1]

            if t1 <= t <= t2:
                # Linear interpolation
                if t2 == t1:
                    ratio = 0.0
                else:
                    ratio = (t - t1) / (t2 - t1)

                # Smooth interpolation
                ratio = self._smoothstep(ratio)

                r = color1[0] + (color2[0] - color1[0]) * ratio
                g = color1[1] + (color2[1] - color1[1]) * ratio
                b = color1[2] + (color2[2] - color1[2]) * ratio

                return np.array([r, g, b], dtype=np.float32)

        # Outside range, use first or last color
        if t < stops[0][0]:
            return np.array(stops[0][1], dtype=np.float32)
        else:
            return np.array(stops[-1][1], dtype=np.float32)

    def _smoothstep(self, t: float) -> float:
        """Smooth interpolation function (Hermite)"""
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)

    def _array_to_image(self, array: np.ndarray) -> Image.Image:
        """Convert float32 numpy array to PIL Image"""
        # Clip and convert to uint8
        array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array, "RGB")


class LightingEngine:
    """
    Advanced lighting engine with multiple light sources and falloff.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def apply_light(
        self,
        background: Image.Image,
        lights: List[LightSource]
    ) -> Image.Image:
        """
        Apply multiple light sources to background.
        Uses screen blending (additive).
        """
        if not lights:
            return background

        # Convert to float32 for processing
        bg_array = np.array(background, dtype=np.float32) / 255.0
        result = bg_array.copy()

        for light in lights:
            light_layer = self._render_light(light)
            light_array = np.array(light_layer, dtype=np.float32) / 255.0

            # Screen blend (additive)
            result = 1.0 - (1.0 - result) * (1.0 - light_array)

        # Convert back to image
        result = np.clip(result * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(result, "RGB")

    def _render_light(self, light: LightSource) -> Image.Image:
        """Render a single light source"""
        img = Image.new("RGB", (self.width, self.height), (0, 0, 0))
        array = np.zeros((self.height, self.width, 3), dtype=np.float32)

        center_x = light.x * self.width
        center_y = light.y * self.height
        radius_px = light.radius * math.sqrt(self.width ** 2 + self.height ** 2)

        for y in range(self.height):
            for x in range(self.width):
                dx = x - center_x
                dy = y - center_y
                dist = math.sqrt(dx ** 2 + dy ** 2)

                # Calculate falloff
                if light.falloff == "smoothstep":
                    falloff = self._smoothstep(1.0 - dist / radius_px)
                elif light.falloff == "gaussian":
                    sigma = radius_px / 3.0
                    falloff = math.exp(-(dist ** 2) / (2 * sigma ** 2))
                else:  # linear
                    falloff = max(0.0, 1.0 - dist / radius_px)

                intensity = light.intensity * falloff
                array[y, x] = (
                    light.color[0] * intensity,
                    light.color[1] * intensity,
                    light.color[2] * intensity,
                )

        array = np.clip(array, 0, 255).astype(np.uint8)
        return Image.fromarray(array, "RGB")

    def _smoothstep(self, t: float) -> float:
        """Smooth step function"""
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)


class VignetteEngine:
    """
    Create professional vignette overlays.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def create(
        self,
        strength: float = 0.55,
        color: Tuple[int, int, int] = (0, 0, 0),
        softness: float = 0.5
    ) -> Image.Image:
        """
        Create vignette overlay with smooth falloff.
        
        Args:
            strength: 0.0-1.0, how dark the vignette is
            color: RGB color of vignette
            softness: 0.0-1.0, how soft the edge is
        """
        # Create radial mask
        array = np.zeros((self.height, self.width), dtype=np.float32)

        center_x = self.width / 2
        center_y = self.height / 2
        max_dist = math.sqrt(center_x ** 2 + center_y ** 2)

        for y in range(self.height):
            for x in range(self.width):
                dx = x - center_x
                dy = y - center_y
                dist = math.sqrt(dx ** 2 + dy ** 2)

                # Normalized distance (0 at center, 1 at corners)
                norm_dist = dist / max_dist

                # Softness control
                t = (norm_dist - (1.0 - softness)) / softness
                t = max(0.0, min(1.0, t))

                # Smooth falloff
                vignette = self._smoothstep(t) * strength

                array[y, x] = vignette * 255

        array = array.astype(np.uint8)
        mask = Image.fromarray(array, "L")

        # Create RGBA image
        vignette_img = Image.new("RGBA", (self.width, self.height), (*color, 0))
        vignette_img.putalpha(mask)

        return vignette_img

    def _smoothstep(self, t: float) -> float:
        """Smooth step function"""
        t = max(0.0, min(1.0, t))
        return t * t * (3.0 - 2.0 * t)


class BackgroundEngine:
    """
    Master Background Engine.
    Orchestrates gradient, lighting, and effects.
    """

    def __init__(self):
        self.width = config.RENDER_WIDTH
        self.height = config.RENDER_HEIGHT
        self.gradient_engine = GradientEngine(self.width, self.height)
        self.lighting_engine = LightingEngine(self.width, self.height)
        self.vignette_engine = VignetteEngine(self.width, self.height)

    def render(
        self,
        bg_type: str = None,
        gradient_stops: List[Tuple[float, Tuple[int, int, int]]] = None,
        lights: List[LightSource] = None,
        apply_vignette: bool = None
    ) -> Image.Image:
        """
        Render complete background with all effects.
        
        Pipeline:
        1. Base gradient or solid
        2. Apply lighting
        3. Apply vignette
        """
        bg_type = bg_type or config.BG_TYPE
        gradient_stops = gradient_stops or config.GRADIENT_STOPS
        apply_vignette = apply_vignette if apply_vignette is not None else config.VIGNETTE_ENABLED

        # Step 1: Create base
        if bg_type == "solid":
            bg = Image.new("RGB", (self.width, self.height), config.BG_COLOR)
        elif bg_type == "gradient":
            bg = self._render_gradient(gradient_stops)
        elif bg_type == "procedural":
            bg = self._render_procedural()
        else:
            bg = self._render_gradient(gradient_stops)

        # Step 2: Apply lighting
        if lights:
            bg = self.lighting_engine.apply_light(bg, lights)
        elif config.CENTER_LIGHT_ENABLED:
            # Default center light
            default_light = LightSource(
                x=0.5,
                y=config.CENTER_LIGHT_INTENSITY * 0.4,  # Slightly above center
                intensity=config.CENTER_LIGHT_INTENSITY,
                radius=config.CENTER_LIGHT_RADIUS,
                color=config.CENTER_LIGHT_COLOR,
                falloff=config.CENTER_LIGHT_FALLOFF,
            )
            bg = self.lighting_engine.apply_light(bg, [default_light])

        # Step 3: Apply vignette
        if apply_vignette:
            vignette = self.vignette_engine.create(
                strength=config.VIGNETTE_STRENGTH,
                color=config.VIGNETTE_COLOR if hasattr(config, 'VIGNETTE_COLOR') else (0, 0, 0)
            )
            bg = Image.alpha_composite(
                Image.new("RGBA", bg.size, (*bg.getdata()[0], 255)) if bg.mode == "RGB" else bg.convert("RGBA"),
                vignette
            )

        return bg

    def _render_gradient(
        self,
        stops: List[Tuple[float, Tuple[int, int, int]]]
    ) -> Image.Image:
        """Render gradient based on config"""
        gradient_type = config.GRADIENT_TYPE

        if gradient_type == "linear":
            return self.gradient_engine.linear(stops, angle=180.0)
        elif gradient_type == "radial":
            return self.gradient_engine.radial(
                stops,
                center=(0.5, 0.45),
                radius=0.7
            )
        elif gradient_type == "angular":
            return self.gradient_engine.angular(stops, center=(0.5, 0.5))
        else:
            return self.gradient_engine.radial(stops)

    def _render_procedural(self) -> Image.Image:
        """Render procedural background (future: Perlin noise, fractals, etc.)"""
        # For now, use gradient as fallback
        return self._render_gradient(config.GRADIENT_STOPS)

    def summary(self) -> None:
        """Print engine summary"""
        print("\n" + "=" * 60)
        print("  BACKGROUND ENGINE")
        print("=" * 60)
        print(f"  Type           : {config.BG_TYPE}")
        print(f"  Gradient       : {config.GRADIENT_TYPE}")
        print(f"  Center Light   : {config.CENTER_LIGHT_ENABLED}")
        print(f"  Light Intensity: {config.CENTER_LIGHT_INTENSITY}")
        print(f"  Vignette       : {config.VIGNETTE_ENABLED}")
        print(f"  Vignette Str   : {config.VIGNETTE_STRENGTH}")
        print("=" * 60 + "\n")


if __name__ == "__main__":
    engine = BackgroundEngine()
    engine.summary()

    # Demo
    print("Rendering background...")
    bg = engine.render()
    bg = bg.resize((540, 960))  # Scale down for display
    bg.save(config.OUTPUT / "demo_background.png")
    print(f"Saved to {config.OUTPUT / 'demo_background.png'}")
