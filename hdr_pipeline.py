"""
QuoteLab Elite v2 - HDR Pipeline
=================================
Float32 internal rendering, tone mapping, color spaces, and HDR output.
Professional-grade color management.
"""

import numpy as np
from typing import Tuple, Optional
from enum import Enum

import config


class ColorSpace(Enum):
    """Supported color spaces"""
    SRGB = "sRGB"
    DCI_P3 = "DCI-P3"
    BT2020 = "BT.2020"
    ACES = "ACES"


class ToneMapping(Enum):
    """Tone mapping operators"""
    ACES = "aces"
    REINHARD = "reinhard"
    FILMIC = "filmic"
    UNCHARTED2 = "uncharted2"
    NONE = "none"


class HDRToneMapper:
    """
    Professional tone mapping operators.
    All operate on float32 numpy arrays in linear light.
    """

    # ACES fitted curve parameters
    ACES_A = 2.51
    ACES_B = 0.03
    ACES_C = 2.43
    ACES_D = 0.59
    ACES_E = 0.14

    # Uncharted 2 parameters
    U2_A = 0.15
    U2_B = 0.50
    U2_C = 0.10
    U2_D = 0.20
    U2_E = 0.02
    U2_F = 0.30
    U2_W = 11.2

    @classmethod
    def aces(cls, x: np.ndarray) -> np.ndarray:
        """
        ACES filmic tone mapping curve.
        Industry standard for cinematic look.
        """
        return np.clip(
            (x * (cls.ACES_A * x + cls.ACES_B)) /
            (x * (cls.ACES_C * x + cls.ACES_D) + cls.ACES_E),
            0.0, 1.0
        )

    @classmethod
    def reinhard(cls, x: np.ndarray) -> np.ndarray:
        """Reinhard tone mapping - simple and smooth"""
        return x / (1.0 + x)

    @classmethod
    def filmic(cls, x: np.ndarray) -> np.ndarray:
        """Filmic tone mapping with S-curve"""
        return np.where(
            x < 0.5,
            2.0 * x ** 2,
            1.0 - 2.0 * (1.0 - x) ** 2
        )

    @classmethod
    def uncharted2_f(cls, x: np.ndarray) -> np.ndarray:
        """Uncharted 2 helper function"""
        return (
            (x * (cls.U2_A * x + cls.U2_C * cls.U2_B) + cls.U2_D * cls.U2_E) /
            (x * (cls.U2_A * x + cls.U2_B) + cls.U2_D * cls.U2_F)
        ) - cls.U2_E / cls.U2_F

    @classmethod
    def uncharted2(cls, x: np.ndarray) -> np.ndarray:
        """Uncharted 2 tone mapping - game industry standard"""
        tone_mapped = cls.uncharted2_f(x * 2.0)
        white_scale = 1.0 / cls.uncharted2_f(np.array([cls.U2_W]))[0]
        return np.clip(tone_mapped * white_scale, 0.0, 1.0)

    @classmethod
    def apply(
        cls,
        x: np.ndarray,
        method: str = "aces",
        exposure: float = 1.0
    ) -> np.ndarray:
        """Apply tone mapping with exposure adjustment"""
        x = np.asarray(x, dtype=np.float32)
        x = x * exposure

        if method == "aces":
            return cls.aces(x)
        elif method == "reinhard":
            return cls.reinhard(x)
        elif method == "filmic":
            return cls.filmic(x)
        elif method == "uncharted2":
            return cls.uncharted2(x)
        else:
            return np.clip(x, 0.0, 1.0)


class ColorSpaceConverter:
    """
    Color space conversion utilities.
    Handles sRGB, DCI-P3, BT.2020, and ACES conversions.
    """

    # Conversion matrices (RGB to XYZ)
    SRGB_TO_XYZ = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=np.float32)

    P3_TO_XYZ = np.array([
        [0.48657095, 0.26566769, 0.19821729],
        [0.22897457, 0.69173850, 0.07928696],
        [0.00000000, 0.04511338, 1.04394437]
    ], dtype=np.float32)

    BT2020_TO_XYZ = np.array([
        [0.63695805, 0.14461690, 0.16888098],
        [0.26270021, 0.67799807, 0.05930172],
        [0.00000000, 0.02807269, 1.06098506]
    ], dtype=np.float32)

    @classmethod
    def rgb_to_xyz(
        cls,
        rgb: np.ndarray,
        source: str = "sRGB"
    ) -> np.ndarray:
        """Convert RGB to XYZ color space"""
        rgb = np.asarray(rgb, dtype=np.float32)
        
        if source == "sRGB":
            matrix = cls.SRGB_TO_XYZ
        elif source == "DCI-P3":
            matrix = cls.P3_TO_XYZ
        elif source == "BT.2020":
            matrix = cls.BT2020_TO_XYZ
        else:
            matrix = cls.SRGB_TO_XYZ

        return np.dot(rgb, matrix.T)

    @classmethod
    def xyz_to_rgb(
        cls,
        xyz: np.ndarray,
        target: str = "sRGB"
    ) -> np.ndarray:
        """Convert XYZ to RGB color space"""
        xyz = np.asarray(xyz, dtype=np.float32)
        
        if target == "sRGB":
            matrix = np.linalg.inv(cls.SRGB_TO_XYZ)
        elif target == "DCI-P3":
            matrix = np.linalg.inv(cls.P3_TO_XYZ)
        elif target == "BT.2020":
            matrix = np.linalg.inv(cls.BT2020_TO_XYZ)
        else:
            matrix = np.linalg.inv(cls.SRGB_TO_XYZ)

        return np.dot(xyz, matrix.T)

    @classmethod
    def convert(
        cls,
        rgb: np.ndarray,
        source: str = "sRGB",
        target: str = "BT.2020"
    ) -> np.ndarray:
        """Convert between color spaces"""
        xyz = cls.rgb_to_xyz(rgb, source)
        return cls.xyz_to_rgb(xyz, target)


class HDRPipeline:
    """
    Master HDR Pipeline.
    
    Handles:
    - Float32 internal rendering
    - Color space conversion
    - Tone mapping
    - HDR metadata
    """

    def __init__(
        self,
        color_space: str = None,
        tone_mapping: str = None,
        exposure: float = None
    ):
        self.color_space = color_space or config.COLOR_SPACE
        self.tone_mapping = tone_mapping or config.TONE_MAPPING
        self.exposure = exposure or config.TONE_MAPPING_EXPOSURE
        self.tone_mapper = HDRToneMapper()
        self.cs_converter = ColorSpaceConverter()

    def process_float(self, image: np.ndarray) -> np.ndarray:
        """
        Process float32 image through HDR pipeline.
        
        Args:
            image: Float32 array in range [0, inf) in linear light
        
        Returns:
            Float32 array in range [0, 1] with tone mapping applied
        """
        if not isinstance(image, np.ndarray):
            image = np.array(image, dtype=np.float32)
        elif image.dtype != np.float32:
            image = image.astype(np.float32)

        # Ensure non-negative
        image = np.maximum(image, 0.0)

        # Apply tone mapping
        if image.ndim == 3 and image.shape[2] >= 3:
            # RGB image
            result = np.zeros_like(image)
            for c in range(min(3, image.shape[2])):
                result[:, :, c] = self.tone_mapper.apply(
                    image[:, :, c],
                    self.tone_mapping,
                    self.exposure
                )
            # Handle alpha channel if present
            if image.shape[2] > 3:
                result[:, :, 3] = image[:, :, 3]
        elif image.ndim == 2:
            # Grayscale
            result = self.tone_mapper.apply(
                image,
                self.tone_mapping,
                self.exposure
            )
        else:
            result = np.clip(image * self.exposure, 0.0, 1.0)

        return result

    def to_uint8(self, image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """Convert float32 to uint8 with gamma correction"""
        image = np.asarray(image, dtype=np.float32)
        # Apply gamma
        image = np.power(np.clip(image, 0.0, 1.0), 1.0 / gamma)
        # Convert to uint8
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)

    def to_uint16(self, image: np.ndarray, gamma: float = 2.2) -> np.ndarray:
        """Convert float32 to uint16 with gamma correction"""
        image = np.asarray(image, dtype=np.float32)
        image = np.power(np.clip(image, 0.0, 1.0), 1.0 / gamma)
        return np.clip(image * 65535.0, 0, 65535).astype(np.uint16)

    def apply_color_space(
        self,
        image: np.ndarray,
        target_space: str = None
    ) -> np.ndarray:
        """Convert image to target color space"""
        target = target_space or self.color_space
        if target == "sRGB":
            return image

        return self.cs_converter.convert(
            image,
            source="sRGB",
            target=target
        )

    def generate_metadata(self) -> dict:
        """Generate HDR metadata dictionary"""
        return {
            "color_space": self.color_space,
            "tone_mapping": self.tone_mapping,
            "exposure": self.exposure,
            "hdr10": getattr(config, "HDR10_ENABLED", False),
            "hlg": getattr(config, "HLG_ENABLED", False),
        }

    def summary(self) -> None:
        """Print HDR pipeline summary"""
        print("\n" + "=" * 60)
        print("  HDR PIPELINE")
        print("=" * 60)
        print(f"  Color Space  : {self.color_space}")
        print(f"  Tone Mapping : {self.tone_mapping}")
        print(f"  Exposure     : {self.exposure}")
        print(f"  Color Depth  : {config.COLOR_DEPTH}")
        print("=" * 60 + "\n")


class HDRCompositor:
    """
    Layer compositor with HDR support.
    Handles blend modes in linear light.
    """

    @staticmethod
    def normalize(a: np.ndarray) -> np.ndarray:
        """Ensure float32 format"""
        a = np.asarray(a, dtype=np.float32)
        if a.max() > 1.0:
            a = a / 255.0
        return a

    @classmethod
    def alpha_composite(
        cls,
        background: np.ndarray,
        foreground: np.ndarray,
        alpha: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Alpha composite foreground over background"""
        bg = cls.normalize(background)
        fg = cls.normalize(foreground)

        if alpha is None:
            if fg.shape[-1] == 4:
                alpha = fg[:, :, 3:4]
                fg = fg[:, :, :3]
            else:
                alpha = np.ones_like(fg[:, :, :1])
        else:
            alpha = cls.normalize(alpha)
            if alpha.ndim == 2:
                alpha = alpha[:, :, np.newaxis]

        if bg.ndim == 2:
            bg = bg[:, :, np.newaxis]
        if fg.ndim == 2:
            fg = fg[:, :, np.newaxis]

        return bg * (1.0 - alpha) + fg * alpha

    @classmethod
    def screen(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Screen blend mode (additive)"""
        a, b = cls.normalize(a), cls.normalize(b)
        return 1.0 - (1.0 - a) * (1.0 - b)

    @classmethod
    def multiply(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Multiply blend mode"""
        a, b = cls.normalize(a), cls.normalize(b)
        return a * b

    @classmethod
    def overlay(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Overlay blend mode"""
        a, b = cls.normalize(a), cls.normalize(b)
        return np.where(
            a < 0.5,
            2 * a * b,
            1 - 2 * (1 - a) * (1 - b)
        )

    @classmethod
    def soft_light(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Soft light blend mode"""
        a, b = cls.normalize(a), cls.normalize(b)
        return np.where(
            b < 0.5,
            2 * a * b + a ** 2 * (1 - 2 * b),
            2 * a * (1 - b) + np.sqrt(a) * (2 * b - 1)
        )

    @classmethod
    def add(cls, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Linear add (lighten)"""
        a, b = cls.normalize(a), cls.normalize(b)
        return a + b


if __name__ == "__main__":
    pipeline = HDRPipeline()
    pipeline.summary()

    # Test tone mapping curves
    x = np.linspace(0, 4, 1000, dtype=np.float32)
    
    print("\nTone Mapping Curves:")
    for method in ["aces", "reinhard", "filmic", "uncharted2"]:
        y = HDRToneMapper.apply(x, method)
        print(f"  {method:12}: min={y.min():.3f}, max={y.max():.3f}, mean={y.mean():.3f}")
