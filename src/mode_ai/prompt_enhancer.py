"""
Prompt enhancer cho FLUX - FLUX hiểu prompt tự nhiên tốt hơn tag soup
"""
from typing import List, Dict, Optional
import random

class PromptEnhancer:
    """Enhance prompt cho FLUX.1-schnell"""
    
    # Style presets
    STYLES = {
        "aesthetic_anime": {
            "prefix": "masterpiece, best quality, very aesthetic,",
            "suffix": "ultra detailed, sharp focus, vibrant colors, soft lighting",
            "negative": "low quality, blurry, deformed, bad anatomy"
        },
        "photorealistic": {
            "prefix": "photorealistic, ultra realistic, 8k, masterpiece, best quality,",
            "suffix": "natural skin texture, detailed skin, pores visible, sharp focus, shot on Sony A7R IV, 85mm f/1.4, natural lighting",
            "negative": "anime, cartoon, illustration, painting, blurry, low quality, deformed, oversaturated, artificial skin"
        },
        "realistic_vietnamese": {
            "prefix": "photorealistic, ultra realistic, 8k, masterpiece,",
            "suffix": "beautiful Vietnamese woman, natural skin texture, natural makeup, candid photo, bokeh background, detailed face",
            "negative": "anime, cartoon, deformed, extra fingers, bad anatomy"
        },
        "cinematic": {
            "prefix": "cinematic, film still, masterpiece,",
            "suffix": "dramatic lighting, shallow depth of field, highly detailed, 8k, volumetric lighting",
            "negative": "blurry, low quality, cartoon"
        },
        "portrait": {
            "prefix": "portrait photography, masterpiece, best quality,",
            "suffix": "detailed face, sharp eyes, soft natural lighting, bokeh, 85mm lens",
            "negative": "full body, deformed, blurry"
        },
        "full_body": {
            "prefix": "masterpiece, best quality, full body portrait,",
            "suffix": "detailed face, detailed hands, five fingers, full body, standing pose, detailed clothing, sharp focus",
            "negative": "extra fingers, mutated hands, bad anatomy, cropped, blurry"
        }
    }
    
    # Quality boosters cho FLUX
    QUALITY_TAGS = [
        "masterpiece",
        "best quality",
        "very aesthetic",
        "ultra detailed",
        "sharp focus",
        "high resolution"
    ]
    
    # Detail tags
    DETAIL_TAGS = {
        "face": "detailed face, detailed eyes, beautiful eyes",
        "hands": "detailed hands, five fingers, natural hands",
        "skin": "detailed skin, natural skin texture",
        "lighting": "soft lighting, natural lighting, volumetric lighting",
    }
    
    def __init__(self, style: str = "aesthetic_anime"):
        self.style = style
        if style not in self.STYLES:
            raise ValueError(f"Style {style} not found. Available: {list(self.STYLES.keys())}")
    
    def enhance(self, prompt: str, style: Optional[str] = None, add_quality: bool = True, add_details: List[str] = None) -> str:
        """Enhance prompt với style và quality tags"""
        style = style or self.style
        style_config = self.STYLES[style]
        
        # Clean prompt
        prompt = prompt.strip().strip(",")
        
        # Build enhanced prompt - FLUX thích câu tự nhiên
        parts = []
        
        # Add prefix if not already present
        if style_config["prefix"] not in prompt:
            parts.append(style_config["prefix"])
        
        parts.append(prompt)
        
        # Add detail tags
        if add_details:
            for detail in add_details:
                if detail in self.DETAIL_TAGS:
                    parts.append(self.DETAIL_TAGS[detail])
        
        # Add suffix
        if style_config["suffix"] not in prompt:
            parts.append(style_config["suffix"])
        
        # Join with comma for aesthetic style, but keep natural for photorealistic
        if "photorealistic" in style or "realistic" in style:
            # More natural sentence structure for realistic
            enhanced = ", ".join(parts)
        else:
            enhanced = ", ".join(parts)
        
        # Clean double commas
        enhanced = enhanced.replace(",,", ",").strip().strip(",")
        
        return enhanced
    
    def get_negative(self, style: Optional[str] = None, extra_negative: str = "") -> str:
        """Get negative prompt (FLUX không cần negative phức tạp nhưng vẫn hữu ích)"""
        style = style or self.style
        base_negative = self.STYLES[style]["negative"]
        
        if extra_negative:
            return f"{base_negative}, {extra_negative}"
        return base_negative
    
    def generate_variations(self, base_prompt: str, count: int = 3) -> List[str]:
        """Tạo variations của prompt"""
        variations = []
        
        lighting_options = [
            "soft afternoon sunlight",
            "golden hour lighting",
            "studio lighting",
            "natural window light",
            "cinematic lighting"
        ]
        
        background_options = [
            "standing under cherry blossoms",
            "in a cozy cafe",
            "in a beautiful garden",
            "urban street background",
            "minimalist background"
        ]
        
        for i in range(count):
            lighting = random.choice(lighting_options)
            bg = random.choice(background_options)
            variation = f"{base_prompt}, {lighting}, {bg}"
            variations.append(self.enhance(variation))
        
        return variations
    
    @classmethod
    def list_styles(cls) -> List[str]:
        return list(cls.STYLES.keys())
    
    @classmethod
    def get_style_info(cls, style: str) -> Dict:
        return cls.STYLES.get(style, {})

# Preset prompts cho testing
PRESET_PROMPTS = {
    "girl_cherry": "1girl, long silver hair, aqua eyes, school uniform, standing under cherry blossoms, soft afternoon sunlight, detailed face, detailed hands, five fingers, full body",
    "vietnamese_aodai": "beautiful Vietnamese woman, 20 years old, long black hair, brown eyes, wearing white ao dai, standing in Hanoi old quarter, morning sunlight, natural makeup",
    "portrait_closeup": "close-up portrait of a beautiful woman, detailed face, sharp eyes, natural skin texture, soft lighting, bokeh background",
    "full_body_fashion": "full body fashion photography, 1girl, stylish outfit, standing pose, urban background, detailed clothing, high fashion",
    "anime_aesthetic": "1girl, long wavy hair, beautiful detailed eyes, school uniform, classroom background, soft lighting, very aesthetic",
}

def enhance_prompt_quick(prompt: str, style: str = "aesthetic_anime") -> tuple[str, str]:
    """Quick function để enhance prompt"""
    enhancer = PromptEnhancer(style=style)
    positive = enhancer.enhance(prompt)
    negative = enhancer.get_negative()
    return positive, negative
