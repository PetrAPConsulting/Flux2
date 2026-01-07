#!/usr/bin/env python3
"""
FLUX.2 [PRO] Structured Prompt Image Generation Script
=======================================================

This script uses the BFL API to generate images with FLUX.2 [PRO] using
structured JSON prompts for precise control over scene composition, colors,
lighting, and more.

SETUP:
    Set your API key as an environment variable:
    export BFL_API_KEY="your_api_key_here"

USAGE:
    1. Edit the STRUCTURED_PROMPT dictionary below with your desired settings
    2. Choose ASPECT_RATIO preset OR set custom WIDTH/HEIGHT
    3. Run: python flux_structured.py

STRUCTURED PROMPT SCHEMA:
    {
        "scene": "overall scene description",
        "subjects": [
            {
                "description": "detailed subject description",
                "position": "where in frame",
                "action": "what they're doing"
            }
        ],
        "style": "artistic style",
        "color_palette": ["#hex1", "#hex2", "#hex3"],
        "lighting": "lighting description",
        "mood": "emotional tone",
        "background": "background details",
        "composition": "framing and layout",
        "camera": {
            "angle": "camera angle",
            "lens": "lens type",
            "depth_of_field": "focus behavior"
        },
        "text_elements": [  # Optional: for images containing text
            {
                "content": "exact text to display",
                "style": "font style description",
                "position": "where the text appears",
                "color": "#hex color"
            }
        ]
    }

RESOLUTION LIMITS:
    - Minimum: 64x64 pixels
    - Maximum: 4 megapixels (4,000,000 pixels total)
    - Dimensions must be multiples of 16
"""

import os
import sys
import time
import json
import base64
import requests
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict, Any

# =============================================================================
# CONFIGURATION - EDIT THESE VALUES
# =============================================================================

# Mode: "generate" for new images, "edit" for image manipulation
MODE = "generate"

# Input image filename (only used when MODE = "edit")
INPUT_IMAGE_FILENAME = "input.png"

# -----------------------------------------------------------------------------
# STRUCTURED PROMPT - Edit this JSON structure for precise control
# -----------------------------------------------------------------------------
STRUCTURED_PROMPT: Dict[str, Any] = {
    "scene": "A cozy coffee shop interior on a rainy afternoon",
    
    "subjects": [
        {
            "description": "A young woman with curly auburn hair wearing a modern navy blue business suite",
            "position": "seated at a window table, center-left of frame",
            "action": "reading a vintage hardcover book with title 'COFFEE & CIGARS' while holding a steamy ceramic latte cup"
        },
        {
            "description": "A black cat",
            "position": "curled up on the chair beside her",
            "action": "sleeping peacefully"
        }
    ],
    
    "style": "Cinematic photography with film grain, reminiscent of Kodak Portra 400",
    
    "color_palette": [
        "#000080",  # Navy Blue
        "#8B4513",  # Saddle brown - wooden furniture
        "#F5F5DC",  # Beige - warm tones
        "#2F4F4F",  # Dark slate gray - rainy atmosphere
        "#FFE4B5",  # Moccasin - warm lighting
        "#DEB887"   # Burlywood - coffee tones
    ],
    
    "lighting": "Soft diffused natural light from rain-streaked windows, warm tungsten accent lights from vintage Edison bulbs overhead, creating a golden glow",
    
    "mood": "Peaceful, nostalgic, hygge atmosphere, intimate and contemplative",
    
    "background": "Exposed brick walls with vintage posters, wooden bookshelves filled with old books, other patrons softly blurred, rain visible through large windows",
    
    "composition": "Rule of thirds, subject placed on left third line, depth created through foreground coffee cup on adjacent table, leading lines from wooden floor planks",
    
    "camera": {
        "angle": "Eye level, slightly angled to capture both subject and window",
        "lens": "50mm prime lens equivalent, creating natural perspective",
        "depth_of_field": "Shallow, f/2.0, subject in sharp focus with creamy bokeh in background"
    },
    
    # Optional: Text elements (if your image should contain text)
    "text_elements": [
         {
             "content": "COFFEE & CIGARS",
             "style": "Vintage hand-painted sign lettering",
             "position": "On a wooden sign hanging on the back wall",
             "color": "#2F1810"
         }
    ]
}

# -----------------------------------------------------------------------------
# ALTERNATIVE: Simple text prompt mode
# Set USE_STRUCTURED_PROMPT = False to use a plain text prompt instead
# -----------------------------------------------------------------------------
USE_STRUCTURED_PROMPT = True

SIMPLE_PROMPT = """Your plain text prompt here if not using structured mode"""

# -----------------------------------------------------------------------------
# RESOLUTION SETTINGS
# -----------------------------------------------------------------------------
# Available presets:
#   "1:1"       - Square (1024x1024) - 1MP
#   "1:1_hd"    - Square HD (1536x1536) - 2.4MP
#   "1:1_max"   - Square Max (2048x2048) - 4MP
#   "16:9"      - Landscape HD (1920x1080) - 2MP
#   "16:9_4k"   - Landscape 4K (2560x1440) - 3.7MP
#   "9:16"      - Portrait HD (1080x1920) - 2MP
#   "9:16_4k"   - Portrait 4K (1440x2560) - 3.7MP
#   "4:3"       - Classic Landscape (1536x1152) - 1.8MP
#   "3:4"       - Classic Portrait (1152x1536) - 1.8MP
#   "3:2"       - Photo Landscape (1536x1024) - 1.6MP
#   "2:3"       - Photo Portrait (1024x1536) - 1.6MP
#   "21:9"      - Ultrawide (2016x864) - 1.7MP
#   "9:21"      - Ultra Tall (864x2016) - 1.7MP

ASPECT_RATIO = "4:3"  # Set to None to use custom WIDTH/HEIGHT

# Option 2: Custom dimensions (only used if ASPECT_RATIO is None)
WIDTH = 1536
HEIGHT = 1152

# -----------------------------------------------------------------------------
# OTHER SETTINGS
# -----------------------------------------------------------------------------
OUTPUT_FORMAT = "png"  # "jpeg" or "png"
SEED = None  # Set integer for reproducible results
SAFETY_TOLERANCE = 5  # 0 = most strict, 5 = least strict

# Save the structured prompt JSON alongside the image
SAVE_PROMPT_JSON = True

# =============================================================================
# ASPECT RATIO PRESETS
# =============================================================================

ASPECT_RATIO_PRESETS = {
    "1:1": (1024, 1024),
    "1:1_hd": (1536, 1536),
    "1:1_max": (2048, 2048),
    "16:9": (1920, 1080),
    "16:9_4k": (2560, 1440),
    "9:16": (1080, 1920),
    "9:16_4k": (1440, 2560),
    "4:3": (1536, 1152),
    "3:4": (1152, 1536),
    "3:2": (1536, 1024),
    "2:3": (1024, 1536),
    "21:9": (2016, 864),
    "9:21": (864, 2016),
}

# =============================================================================
# API CONFIGURATION
# =============================================================================

API_BASE_URL = "https://api.bfl.ai"
FLUX2_PRO_ENDPOINT = "/v1/flux-2-pro"
RESULT_ENDPOINT = "/v1/get_result"

MIN_DIMENSION = 64
MAX_MEGAPIXELS = 4_000_000

MAX_POLL_ATTEMPTS = 120
POLL_INTERVAL = 2


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def hex_to_color_name(hex_color: str) -> str:
    """Convert hex color to approximate descriptive name."""
    # Remove # if present
    hex_color = hex_color.lstrip('#')
    
    # Convert to RGB
    try:
        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
    except (ValueError, IndexError):
        return hex_color
    
    # Basic color categorization
    if r > 200 and g > 200 and b > 200:
        return "white/cream"
    if r < 50 and g < 50 and b < 50:
        return "black/dark"
    if r > g and r > b:
        if r > 200:
            return "bright red/coral"
        return "red/burgundy"
    if g > r and g > b:
        if g > 200:
            return "bright green/lime"
        return "green/forest"
    if b > r and b > g:
        if b > 200:
            return "bright blue/sky"
        return "blue/navy"
    if r > 180 and g > 150 and b < 100:
        return "golden/amber"
    if r > 150 and g > 100 and b < 80:
        return "brown/tan"
    if r > 200 and g > 150 and b > 150:
        return "pink/rose"
    if r > 100 and g > 100 and b > 100:
        return "gray"
    
    return f"({hex_color})"


def build_prompt_from_structure(structured: Dict[str, Any]) -> str:
    """
    Convert a structured prompt dictionary into a detailed text prompt
    optimized for FLUX.2 [PRO].
    """
    sections = []
    
    # Scene description (primary)
    if "scene" in structured and structured["scene"]:
        sections.append(f"Scene: {structured['scene']}")
    
    # Subjects with details
    if "subjects" in structured and structured["subjects"]:
        subject_descriptions = []
        for i, subject in enumerate(structured["subjects"], 1):
            parts = []
            if "description" in subject:
                parts.append(subject["description"])
            if "position" in subject:
                parts.append(f"positioned {subject['position']}")
            if "action" in subject:
                parts.append(subject["action"])
            
            if parts:
                subject_descriptions.append(f"Subject {i}: " + ", ".join(parts))
        
        if subject_descriptions:
            sections.append("\n".join(subject_descriptions))
    
    # Style
    if "style" in structured and structured["style"]:
        sections.append(f"Style: {structured['style']}")
    
    # Color palette with descriptions
    if "color_palette" in structured and structured["color_palette"]:
        colors = structured["color_palette"]
        color_descriptions = [f"{c} ({hex_to_color_name(c)})" for c in colors]
        sections.append(f"Color palette: {', '.join(color_descriptions)}")
    
    # Lighting
    if "lighting" in structured and structured["lighting"]:
        sections.append(f"Lighting: {structured['lighting']}")
    
    # Mood
    if "mood" in structured and structured["mood"]:
        sections.append(f"Mood and atmosphere: {structured['mood']}")
    
    # Background
    if "background" in structured and structured["background"]:
        sections.append(f"Background: {structured['background']}")
    
    # Composition
    if "composition" in structured and structured["composition"]:
        sections.append(f"Composition: {structured['composition']}")
    
    # Camera settings
    if "camera" in structured and structured["camera"]:
        cam = structured["camera"]
        cam_parts = []
        if "angle" in cam:
            cam_parts.append(f"Camera angle: {cam['angle']}")
        if "lens" in cam:
            cam_parts.append(f"Lens: {cam['lens']}")
        if "depth_of_field" in cam:
            cam_parts.append(f"Depth of field: {cam['depth_of_field']}")
        if cam_parts:
            sections.append(" | ".join(cam_parts))
    
    # Text elements
    if "text_elements" in structured and structured["text_elements"]:
        text_parts = []
        for text in structured["text_elements"]:
            text_desc = []
            if "content" in text:
                text_desc.append(f'Text reading "{text["content"]}"')
            if "style" in text:
                text_desc.append(f"in {text['style']}")
            if "position" in text:
                text_desc.append(f"located {text['position']}")
            if "color" in text:
                text_desc.append(f"in {hex_to_color_name(text['color'])} color")
            if text_desc:
                text_parts.append(" ".join(text_desc))
        
        if text_parts:
            sections.append("Text elements: " + "; ".join(text_parts))
    
    # Build the main prompt
    main_prompt = "\n\n".join(sections)
    
    return main_prompt


def validate_structured_prompt(structured: Dict[str, Any]) -> List[str]:
    """Validate the structured prompt and return any warnings."""
    warnings = []
    
    if not structured.get("scene"):
        warnings.append("No 'scene' description provided - consider adding one for better results")
    
    if not structured.get("subjects"):
        warnings.append("No 'subjects' defined - the image may lack a clear focal point")
    
    if structured.get("color_palette"):
        for color in structured["color_palette"]:
            if not color.startswith("#") or len(color) != 7:
                warnings.append(f"Invalid hex color format: {color} (should be #RRGGBB)")
    
    if structured.get("text_elements"):
        for text in structured["text_elements"]:
            if "content" in text and len(text["content"]) > 50:
                warnings.append(f"Text element '{text['content'][:20]}...' is long - AI may struggle with lengthy text")
    
    return warnings


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_api_key():
    """Get API key from environment variable."""
    api_key = os.environ.get("BFL_API_KEY")
    if not api_key:
        print("ERROR: BFL_API_KEY environment variable not set.")
        print("Please set it with: export BFL_API_KEY='your_api_key_here'")
        sys.exit(1)
    return api_key


def get_script_directory():
    """Get the directory where this script is located."""
    return Path(__file__).parent.resolve()


def round_to_multiple_of_16(value):
    """Round a value to the nearest multiple of 16."""
    return round(value / 16) * 16


def validate_dimensions(width, height):
    """Validate and adjust dimensions to meet API requirements."""
    errors = []
    warnings = []
    
    if width < MIN_DIMENSION:
        errors.append(f"Width {width} is below minimum ({MIN_DIMENSION})")
    if height < MIN_DIMENSION:
        errors.append(f"Height {height} is below minimum ({MIN_DIMENSION})")
    
    total_pixels = width * height
    if total_pixels > MAX_MEGAPIXELS:
        errors.append(
            f"Total pixels ({total_pixels:,}) exceeds maximum ({MAX_MEGAPIXELS:,} = 4MP). "
            f"Current: {total_pixels/1_000_000:.2f}MP"
        )
    
    adjusted_width = width
    adjusted_height = height
    
    if width % 16 != 0:
        adjusted_width = round_to_multiple_of_16(width)
        warnings.append(f"Width adjusted from {width} to {adjusted_width} (must be multiple of 16)")
    
    if height % 16 != 0:
        adjusted_height = round_to_multiple_of_16(height)
        warnings.append(f"Height adjusted from {height} to {adjusted_height} (must be multiple of 16)")
    
    if adjusted_width < MIN_DIMENSION:
        adjusted_width = MIN_DIMENSION
        warnings.append(f"Width set to minimum: {MIN_DIMENSION}")
    
    if adjusted_height < MIN_DIMENSION:
        adjusted_height = MIN_DIMENSION
        warnings.append(f"Height set to minimum: {MIN_DIMENSION}")
    
    return adjusted_width, adjusted_height, errors, warnings


def get_dimensions():
    """Get and validate dimensions based on configuration."""
    if ASPECT_RATIO is not None:
        if ASPECT_RATIO not in ASPECT_RATIO_PRESETS:
            print(f"ERROR: Unknown aspect ratio '{ASPECT_RATIO}'")
            print(f"Available presets: {', '.join(ASPECT_RATIO_PRESETS.keys())}")
            sys.exit(1)
        width, height = ASPECT_RATIO_PRESETS[ASPECT_RATIO]
        print(f"Using preset: {ASPECT_RATIO} ({width}x{height})")
    else:
        width, height = WIDTH, HEIGHT
        print(f"Using custom dimensions: {width}x{height}")
    
    adj_width, adj_height, errors, warnings = validate_dimensions(width, height)
    
    for warning in warnings:
        print(f"WARNING: {warning}")
    
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        sys.exit(1)
    
    megapixels = (adj_width * adj_height) / 1_000_000
    print(f"Final dimensions: {adj_width}x{adj_height} ({megapixels:.2f}MP)")
    
    return adj_width, adj_height


def encode_image_to_base64(image_path):
    """Encode an image file to base64 data URL."""
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    
    extension = image_path.suffix.lower()
    mime_types = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime_type = mime_types.get(extension, "image/jpeg")
    
    return f"data:{mime_type};base64,{image_data}"


def submit_task(api_key, prompt, width, height, input_image_path=None):
    """Submit a generation or edit task to the API."""
    headers = {
        "x-key": api_key,
        "Content-Type": "application/json",
    }
    
    payload = {
        "prompt": prompt,
        "width": width,
        "height": height,
        "output_format": OUTPUT_FORMAT,
        "safety_tolerance": SAFETY_TOLERANCE,
    }
    
    if SEED is not None:
        payload["seed"] = SEED
    
    if input_image_path:
        print(f"Encoding input image: {input_image_path}")
        payload["input_image"] = encode_image_to_base64(input_image_path)
    
    url = f"{API_BASE_URL}{FLUX2_PRO_ENDPOINT}"
    print(f"Submitting task to {url}...")
    
    response = requests.post(url, headers=headers, json=payload)
    
    if response.status_code != 200:
        print(f"ERROR: API returned status {response.status_code}")
        print(f"Response: {response.text}")
        sys.exit(1)
    
    return response.json()


def poll_for_result(api_key, task_id):
    """Poll the API until the task is complete."""
    headers = {
        "x-key": api_key,
    }
    
    url = f"{API_BASE_URL}{RESULT_ENDPOINT}"
    
    for attempt in range(MAX_POLL_ATTEMPTS):
        print(f"Polling for result (attempt {attempt + 1}/{MAX_POLL_ATTEMPTS})...")
        
        response = requests.get(url, headers=headers, params={"id": task_id})
        
        if response.status_code != 200:
            print(f"WARNING: Poll returned status {response.status_code}")
            time.sleep(POLL_INTERVAL)
            continue
        
        result = response.json()
        status = result.get("status")
        
        if status == "Ready":
            print("Task completed successfully!")
            return result
        elif status == "Error":
            print(f"ERROR: Task failed with error: {result.get('error', 'Unknown error')}")
            sys.exit(1)
        elif status in ["Pending", "Processing", "Queued"]:
            print(f"Status: {status}...")
            time.sleep(POLL_INTERVAL)
        else:
            print(f"Unknown status: {status}")
            time.sleep(POLL_INTERVAL)
    
    print("ERROR: Maximum polling attempts reached. Task may still be processing.")
    sys.exit(1)


def download_and_save_image(image_url, output_path):
    """Download the generated image and save it."""
    print(f"Downloading image from {image_url}...")
    
    response = requests.get(image_url)
    
    if response.status_code != 200:
        print(f"ERROR: Failed to download image. Status: {response.status_code}")
        sys.exit(1)
    
    with open(output_path, "wb") as f:
        f.write(response.content)
    
    print(f"Image saved to: {output_path}")


def save_prompt_metadata(output_path: Path, structured_prompt: Dict, final_prompt: str):
    """Save the prompt metadata as JSON alongside the image."""
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "structured_prompt": structured_prompt,
        "generated_prompt": final_prompt,
        "settings": {
            "aspect_ratio": ASPECT_RATIO,
            "output_format": OUTPUT_FORMAT,
            "seed": SEED,
            "safety_tolerance": SAFETY_TOLERANCE,
        }
    }
    
    json_path = output_path.with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"Prompt metadata saved to: {json_path}")


# =============================================================================
# EXAMPLE PROMPT TEMPLATES
# =============================================================================

EXAMPLE_PROMPTS = {
    "product_photography": {
        "scene": "Minimalist product photography setup",
        "subjects": [
            {
                "description": "A sleek matte black wireless headphone",
                "position": "center of frame, slightly angled",
                "action": "resting on a geometric concrete pedestal"
            }
        ],
        "style": "High-end commercial product photography, clean and modern",
        "color_palette": ["#1a1a1a", "#ffffff", "#c0c0c0", "#2d2d2d"],
        "lighting": "Three-point studio lighting with soft key light from upper left, fill from right, and rim light from behind creating subtle edge highlights",
        "mood": "Premium, sophisticated, minimalist",
        "background": "Seamless gradient from light gray to white",
        "composition": "Product centered with generous negative space, shot from slight three-quarter angle",
        "camera": {
            "angle": "Eye level, 30-degree rotation",
            "lens": "100mm macro for product detail",
            "depth_of_field": "Deep focus, f/8, entire product sharp"
        }
    },
    
    "fantasy_landscape": {
        "scene": "Ancient floating islands above a mystical sea at twilight",
        "subjects": [
            {
                "description": "A massive floating island with waterfalls cascading into the clouds",
                "position": "upper center of frame",
                "action": "hovering majestically with ancient ruins visible"
            },
            {
                "description": "A small sailing ship with glowing sails",
                "position": "lower third, sailing between islands",
                "action": "navigating through the cloud sea"
            }
        ],
        "style": "Digital matte painting, epic fantasy art style inspired by Studio Ghibli and classical romanticism",
        "color_palette": ["#1e3a5f", "#ff7e47", "#9b4dca", "#ffd700", "#2e8b57"],
        "lighting": "Golden hour twilight with the sun setting behind the main island, creating dramatic god rays through the clouds",
        "mood": "Awe-inspiring, adventurous, dreamlike wonder",
        "background": "Endless sea of clouds with distant floating islands, stars beginning to appear in the darkening sky",
        "composition": "Epic wide shot with strong vertical elements, rule of thirds placing main island on upper intersection",
        "camera": {
            "angle": "Low angle looking up at the floating islands",
            "lens": "Wide angle 24mm for epic scale",
            "depth_of_field": "Deep focus with atmospheric perspective for depth"
        }
    },
    
    "portrait": {
        "scene": "Environmental portrait in an artist's studio",
        "subjects": [
            {
                "description": "An elderly Japanese ceramicist with weathered hands and kind eyes, wearing a traditional indigo work apron",
                "position": "right third of frame, three-quarter view",
                "action": "carefully shaping a clay vessel on a potter's wheel"
            }
        ],
        "style": "Documentary portrait photography, natural and authentic",
        "color_palette": ["#3d5a80", "#8b7355", "#f4e4c1", "#2c2c2c", "#c17f59"],
        "lighting": "Natural light from a large north-facing window, soft and even with subtle shadows defining facial features",
        "mood": "Contemplative, dignified, timeless craftsmanship",
        "background": "Shelves of finished pottery, raw clay, and traditional tools slightly out of focus",
        "composition": "Environmental portrait showing both subject and their workspace, shallow depth isolating the artist",
        "camera": {
            "angle": "Slightly below eye level, showing respect",
            "lens": "85mm portrait lens",
            "depth_of_field": "Shallow f/2.8, eyes sharp, background softly blurred"
        }
    },
    
    "text_poster": {
        "scene": "Vintage movie poster design",
        "subjects": [
            {
                "description": "A noir detective in a trench coat and fedora",
                "position": "lower half of frame, dramatic upward angle",
                "action": "lighting a cigarette, face half in shadow"
            }
        ],
        "style": "1940s film noir movie poster, dramatic illustration style",
        "color_palette": ["#1a1a2e", "#e94560", "#0f3460", "#f1c40f"],
        "lighting": "Harsh single light source from above right, creating dramatic shadows and high contrast",
        "mood": "Mysterious, dangerous, suspenseful",
        "background": "Rain-slicked city streets with neon signs reflected, Art Deco buildings",
        "composition": "Vertical poster format with dramatic diagonal composition",
        "camera": {
            "angle": "Low dramatic angle looking up",
            "lens": "Wide angle for distortion",
            "depth_of_field": "Stylized illustration, not photographic"
        },
        "text_elements": [
            {
                "content": "SHADOWS OF DECEIT",
                "style": "Bold Art Deco typography with subtle 3D effect",
                "position": "Top third of poster, arched",
                "color": "#f1c40f"
            },
            {
                "content": "COMING FALL 1947",
                "style": "Smaller condensed sans-serif",
                "position": "Bottom of poster",
                "color": "#ffffff"
            }
        ]
    }
}


def print_example_prompts():
    """Print available example prompt templates."""
    print("\n" + "=" * 60)
    print("AVAILABLE EXAMPLE PROMPT TEMPLATES")
    print("=" * 60)
    
    for name, prompt in EXAMPLE_PROMPTS.items():
        print(f"\n{name.upper().replace('_', ' ')}")
        print("-" * 40)
        print(f"Scene: {prompt.get('scene', 'N/A')}")
        print(f"Style: {prompt.get('style', 'N/A')}")
        print(f"Mood: {prompt.get('mood', 'N/A')}")
    
    print("\n" + "=" * 60)
    print("To use an example, copy it to STRUCTURED_PROMPT variable")
    print("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main function to run the image generation/editing task."""
    print("=" * 60)
    print("FLUX.2 [PRO] Structured Prompt Image Generation")
    print("=" * 60)
    
    # Get API key
    api_key = get_api_key()
    print("API key loaded successfully.")
    
    # Get script directory
    script_dir = get_script_directory()
    
    # Get and validate dimensions
    width, height = get_dimensions()
    
    # Build the prompt
    if USE_STRUCTURED_PROMPT:
        print("\nUsing STRUCTURED PROMPT mode")
        print("-" * 40)
        
        # Validate the structured prompt
        warnings = validate_structured_prompt(STRUCTURED_PROMPT)
        for warning in warnings:
            print(f"WARNING: {warning}")
        
        # Build the text prompt from structure
        final_prompt = build_prompt_from_structure(STRUCTURED_PROMPT)
        
        print("\nGenerated prompt from structure:")
        print("-" * 40)
        print(final_prompt)
        print("-" * 40)
    else:
        print("\nUsing SIMPLE PROMPT mode")
        final_prompt = SIMPLE_PROMPT
    
    # Prepare input image path for edit mode
    input_image_path = None
    if MODE == "edit":
        input_image_path = script_dir / INPUT_IMAGE_FILENAME
        if not input_image_path.exists():
            print(f"ERROR: Input image not found: {input_image_path}")
            print(f"Please place your image in: {script_dir}")
            sys.exit(1)
        print(f"Mode: EDIT (using input image: {INPUT_IMAGE_FILENAME})")
    else:
        print("Mode: GENERATE (creating new image)")
    
    print(f"Output format: {OUTPUT_FORMAT}")
    print("-" * 60)
    
    # Submit the task
    task_response = submit_task(api_key, final_prompt, width, height, input_image_path)
    task_id = task_response.get("id")
    
    if not task_id:
        print("ERROR: No task ID received from API.")
        print(f"Response: {task_response}")
        sys.exit(1)
    
    print(f"Task submitted. ID: {task_id}")
    
    if "cost" in task_response and task_response["cost"] is not None:
        print(f"Cost: {task_response['cost']} credits")
    if "output_mp" in task_response and task_response["output_mp"] is not None:
        print(f"Output: {task_response['output_mp']}MP")
    
    # Poll for result
    result = poll_for_result(api_key, task_id)
    
    # Get the image URL
    image_url = result.get("result", {}).get("sample")
    
    if not image_url:
        print("ERROR: No image URL in result.")
        print(f"Result: {result}")
        sys.exit(1)
    
    # Generate output filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_prefix = "edited" if MODE == "edit" else "generated"
    output_filename = f"{mode_prefix}_{timestamp}.{OUTPUT_FORMAT}"
    output_path = script_dir / output_filename
    
    # Download and save the image
    download_and_save_image(image_url, output_path)
    
    # Save prompt metadata if enabled
    if SAVE_PROMPT_JSON and USE_STRUCTURED_PROMPT:
        save_prompt_metadata(output_path, STRUCTURED_PROMPT, final_prompt)
    
    print("-" * 60)
    print("DONE!")
    print(f"Output saved to: {output_path}")


if __name__ == "__main__":
    # Uncomment to see example prompt templates:
    # print_example_prompts()
    # sys.exit(0)
    
    main()
