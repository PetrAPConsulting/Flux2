# Flux2
This script uses the BFL API to generate images with FLUX.2 [PRO] using
structured JSON prompts for precise control over scene composition, colors,
lighting, and more.

SETUP:

Get your API Key from [Black Forest Labs](https://dashboard.bfl.ai/)

Set your API key as an environment variable:
```
bash
export BFL_API_KEY="your_api_key_here"
```
USAGE:
1. Edit the STRUCTURED_PROMPT dictionary in ```flux2.py``` with your desired settings
2. Choose ASPECT_RATIO preset OR set custom WIDTH/HEIGHT
3. Run: python flux2.py

STRUCTURED PROMPT SCHEMA:
```
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
```
RESOLUTION LIMITS:
- Minimum: 64x64 pixels
- Maximum: 4 megapixels (4,000,000 pixels total)
- Dimensions must be multiples of 16
