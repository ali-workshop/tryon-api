import io
from PIL import Image
from google.genai import types
from app.gemini_client import get_gemini_client

PROMPT = """
# **Half-Body Garment Replacement Prompt (Strict Garment Preservation)**
**TASK:**
Apply the garment from the second image onto the model in the first image.
This prompt is ONLY for **half-body clothing replacement** (upper-body or lower-body).
All original clothing on the model must be fully removed.

---

## **CRITICAL REQUIREMENTS (STRICT RULES)**

### **1. Identity Preservation — ABSOLUTE**
- Preserve 100% of the model’s face, expression, hair, skin tone, body shape, and pose.
- No beautification, morphing, or alteration.

### **2. Background Preservation — ABSOLUTE**
- Keep the original background, lighting, shadows, and environment fully unchanged.

---

## **3. Garment Integrity — DO NOT CHANGE THE GARMENT UNDER ANY CIRCUMSTANCE**
**The garment MUST appear EXACTLY the same after try-on as in the original garment image.
NO editing of the garment is allowed. NO redesign. NO cleanup. NO improvements.**

This includes preserving:
- Original colors (NO color correction or tone matching)
- Original textures (fabric weave, thread, material, shine)
- Original shape and silhouette (NO reshaping or smoothing)
- Original patterns and graphics
- Logos, prints, icons, badges
- Wrinkles, folds, creases
- Shadows and lighting that exist on the garment
- Fabric wear, imperfections, distortions
- **Watermarks or brand stamps**
- Tags, labels, embroidery
- Any visible flaws, marks, or photographer artifacts

**The garment must be transferred AS-IS. Zero modification. Zero reinterpretation. Zero enhancement.**

---

## **4. Half-Body Clothing Replacement — ABSOLUTE**
- Only modify the clothing area in the visible half-body region.
- Remove the original clothing entirely with no trace.
- The new garment must conform naturally to the model’s pose while **keeping ALL garment details unchanged**.

---

## **PHOTOREALISTIC QUALITY (8K / Ultra-Realistic)**
- Maximum sharpness, no blur.
- Realistic fabric tension and draping.
- Lighting consistent with the original background.
- Clean blending at garment–skin boundaries.

---

## **NEGATIVE INSTRUCTIONS — AVOID AT ALL COSTS**
- Any changes to the garment’s design
- Repainting, recoloring, adjusting, or “enhancing” the garment
- Removing wrinkles, watermarks, flaws, folds, shadows
- Wrong face, wrong pose, wrong proportions
- Background changes
- Pattern or logo distortion
- CGI, anime, fake-looking textures
- Body warping, unnatural edges

---

## **FINAL DOUBLE-CHECK (MANDATORY)**
Before generating the final output, DOUBLE-CHECK that:
1. **The garment is transferred EXACTLY as in the garment image with ZERO EDITING.**
2. **Every detail, watermark, wrinkle, color, pattern, and texture is preserved perfectly.**
3. **The garment is worn naturally and correctly on the model with no distortion or redesign.**
4. **Identity and background remain untouched and fully preserved.**
"""

def run_tryon(model_file_path: str, garment_file_path: str):
    client = get_gemini_client()

    # Upload both files to Gemini
    model_file = client.files.upload(file=model_file_path)
    garment_file = client.files.upload(file=garment_file_path)

    # Request to Gemini
    response = client.models.generate_content(
        model="models/gemini-2.5-flash-image",
        contents=[
            types.Part(text=PROMPT),
            model_file,
            garment_file,
        ]
    )

    # Extract image bytes
    image_bytes = None
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if part.inline_data and "image" in part.inline_data.mime_type:
                image_bytes = part.inline_data.data  # already raw bytes
                break
    
    if not image_bytes:
        raise Exception("❌ Gemini returned no image. Full response:\n" + str(response))
    
    return image_bytes
