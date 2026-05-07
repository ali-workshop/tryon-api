import io
from PIL import Image
from google.genai import types
from app.gemini_client import get_gemini_client
client = get_gemini_client()
PROMPT = """
# **Half-Body Garment Replacement Prompt (FORCED CLOTHING REMOVAL + NO LAYERING)**

**TASK:**
Apply the garment from Image 2 onto the model in Image 1.
This prompt is ONLY for **half-body clothing replacement** (upper-body OR lower-body).
The original clothing in the target region MUST be COMPLETELY REMOVED before applying the new garment.

---

## **CRITICAL REQUIREMENTS (STRICT RULES)**

### **1. Identity Preservation — ABSOLUTE**
- Preserve 100% of the model’s face, expression, hair, skin tone, body shape, and pose.
- NO beautification, morphing, smoothing, or alteration.

### **2. Background Preservation — ABSOLUTE**
- Keep the original background, lighting, shadows, and environment fully unchanged.

---

## **3. FORCED ORIGINAL CLOTHING REMOVAL — ABSOLUTE (NO LAYERING)**
Before placing the new garment:
- **Erase the original garment entirely from the target half-body region.**
- The original clothing MUST NOT be visible in any form:
  - NO collars
  - NO sleeves
  - NO hems
  - NO waistbands
  - NO fabric edges
  - NO textures showing through
- The new garment MUST be the **ONLY visible clothing** in the replaced region.

**Layering, overlapping, or placing the new garment on top of the old one is STRICTLY FORBIDDEN.**

---

## **4. Garment Integrity — ZERO MODIFICATION (DO NOT CHANGE THE GARMENT)**
The garment from Image 2 MUST appear EXACTLY as provided:
- NO recoloring
- NO retouching
- NO cleanup
- NO smoothing
- NO resizing or reshaping
- NO pattern correction
- NO logo distortion
- NO fabric reinterpretation

Preserve ALL:
- Colors, tones, saturation
- Fabric texture and weave
- Wrinkles, folds, creases
- Prints, graphics, logos
- Watermarks, stamps, labels
- Lighting and shadows visible on the garment itself
- Imperfections, distortions, photo artifacts

**Transfer the garment AS-IS. No enhancements. No improvements. No corrections.**

---

## **5. Half-Body Replacement Scope — ABSOLUTE**
- Modify ONLY the target half-body area (upper or lower).
- Do NOT affect non-target clothing.
- Do NOT extend the garment beyond its natural boundaries.
- The garment must conform naturally to the model’s pose WITHOUT altering garment details.

---

## **PHOTOREALISM (ULTRA-REALISTIC)**
- Sharp, high-resolution output.
- Natural fabric tension and drape.
- Correct occlusion where the garment meets skin or other clothing.
- Clean, realistic garment–skin edges.

---

## **NEGATIVE INSTRUCTIONS — STRICTLY FORBIDDEN**
- Any part of the original clothing visible
- Garment layering or stacking
- Ghost outlines of the original garment
- Transparency revealing original clothes
- Fabric bleed-through
- Altering the garment in any way
- Background or lighting changes
- CGI, plastic, or painted textures
- Body warping or incorrect anatomy

---

## **FINAL ENFORCEMENT CHECK (MANDATORY)**
Before generating the final output, CONFIRM:

1. **The original clothing in the target region has been COMPLETELY ERASED.**
2. **Only ONE garment exists in the replaced area.**
3. **No collars, sleeves, hems, or fabric edges from the original garment are visible.**
4. **The new garment matches Image 2 EXACTLY with ZERO modification.**
5. **Identity, pose, and background remain untouched.**



"""

def run_tryon(model_file_path: str, garment_file_path: str):
    

    # Upload both files to Gemini
    model_file = client.files.upload(file=model_file_path)
    garment_file = client.files.upload(file=garment_file_path)

    # Request to Gemini
    response = client.models.generate_content(
        model="models/gemini-2.5-flash-image",
        contents=[
           
            model_file,
            garment_file,
             types.Part(text=PROMPT),
        ]
    )

    # Extract image bytes
    image_bytes = None
    for candidate in response.candidates:
        for part in candidate.content.parts:
            if part.inline_data and "image" in part.inline_data.mime_type:
                image_bytes = part.inline_data.data  # already raw bytes
                break
        if image_bytes:
            break
    
    if not image_bytes:
        raise Exception("❌ Gemini returned no image. Full response:\n" + str(response))
    
    return image_bytes
