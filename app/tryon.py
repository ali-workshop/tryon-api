import io
from PIL import Image
from google.genai import types
from app.gemini_client import get_gemini_client

PROMPT = """
Apply the garment from the second image onto the model in the first image. The garment may be a dress, lingerie, or other clothing item.

CRITICAL REQUIREMENTS (STRICT RULES):
- **Identity Preservation (Absolute):** Preserve 100% of the model's face, features, expressions, body pose, and proportions. The model's identity MUST NOT change.
- **Background Preservation (Absolute):** Preserve 100% of the background, lighting, and environment.
- **Garment Integrity (Absolute):** Preserve 100% of the garment's patterns, logos, textures, and colors.
- **Modification Scope (Absolute):** ONLY modify the clothing region. For a **dress**, ensure seamless full-body coverage. For **lingerie**, render sheer and lace fabrics with photorealistic transparency and detail.

PHOTOREALISTIC 8K QUALITY:
- Ultra-detailed fabric textures, including realistic rendering of sheer, lace, silk, and delicate materials.
- Professional lighting and realistic draping/body interaction, conforming naturally to the model's body.
- Sharp focus, zero blur.

NEGATIVE (AVOID AT ALL COSTS): blur, distortions, anime, CGI, wrong face, wrong pose, artifacts, color shift, pattern distortion, unnatural body shape, change in model's identity, background alteration.
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
