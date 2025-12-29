import streamlit as st
import torch
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from PIL import Image
import io

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Vision Nexus | Text to Image",
    layout="centered"
)

# --------------------------------------------------
# ADVANCED DARK OCEAN UI
# --------------------------------------------------
st.markdown("""
<style>
body {
    background: radial-gradient(circle at top, #061a2d, #000000);
    color: #e6f1ff;
}

h1 {
    text-align: center;
    font-weight: 800;
    letter-spacing: 1px;
    color: #3fb6ff;
}

.subtitle {
    text-align: center;
    color: #7fbfff;
    margin-bottom: 30px;
}

.cyber-panel {
    background: linear-gradient(145deg, rgba(6,26,45,0.95), rgba(2,8,15,0.95));
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 24px;
    box-shadow:
        0 20px 40px rgba(0,0,0,0.7),
        inset 0 0 0 1px rgba(63,182,255,0.25);
}

input, textarea {
    background-color: #020c18 !important;
    color: #e6f1ff !important;
    border: 1px solid #1f6cff !important;
}

.stButton button {
    background: linear-gradient(135deg, #1f6cff, #00c6ff);
    color: black;
    font-weight: 700;
    border-radius: 12px;
    padding: 0.6em 1.6em;
}

.stButton button:hover {
    transform: scale(1.03);
}

.status {
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    background: rgba(63,182,255,0.15);
    border: 1px solid rgba(63,182,255,0.4);
    color: #7fbfff;
    font-size: 13px;
    font-weight: 600;
    margin-right: 10px;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<h1>Vision Nexus</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Deep Ocean AI • High-Fidelity Text-to-Image Engine</div>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD MODEL (CACHED & FAST)
# --------------------------------------------------
@st.cache_resource(show_spinner=True)
def load_pipeline():
    scheduler = EulerDiscreteScheduler.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        subfolder="scheduler"
    )

    pipe = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        scheduler=scheduler,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = pipe.to(device)
    pipe.safety_checker = None
    return pipe, device

pipe, device = load_pipeline()

# --------------------------------------------------
# STATUS HUD
# --------------------------------------------------
st.markdown(f"""
<div class="status">Model: SD 1.5</div>
<div class="status">Scheduler: Euler</div>
<div class="status">Device: {"GPU" if device=="cuda" else "CPU"}</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# PROMPT PANEL
# --------------------------------------------------
st.markdown("<div class='cyber-panel'>", unsafe_allow_html=True)

example_prompt = st.selectbox(
    "Example Prompts",
    [
        "A futuristic cyberpunk city illuminated by deep blue neon lights",
        "Ultra-realistic robot portrait with glowing blue eyes",
        "A deep-sea research base with bioluminescent architecture",
        "A cinematic space station orbiting Earth at night",
        "A high-tech AI control room with holographic displays"
    ]
)

prompt = st.text_input(
    "Text Prompt",
    value=example_prompt
)

negative_prompt = st.text_input(
    "Negative Prompt",
    "blurry, low quality, distorted, watermark, extra limbs"
)

steps = st.slider("Inference Steps", 15, 30, 20)
cfg = st.slider("Guidance Scale (CFG)", 5.0, 10.0, 7.5)

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# GENERATE
# --------------------------------------------------
if st.button("Generate Image"):
    with st.spinner("Synthesizing image..."):
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                num_inference_steps=steps,
                guidance_scale=cfg
            )

        image = result.images[0]

    # --------------------------------------------------
    # OUTPUT PANEL
    # --------------------------------------------------
    st.markdown("<div class='cyber-panel'>", unsafe_allow_html=True)
    st.image(image, caption="Generated Image", width=512)
    st.markdown("</div>", unsafe_allow_html=True)

    # --------------------------------------------------
    # DOWNLOAD
    # --------------------------------------------------
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    st.download_button(
        "Download Image",
        data=buf.getvalue(),
        file_name="vision_nexus_output.png",
        mime="image/png"
    )

    # --------------------------------------------------
    # METADATA
    # --------------------------------------------------
    st.markdown("<div class='cyber-panel'>", unsafe_allow_html=True)
    st.markdown("### Generation Metadata")
    st.code(f"""
Prompt: {prompt}
Negative Prompt: {negative_prompt}
Steps: {steps}
CFG Scale: {cfg}
Device: {device.upper()}
Model: Stable Diffusion 1.5
Scheduler: Euler
""")
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    "<div style='text-align:center;color:#7fbfff;opacity:0.7;'>"
    "Vision Nexus • Local Stable Diffusion • Internship-Ready"
    "</div>",
    unsafe_allow_html=True
)
