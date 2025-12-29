import streamlit as st
import os
import httpx
import asyncio
import time
from urllib.parse import quote
import io

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="Vision Nexus | Instant AI Art",
    layout="centered"
)

# --- API Configuration ---
# Fetching Gemini Key for Prompt Enhancement (Optional/Free Tier)
if "API_KEY" not in st.session_state:
    try:
        st.session_state.API_KEY = st.secrets.get("VITE_GEMINI_API_KEY") or os.environ.get("VITE_GEMINI_API_KEY")
    except Exception:
        st.session_state.API_KEY = os.environ.get("VITE_GEMINI_API_KEY", "")

# --------------------------------------------------
# ADVANCED DARK OCEAN UI (CSS Grid & Modern Styling)
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
    width: 100%;
    padding: 0.8em;
    transition: 0.3s;
}

.stButton button:hover {
    transform: translateY(-2px);
    box-shadow: 0 10px 20px rgba(31, 108, 255, 0.4);
}

.status-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
    gap: 10px;
    margin-bottom: 20px;
}

.status-item {
    padding: 8px 14px;
    border-radius: 999px;
    background: rgba(63,182,255,0.1);
    border: 1px solid rgba(63,182,255,0.3);
    color: #7fbfff;
    font-size: 12px;
    text-align: center;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown("<h1>Vision Nexus</h1>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Instant Cloud AI • High-Speed Synthesis Engine</div>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# HELPER FUNCTIONS (API DRIVEN)
# --------------------------------------------------
async def enhance_prompt(client, user_input: str):
    """Uses Gemini to expand the prompt for cinematic results."""
    api_key = st.session_state.get("API_KEY")
    if not api_key: return user_input # Fallback to original if no key
        
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={api_key}"
    payload = {
        "contents": [{"parts": [{"text": f"Act as an AI artist. Expand this prompt for a cinematic image: {user_input}"}]}],
        "systemInstruction": {"parts": [{"text": "Output only the expanded prompt text."}]}
    }
    try:
        response = await client.post(url, json=payload, timeout=10.0)
        return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()
    except:
        return user_input

async def generate_cloud_image(client, prompt: str, model="turbo"):
    """Fetches image from Pollinations.ai (Free & Instant)."""
    encoded = quote(prompt)
    seed = int(time.time())
    # Turbo model is optimized for sub-second responses
    url = f"https://image.pollinations.ai/prompt/{encoded}?width=1024&height=1024&seed={seed}&nologo=true&model={model}"
    try:
        response = await client.get(url, timeout=30.0, follow_redirects=True)
        if response.status_code == 200:
            return response.content
        return None
    except:
        return None

# --------------------------------------------------
# STATUS HUD (CSS GRID)
# --------------------------------------------------
st.markdown(f"""
<div class="status-grid">
    <div class="status-item">Engine: Pollinations</div>
    <div class="status-item">Model: Turbo-Instant</div>
    <div class="status-item">Mode: Cloud Hybrid</div>
</div>
""", unsafe_allow_html=True)

# --------------------------------------------------
# PROMPT PANEL
# --------------------------------------------------
st.markdown("<div class='cyber-panel'>", unsafe_allow_html=True)

user_input = st.text_area("What should the AI visualize?", placeholder="e.g. A cyberpunk library in a deep blue ocean...")
auto_enhance = st.checkbox("AI Prompt Enhancement (Requires Gemini Key)", value=True)
model_choice = st.selectbox("Vision Engine", ["turbo", "flux"])

if st.button("Generate Artwork"):
    if not user_input:
        st.warning("Please enter a prompt first.")
    else:
        async def process():
            async with httpx.AsyncClient() as client:
                start_time = time.time()
                
                # Step 1: Enhance (Gemini is fast)
                final_prompt = user_input
                if auto_enhance and st.session_state.API_KEY:
                    with st.spinner("🧠 Brainstorming details..."):
                        final_prompt = await enhance_prompt(client, user_input)
                
                # Step 2: Generate (Cloud API is instant)
                with st.spinner("⚡ Synthesizing pixels..."):
                    img_bytes = await generate_cloud_image(client, final_prompt, model_choice)
                
                if img_bytes:
                    elapsed = round(time.time() - start_time, 2)
                    st.session_state.last_image = img_bytes
                    st.session_state.last_elapsed = elapsed
                    st.session_state.last_prompt = final_prompt
                else:
                    st.error("Engine busy. Please try again.")

        asyncio.run(process())

st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# OUTPUT PANEL
# --------------------------------------------------
if "last_image" in st.session_state:
    st.markdown("<div class='cyber-panel'>", unsafe_allow_html=True)
    st.image(st.session_state.last_image, caption=f"Generated in {st.session_state.last_elapsed}s", use_container_width=True)
    
    st.download_button(
        "Download Masterpiece",
        data=st.session_state.last_image,
        file_name="vision_nexus_instant.png",
        mime="image/png"
    )
    
    with st.expander("View Engineering Metadata"):
        st.code(st.session_state.last_prompt)
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    "<div style='text-align:center;color:#7fbfff;opacity:0.7;'>"
    "Vision Nexus • Instant Cloud Hybrid • Prodigy WD-02"
    "</div>",
    unsafe_allow_html=True
)