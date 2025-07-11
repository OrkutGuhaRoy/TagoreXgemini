import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from google import genai

# Show title and description.
st.title("📜 TagoreX + Gemini")

with st.expander("ℹ️ Please READ"):
    st.markdown("""
    Welcome to **TagoreX + Gemini**, a prompt generation and refinement project. This project is a humble attempt to bring a fragment of Rabindranath Tagore’s poetic world into the realm of AI—using small foundation models and free compute resources. While it does not replicate the poet's genius, it seeks to echo the rhythm and thoughtfulness of his work within the creative constraints of modern LLMs.

    ---

    ### ❗ Purpose & Disclaimer:

    1. **Educational Project Only**  
    This tool was created purely for learning and experimentation. It is **not** intended to copy, impersonate, or reproduce the works or thoughts of Rabindranath Tagore in any form.

    2. **Training on Tagore’s Works**  
    The TagoreX model was fine-tuned on a dataset based on the literary works of Tagore. However, the dataset was **not perfectly cleaned**—it may contain **spelling errors, formatting issues, and incomplete texts**. It does not claim to be scholarly or exhaustive.

    3. **Model Details**  
    - The base model is **AddaGPT2.0**, a LoRA-adapted GPT-2 model trained on a Bengali NER dataset ([available here](https://huggingface.co/SwastikGuhaRoy/Addagpt2.0)).  
    - It was limited in its ability to form meaningful Bengali sentences.  
    - I fine-tuned it for **22 epochs** using the Tagore dataset (as a symbolic tribute to Tagore and “22 শ্রাবণ”) to create **TagoreX**, which you can explore on [Hugging Face](https://huggingface.co/SwastikGuhaRoy/TagoreX).

    4. **How It Works**  
    - You provide a **Bengali seed prompt**.  
    - The model appends ~256 tokens to it using TagoreX.  
    - The raw text may not always be coherent due to the limited training data and GPT-2’s minimal exposure to Bengali.  
    - Then, using **Gemini 2.5-flash**, the output is refined grammatically and stylistically, followed by a **brief English interpretation**.

    5. **Important Note**  
    This app is meant as a **fun and educational experiment**. Please do **not** interpret any of the generated or refined outputs as genuine works or words of Rabindranath Tagore. It represents **no one’s thoughts**—not his, not mine.

    6. **Contact**  
    Got questions or cool outputs you’d like to share?  
    📧 Reach out: **swastikguharoy@googlemail.com**
    
    7. **Working Best at .7 Temperature**

    ---

    Thank you for using this project with curiosity and respect 🙏
    """)

# Ask user for their OpenAI API key via `st.text_input`.
# Alternatively, you can store the API key in `./.streamlit/secrets.toml` and access it
# via `st.secrets`, see https://docs.streamlit.io/develop/concepts/connections/secrets-management
@st.cache_resource
def load_tagorex_model():
    tagorex_repo = "SwastikGuhaRoy/TagoreX"  # Public model on Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(tagorex_repo)
    model = AutoModelForCausalLM.from_pretrained(tagorex_repo)
    return tokenizer, model

# --- Generate with TagoreX ---
def tagorex_generate(prompt, tokenizer, model, temperature):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=temperature, do_sample=True)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# --- Gemini Setup ---
def setup_gemini():
    gemini_api_key = st.secrets["gemini_key"]
    
client=genai.Client(api_key=st.secrets["gemini_key"])
    
# --- Refine Output ---
def refine_with_gemini(model, text):
    prompt = f"""
    You are a refined literary assistant. Your task is to polish Bengali poetic or philosophical writing.

    Instructions:
    - Preserve the style, tone, and emotional depth of the original.
    - Keep the original wording and structure as intact as possible while improving grammar and clarity.
    - Then provide a brief English interpretation (along with a translation), capturing the *essence and emotional intent* of the original.
    - Do not add any commentary or analysis.

    Format your response exactly like this:

    🔹 Bengali (Refined):
    <Your improved Bengali version here>

    🔸 English Interpretation:
    <Your concise interpretation and translation here>

    Text to refine:
    {text}
    """


    response = client.models.generate_content(
    model="gemini-2.5-flash", contents= prompt)
    return response.text.strip()

# --- Streamlit UI ---
st.set_page_config(page_title="TagoreX + Gemini", layout="centered")
st.title("📜 TagoreX + Gemini")

user_prompt = st.text_area("🔹 Enter a Bengali Prompt", "তুমি রবে নীরবে হৃদয়ে মম", height=100)
temperature = st.slider("🔥 TagoreX Temperature",  min_value=0.0, max_value=1.0, value=0.7, step=0.1)

if st.button("🚀 Generate"):
    tokenizer, model = load_tagorex_model()
    with st.spinner("Generating from TagoreX..."):
        tagore_output = tagorex_generate(user_prompt, tokenizer, model, temperature)
    st.markdown("### 📜 TagoreX Output")
    st.success(tagore_output)

    try:
        gemini = setup_gemini()
        with st.spinner("Refining with Gemini..."):
            refined = refine_with_gemini(gemini, tagore_output)
        st.markdown("### 🎨 Gemini Refined Output")
        st.info(refined)
    except Exception as e:
        st.error(f"Gemini API Error: {e}")
