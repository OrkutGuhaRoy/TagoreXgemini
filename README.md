
# 🕊️ TagoreXGemini – A Bengali Text Generator Inspired by Rabindranath Tagore

> "AI doesn’t have to be massive. It can be local, soulful, and deeply human."

🎯 **Live Demo → [tagorexgemini.streamlit.app](https://tagorexgemini.streamlit.app)**

---

## 📘 What is TagoreX?

**TagoreX** is a Bengali poetic text generation model trained on the works of **Rabindranath Tagore** using AddaGPT 2.0. While the model produces fragments of raw poetic thought, the **true magic happens when it's refined using Gemini 2.5 Flash** — which extracts, clarifies, and polishes the deeper poetic or philosophical undertone.

---

## 🌸 Features

* ✍️ Input a Bengali phrase and generate poetic/philosophical continuations
* 🧠 Model: [`SwastikGuhaRoy/TagoreX`](https://huggingface.co/SwastikGuhaRoy/TagoreX) 
* ✨ Gemini 2.5 Flash automatically interprets and refines output to enhance readability
* 📜 Simple, minimal UI built with Streamlit

---

## 🧠 Why Gemini is Needed

TagoreX on its own produces raw outputs that often feel **fragmented or nonsensical** at surface level.

But when these are passed through **Gemini 2.5 Flash**, it:

* Detects hidden poetic/philosophical layers
* Refines and polishes the text
* Makes the output **readable, expressive, and meaningful**
* Offers interpretations grounded in **Indic context and poetic structure**

Without Gemini, the app’s output would often seem incoherent — **Gemini is not optional; it’s essential** to this experience.

---

## 🔧 Technical Summary

| Component       | Details                                    |
| --------------- | ------------------------------------------ |
| Base Model      | GPT-2 (117M parameters)                    |
| Training Style | Full-tuned on Tagore's works (AddaGPT2.0 variant) |
| Refinement      | Gemini 2.5 Flash API                       |
| Max Tokens      | 256                                        |
| Dataset         | [Imperfect corpus of Tagore’s works](https://huggingface.co/datasets/SwastikGuhaRoy/WorksofTagore)       |
| Training Epochs | 22 (symbolic of ২২শে শ্রাবণ)               |
| Framework       | PyTorch + HuggingFace Transformers         |

---

## 🚀 How to Run Locally

```bash
git clone https://github.com/OrkutGuhaRoy/tagorexgemini.git
cd tagorex-streamlit

# Install dependencies
pip install -r requirements.txt

# Add your Gemini key
mkdir .streamlit
echo '[gemini_key]\ngemini_key = "YOUR_GEMINI_API_KEY"' > .streamlit/secrets.toml

# Run the app
streamlit run app.py
```

---

## 🧪 Sample Prompt (Raw Model Use)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("SwastikGuhaRoy/TagoreX")
model = AutoModelForCausalLM.from_pretrained("SwastikGuhaRoy/TagoreX")

prompt = "তুমি রবে নীরবে"
inputs = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=256, do_sample=True, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

👉 **Note:** This raw output will likely be stylistic but unstructured. For meaningful poetic coherence, use the **Streamlit app with Gemini refinement**.

---

## 🎨 Intended Use

* 📝 Bengali poetic experimentation
* 🎭 Creative prompt generation
* 🌱 Exploring Indic LLMs in low-resource settings

Not suitable for:

* ❌ Scholarly or critical literary work
* ❌ Commercial or factual generation
* ❌ High-stakes deployment

---

## ⚠️ Limitations

* Raw model outputs can be fragmented
* Relies on Gemini for interpretive polishing
* Dataset is not academically curated
* Output reflects training biases and randomness

## 🚫 Disclaimer
* The generated content does not reflect the thoughts, views, or philosophy of Rabindranath Tagore or any individual, living or deceased.

* It is a machine-generated approximation inspired by the style and tone of Tagore’s literary works.

* The outputs are creative imitations, not authentic writings.

* This project is intended for artistic and educational exploration only — not for scholarly or critical representation of Tagore’s legacy.

---

## 📫 Contact

Have ideas, thoughts, or beautiful generations to share?

📧 Email: [swastikguharoy@googlemail.com](mailto:swastikguharoy@googlemail.com)

---

Would you like this automatically exported as a `README.md` file or committed to a GitHub repo with license and `requirements.txt` as well?
