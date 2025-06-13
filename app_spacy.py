import spacy
import streamlit as st
from spacy.training.example import Example
import os
import json
import random

st.set_page_config(page_title="📄 Resume NER App", layout="wide")
st.title("📄 Resume NER with spaCy (Auto-Trained)")
st.write("Paste resume text below to extract entities like name, job title, skills, etc.")

MODEL_DIR = "custom_ner_model_spacy"
TRAIN_FILE = "spacy_resume_train_data_50.json"

# =====================
# 🔁 Auto-train model
# =====================
if not os.path.exists(MODEL_DIR):
    st.info("🔧 Training spaCy model for the first time...")

    # Load training data
    with open(TRAIN_FILE) as f:
        TRAIN_DATA = json.load(f)

    # Create blank English NLP model
    nlp = spacy.blank("en")

    # Add NER pipe
    if "ner" not in nlp.pipe_names:
        ner = nlp.add_pipe("ner")
    else:
        ner = nlp.get_pipe("ner")

    # Add entity labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])

    # Train
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.begin_training()
        for i in range(30):
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                doc = nlp.make_doc(text)
                example = Example.from_dict(doc, annotations)
                nlp.update([example], drop=0.3, losses=losses)
            print(f"Iteration {i+1}, Losses: {losses}")

    # Save model
    nlp.to_disk(MODEL_DIR)
    st.success(f"✅ Model trained and saved at: {MODEL_DIR}")

# =====================
# 🚀 Load model
# =====================
nlp = spacy.load(MODEL_DIR)

# =====================
# 📥 Text input
# =====================
text_input = st.text_area("Paste resume text here 👇", height=250)

# =====================
# 🎯 Entity Extraction
# =====================
if st.button("🔍 Extract Entities"):
    if text_input:
        doc = nlp(text_input)
        st.subheader("🔎 Recognized Entities")

        # Custom colors for each label
        colors = {
            "NAME": "#FFD700",         # gold
            "JOB_TITLE": "#00BFFF",    # deep sky blue
            "ORG": "#32CD32",          # lime green
            "EMAIL": "#FF69B4",        # pink
            "PHONE": "#FFA07A",        # light salmon
            "SKILL": "#00CED1",        # dark turquoise
            "EDUCATION": "#BA55D3",    # medium orchid
        }
        options = {"ents": list(colors.keys()), "colors": colors}
        html = spacy.displacy.render(doc, style="ent", options=options, page=True)
        st.write(html, unsafe_allow_html=True)
    else:
        st.warning("⚠️ Please paste some resume text first.")
