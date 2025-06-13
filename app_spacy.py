import streamlit as st
import spacy
import os
from spacy.training.example import Example

TRAIN_DATA = [
    ("John Doe is a Senior Data Scientist at TCS skilled in Python and Machine Learning.",
     {"entities": [(0, 8, "NAME"), (15, 35, "JOB_TITLE"), (39, 42, "ORG"), (60, 66, "SKILL"), (71, 89, "SKILL")]}),
    ("Jane Smith graduated from IIT Delhi and works as a Backend Developer at Amazon.",
     {"entities": [(0, 10, "NAME"), (25, 34, "EDUCATION"), (51, 68, "JOB_TITLE"), (72, 78, "ORG")]}),
    ("Contact: rahul@gmail.com, Phone: 9876543210",
     {"entities": [(9, 26, "EMAIL"), (35, 45, "PHONE")]}),
]

MODEL_PATH = "custom_ner_model_spacy"

@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        return spacy.load(MODEL_PATH)

    nlp = spacy.blank("en")
    ner = nlp.add_pipe("ner")

    for _, annotations in TRAIN_DATA:
        for ent in annotations["entities"]:
            ner.add_label(ent[2])

    nlp.begin_training()
    for i in range(20):
        losses = {}
        for text, annotations in TRAIN_DATA:
            example = Example.from_dict(nlp.make_doc(text), annotations)
            nlp.update([example], losses=losses)
    nlp.to_disk(MODEL_PATH)
    return nlp

st.title("ðŸ“„ Resume Parser App - spaCy NER")
st.write("Paste resume/job text below:")

nlp = load_or_train_model()

text = st.text_area("Enter text here")

if st.button("Extract Entities"):
    doc = nlp(text)
    st.subheader("Extracted Entities")
    for ent in doc.ents:
        st.markdown(f"**{ent.label_}**: {ent.text}")
