import streamlit as st
import spacy

nlp = spacy.load("custom_ner_model_spacy")

st.title("📄 Resume Parser App - spaCy NER")
st.write("Paste resume/job text below:")

text = st.text_area("Enter text here")

if st.button("Extract Entities"):
    doc = nlp(text)
    st.subheader("Extracted Entities")
    for ent in doc.ents:
        st.markdown(f"**{ent.label_}**: {ent.text}")