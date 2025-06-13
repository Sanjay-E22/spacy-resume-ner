import spacy
from spacy.training.example import Example
from data.train_data import TRAIN_DATA

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
    print(f"Iteration {i}: Losses {losses}")

nlp.to_disk("custom_ner_model_spacy")
print("✅ Model saved")