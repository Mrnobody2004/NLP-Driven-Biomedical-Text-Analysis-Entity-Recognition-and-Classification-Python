# required libraries
import pandas as pd
import re
import spacy
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
import torch
import nltk

# download NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')


data = pd.read_csv('/content/clinvar_conflicting.csv')

!pip install scispacy
!pip install https://s3-us-west-2.amazonaws.com/scispacy/releases/v0.5.1/en_core_sci_sm-0.5.1.tar.gz

import scispacy
import spacy
!python -m spacy download en_core_sci_sm


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    tokens = word_tokenize(text.lower())
    # Stopword removal
    tokens = [word for word in tokens if word.isalpha() and word not in stop_words]
    # Lemmatization
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return ' '.join(tokens)


data['processed_text'] = data['text'].apply(preprocess_text)


def extract_entities(text):
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]

data['entities'] = data['processed_text'].apply(extract_entities)


def extract_phenotypes(text):
    doc = nlp(text)
    entities = {'disease': [], 'symptom': [], 'medication': []}

    for ent in doc.ents:
        if ent.label_ == 'DISEASE':
            entities['disease'].append(ent.text)
        elif ent.label_ == 'SYMPTOM':
            entities['symptom'].append(ent.text)
        elif ent.label_ == 'MEDICATION':
            entities['medication'].append(ent.text)
    return entities

data['phenotypes'] = data['processed_text'].apply(extract_phenotypes)

def extract_relations(text):
    relations = []
    doc = nlp(text)
    for token in doc:
        if token.dep_ in ["nsubj", "dobj"]:  
            relations.append((token.head.text, token.text))
    return relations

data['relations'] = data['processed_text'].apply(extract_relations)


tfidf = TfidfVectorizer(max_features=1000)
X = tfidf.fit_transform(data['processed_text']).toarray()
y = data['target']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


svm_model = SVC()
svm_model.fit(X_train, y_train)

rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)

model_name = "dmis-lab/biobert-base-cased-v1.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(y.unique()))

train_encodings = tokenizer(list(X_train), truncation=True, padding=True, max_length=128)
test_encodings = tokenizer(list(X_test), truncation=True, padding=True, max_length=128)

class TextDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

train_dataset = TextDataset(train_encodings, y_train.values)
test_dataset = TextDataset(test_encodings, y_test.values)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy="epoch",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

trainer.train()

y_pred_svm = svm_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

accuracy_rf = accuracy_score(y_test, y_pred_rf)
precision_rf = precision_score(y_test, y_pred_rf, average='weighted')
recall_rf = recall_score(y_test, y_pred_rf, average='weighted')
f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
roc_auc_rf = roc_auc_score(y_test, y_pred_rf) if len(y.unique()) == 2 else None

print("Random Forest Model Evaluation:")
print("Accuracy:", accuracy_rf)
print("Precision:", precision_rf)
print("Recall:", recall_rf)
print("F1-score:", f1_rf)
if roc_auc_rf:
    print("ROC-AUC:", roc_auc_rf)

results = trainer.evaluate()
print("BioBERT Model Evaluation Results:", results)