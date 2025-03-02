# NLP-Driven Biomedical Text Analysis

## Description
This project leverages Natural Language Processing (NLP) to analyze biomedical text. It includes text preprocessing, Named Entity Recognition (NER) for extracting diseases, symptoms, and medications, and relationship extraction. Machine learning models (SVM, Random Forest) and a deep learning model (BioBERT) are used for text classification.

## Features
- Text preprocessing (tokenization, stopword removal, lemmatization)
- Named Entity Recognition (NER) with SciSpacy
- Relationship extraction using dependency parsing
- Classification with SVM, Random Forest, and BioBERT
- Model evaluation using accuracy, precision, recall, and F1-score

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Mrnobody2004/nlp-biomedical-analysis.git
   cd nlp-biomedical-analysis
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
Run the Jupyter Notebook:
```bash
jupyter notebook nlpproject.ipynb
```

## Dependencies
- Python 3.x
- pandas
- NLTK
- SciSpacy
- Scikit-learn
- Transformers (Hugging Face)
- PyTorch

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author
[Mrnobody2004](https://github.com/Mrnobody2004)

