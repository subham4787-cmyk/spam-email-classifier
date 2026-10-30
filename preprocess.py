"""Text cleaning shared by training and prediction."""
import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

try:
    _STOP_WORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    _STOP_WORDS = set(stopwords.words("english"))

_STEMMER = PorterStemmer()


def clean_text(text: str) -> str:
    """Lowercase, strip emails/URLs/punctuation/numbers, drop stopwords, stem."""
    text = text.lower()
    text = re.sub(r"\S+@\S+", " ", text)          # email addresses
    text = re.sub(r"http\S+|www\S+", " ", text)   # URLs
    text = re.sub(r"[^a-z\s]", " ", text)         # punctuation & numbers
    words = [w for w in text.split() if w not in _STOP_WORDS and len(w) > 2]
    return " ".join(_STEMMER.stem(w) for w in words)
