# **Fake News Detector Using LSTM**

## **Overview**

This project utilizes a trained LSTM model (`model_detectFakeNews.pkl`) to classify news articles as **"REAL"** or **"FAKE"** with an impressive **F1 score of 92%**. The model leverages advanced text vectorization techniques, deep learning, and NLP to detect misinformation effectively.

---

## **Model Details**

- **Trained Model**: `model_detectFakeNews.pkl`  
- **Performance**: F1 Score of 92%  
- **Framework**: TensorFlow, Keras

---

## **Dependencies**

Ensure the following libraries are installed before running the model:

```python
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot

from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
from nltk.stem.porter import PorterStemmer
```

To install the required packages, run:

```bash
pip install tensorflow nltk
```

---

## **Project Files**

- **`model_detectFakeNews.pkl`**  
  The trained LSTM model used to classify news articles.

- **`IMPLEMENTING_FAKENEWS_CLASSIFIER.py`**  
  Contains the necessary functions and libraries to run the model. Includes a demo showcasing how to use the model with sample inputs.

- **`Fake News Class LSTM.ipynb`**  
  A Jupyter notebook demonstrating how the model was created. You can also use this file to fine-tune the model for different applications.

---

## **Input Format**

The model requires the input text to be in a **list or nested list format** to process sentences correctly and ensure accurate results.

**Example**:

```python
input_text = ["This news article is an example.", "Another example sentence."]
```

---

## **How to Run the Model**

### **1. Load the Trained Model**

```python
import pickle

# Load the trained LSTM model
model = pickle.load(open("model_detectFakeNews.pkl", "rb"))
```

### **2. Preprocess the Input Text**

```python
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re

# Sample preprocessing function
def preprocess_text(text):
    ps = PorterStemmer()
    stop_words = stopwords.words('english')
    text = re.sub('[^a-zA-Z]', ' ', text).lower().split()
    return ' '.join([ps.stem(word) for word in text if word not in stop_words])

# Example input
input_text = ["This is a sample news article."]
preprocessed_input = [preprocess_text(sentence) for sentence in input_text]
```

### **3. Get Predictions**

```python
prediction = model.predict(preprocessed_input)
print("Prediction:", prediction)
```

### **4. Sample Output**

```
Prediction: ['FAKE']
```

---

## **Tuning the Model**

To fine-tune the model or understand how it was built, refer to the `Fake News Class LSTM.ipynb` file. It contains step-by-step instructions on model training, text preprocessing, and parameter tuning.

---

## **Acknowledgments**

Special thanks to **Dr. T. Sathish Kumar** and the Department of Computer Science and Engineering at the **Hyderabad Institute of Technology and Management** for their guidance and support.
