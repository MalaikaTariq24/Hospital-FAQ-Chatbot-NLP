# 🏥 Hospital FAQ Chatbot & Urgency Detector

An intelligent, dual-task Natural Language Processing (NLP) pipeline that classifies patient inquiries into hospital departments while dynamically detecting urgency levels.

## 📌 System Architecture

- **Intent Classification Model:** Multi-class classification across 9 operational categories (*Billing, Emergency, Parking, Appointments, etc.*).
- **Urgency Detection Model:** Binary classification (*Routine vs. Emergency*).
- **Feature Extraction:** TF-IDF Vectorization.
- **Algorithm:** Random Forest Classifier.

## 📊 Performance & Evaluation Metrics

| Model | Metric | Validation Accuracy | Live Test Accuracy |
| :--- | :--- | :--- | :--- |
| **Category Model** | Multi-class Accuracy | **100.00%** | **100.00%** |
| **Urgency Model** | Binary Accuracy | **100.00%** | **87.50%** |

## 🛠️ Pipeline Steps
1. **Text Normalization & Regex Cleaning**
2. **Word Tokenization (NLTK)**
3. **Stop Words Filtering & Lemmatization**
4. **POS Tagging & Named Entity Recognition (NER)**
5. **TF-IDF Feature Matrix Generation**
6. **Random Forest Training & Evaluation**
7. **Real-time Live Inference**
