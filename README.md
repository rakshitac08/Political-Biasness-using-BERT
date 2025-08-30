# Political-Biasness-using-BERT
# 📌 Overview
This project aims to detect political bias in Indian news articles using a fine-tuned **BERT (Bidirectional Encoder Representations from Transformers)** model. The system classifies articles as:

* 🟠 **BJP**
* 🟣 **AAP**
* 🔵 **Congress**
* ⚪ **Neutral**

The model is deployed through a user-friendly **Streamlit interface** for real-time predictions.

---
### 👥 Contributors

* **Anisha Kumari**
* **Rakshita**
* **Nitin**
* **Mentor**: Prof. Gaurav Singhal

---

### 🧠 Motivation

In an age of information overload, unbiased news is vital. This project explores how deep learning and natural language processing can help detect and quantify political bias in textual content, promoting responsible media consumption.

---

### 🗂️ Dataset

* **Source**: Collected from diverse Indian news sources
* **Format**: CSV with `text` and `label` columns
* **Labels**: `BJP`, `AAP`, `Congress`, `None`

#### Example Distribution:

```
BJP       : 287 articles
Congress  : 159 articles
AAP       : 37 articles
None      : 489 articles
```

---

### ⚙️ Technologies Used

| Component     | Description                                 |
| ------------- | ------------------------------------------- |
| Model         | BERT (`bert-base-uncased`)                  |
| Frameworks    | PyTorch, Hugging Face Transformers          |
| Interface     | Streamlit                                   |
| Evaluation    | Scikit-learn (`classification_report`, AUC) |
| Visualization | Matplotlib, Seaborn                         |

---

### 🧪 Methodology

1. **Preprocessing**: Text cleaning, label normalization, tokenization
2. **Modeling**: Fine-tuned BERT for 4-class classification
3. **Training**:

   * Loss Function: CrossEntropyLoss
   * Optimizer: AdamW
   * Epochs: 7
4. **Evaluation**: Confusion matrix, macro/micro/weighted scores, ROC curves
5. **Deployment**: Streamlit app for live predictions

---

### 📊 Performance Metrics (BERT Model)

| Metric            | Value |
| ----------------- | ----- |
| Accuracy          | 91.7% |
| Precision (macro) | 92.6% |
| Recall (macro)    | 91.7% |
| F1-score (macro)  | 91.9% |

#### 🔹 Baseline (TF-IDF + Logistic Regression)

* Accuracy: 79.1%

This highlights the advantage of BERT’s deep contextual understanding over traditional feature-based methods.

---

### 📊 Visualizations

* Confusion Matrix
* ROC Curves
* Word Clouds
* Label Distribution
* Text Length Histogram

---

### 🚀 Running the Project

```bash
# Clone the repository
git clone https://github.com/your-username/political-bias-detection.git
cd political-bias-detection

# Install dependencies
pip install -r requirements.txt

# Launch the app
streamlit run app.py
```

---

### 📒 File Structure

```
├── Model.ipynb                    # Model training
├── stream_app.py                         # Streamlit app
├── Final_Data_Article_Annotation.csv  # Dataset
├── model12/                        # Trained model files
├── Data Visualizaton.ipnyb          # Dataset Analysis
└── README.md                      # Documentation
```

---

### 💡 Future Work

* Improve class balance and add more neutral data
* Test on multilingual datasets
* Integrate interpretability tools like SHAP/LIME

---

