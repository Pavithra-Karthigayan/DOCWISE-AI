# 🏥 DOCWISE AI
## A Smart Medical History Analyzer and Doctor Recommendation System

**DOCWISE AI** is an AI-powered healthcare intelligence platform that automates medical PDF report summarization and provides specialist doctor recommendations based on patient symptoms and location.  
The system leverages **Natural Language Processing (NLP)** and **Machine Learning** techniques to support faster clinical decision-making and improve healthcare accessibility.

---

## 🚀 Features

### 👨‍⚕️ Doctor Dashboard
- Upload medical PDF reports
- Automatic medical text extraction
- Transformer-based medical report summarization
- Adjustable summary length
- Downloadable summary output
- Processing time and compression metrics

### 🧑‍🤝‍🧑 Patient Dashboard
- Symptom or disease-based input
- Location-aware doctor filtering
- Specialist prediction
- Doctor ranking based on experience and ratings
- Clean and user-friendly interface

---

## 🧠 Technologies Used
- Python 3.9+
- Streamlit – Web application framework
- Hugging Face Transformers – Medical text summarization
- BART (facebook/bart-large-cnn) – Transformer model
- PyPDF2 – PDF text extraction
- Pandas – Data handling
- Machine Learning – Disease–specialist mapping logic
- Matplotlib – Performance graphs and system diagrams

---
## 🏗️ Project Architecture
```text
DOCWISE_AI/
│
├── app.py
├── requirements.txt
│
├── data/
│   ├── doctor_profiles.csv
│   └── disease_to_doctor.csv
│
├── modules/
│   ├── disease_mapper.py
│   ├── doctor_filtering.py
│   └── __init__.py
│
└── README.md
```

## ⚙️ Installation

### Install Dependencies

pip install -r requirements.txt

▶️ Run the Application
streamlit run app.py

The application will be available at:
http://localhost:8501

📊 Sample Outputs

🔹 Medical Report Summarization

Input PDF size: 301 KB

Original words: 2578

Summary words: 427

Compression ratio: 83.4%

Processing time: ~99 seconds

🔹 Doctor Recommendation

Input disease: Diabetes

Location: Madurai

Recommended specialist: Endocrinologist

Top doctors ranked by experience and ratings

📈 Performance Metrics

High-quality abstractive summarization using transformer models

Accurate specialist mapping based on symptoms

Real-time doctor filtering

Scalable for telemedicine platforms

🧪 Evaluation

Summary compression ratio

Processing time analysis

Specialist prediction accuracy

Doctor recommendation relevance

Evaluation notebooks are available in the notebooks/ directory.

🎬 Demo Video

https://drive.google.com/file/d/1WHBkxeTZMh_nP_64iHMWtXCepJPM6pZI/view?usp=drive_link

🔮 Future Enhancements

OCR support for scanned medical PDFs

Multi-language medical report summarization

Integration with telemedicine platforms
