# spam-detection-ml-project
Machine Learning spam classifier achieving 98% accuracy using SVM, NLP, and TF-IDF vectorization
# 📧 AI-Powered Spam Detection System

<div align="center">

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Accuracy](https://img.shields.io/badge/accuracy-98.48%25-brightgreen.svg)

</div>

## 🎯 Overview

An intelligent spam detection system that classifies SMS/email messages with **98.48% accuracy** using machine learning. This project compares multiple algorithms and provides comprehensive performance analysis with visualizations.

### ✨ Key Features

- 🤖 **4 ML Algorithms**: Naive Bayes, Logistic Regression, SVM, Random Forest
- 📊 **Rich Visualizations**: Word clouds, confusion matrices, ROC curves
- 🎯 **High Precision**: 100% precision on spam detection
- 📱 **Production Ready**: Clean, documented code

## 📊 Results

| Model | Accuracy |
|-------|----------|
| **SVM** | **98.48%** |
| Logistic Regression | 97.04% |
| Random Forest | 97.31% |
| Naive Bayes | 96.95% |

## 🚀 Quick Start

### Installation
```bash
# Clone repository
git clone https://github.com/HarshithanagashreeR/spam-detection-ml-project.git
cd spam-detection-ml-project

# Install dependencies
pip install -r requirements.txt

# Download dataset
wget -O spam.csv https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv

# Run the detector
python spam_detector.py
```

### Google Colab
```python
!git clone https://colab.research.google.com/drive/1mo4wbDGyh9Ezl9nsKrJ4DUzKZ44uX7fP?usp=sharing
%cd spam-detection-ml-project
!pip install -r requirements.txt
!wget -O spam.csv https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv
!python spam_detector.py
```

## 🛠️ Technologies

- **Python 3.8+**
- **scikit-learn** - Machine learning
- **pandas & numpy** - Data processing
- **matplotlib & seaborn** - Visualization
- **TF-IDF** - Feature extraction
- **NLP** - Text preprocessing

## 📈 Project Workflow

1. **Data Loading**: 5,572 SMS messages
2. **Preprocessing**: Text cleaning with regex and NLP
3. **Feature Extraction**: TF-IDF vectorization with bigrams
4. **Model Training**: 4 algorithms compared
5. **Evaluation**: Comprehensive metrics and visualizations
6. **Prediction**: Interactive function for new messages

## 💡 Key Insights

### Spam Characteristics
- Longer messages with urgency language
- Keywords: "free", "win", "prize", "urgent"
- Excessive punctuation and capitalization

### Model Performance
- **SVM** achieved best results with linear kernel
- **100% precision** = zero false positives
- **89% recall** on spam detection
- TF-IDF with bigrams captured context effectively

## 🎨 Visualizations

## 📸 Visualizations

visualizations/WhatsApp Image 2025-11-30 at 23.50.42.jpeg
visualizations/WhatsApp Image 2025-11-30 at 23.50.43 (1).jpeg
visualizations/WhatsApp Image 2025-11-30 at 23.50.43 (2).jpeg
visualizations/WhatsApp Image 2025-11-30 at 23.50.43 (3).jpeg
visualizations/WhatsApp Image 2025-11-30 at 23.50.43.jpeg
visualizations/WhatsApp Image 2025-11-30 at 23.50.44.jpeg

## 🔮 Sample Predictions
```python
Input: "Congratulations! You've won $1000!"
Output: SPAM (93.27% confidence)

Input: "Meeting at 3pm tomorrow?"
Output: HAM (99.66% confidence)
```

## 🚀 Future Enhancements

- [ ] REST API deployment with Flask
- [ ] Deep learning models (LSTM, BERT)
- [ ] Real-time web interface
- [ ] Multi-language support
- [ ] Model explainability (LIME/SHAP)

## 📄 License

MIT License - see LICENSE file

## 👤 Author

**[Your Name]**
- GitHub: [@HarshithanagashreeR](https://github.com/HarshithanagashreeR)
- LinkedIn: www.linkedin.com/in/harshitha-nagashree-r-1b1277326
- Email: harshithanagashree.r@gmail.com

## 🙏 Acknowledgments

- Dataset: [SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)
- Inspiration: Real-world spam filtering challenges

---

⭐ If you found this helpful, please star the repo!
