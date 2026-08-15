# Iris Flower Classification using Machine Learning

A machine learning project that classifies Iris flower species using three different classification algorithms: K-Nearest Neighbors (KNN), Logistic Regression, and Naive Bayes.

## 📋 Problem Statement

A botanical research centre manually identifies Iris flower species by measuring sepal and petal dimensions. This process is:
- **Time-consuming** — Manual measurements take hours
- **Error-prone** — Human mistakes in identification
- **Not scalable** — Cannot handle large datasets

**Solution:** Build automated ML-based classification models to predict Iris species based on physical measurements.

---

## 📊 Dataset Description

**Source:** Iris Dataset (UCI Machine Learning Repository)  
**Size:** 150 samples  
**Classes:** 3 species (50 samples each)  
**Features:** 4 numerical measurements

| Feature | Description |
|---------|-------------|
| SepalLengthCm | Length of sepal (cm) |
| SepalWidthCm | Width of sepal (cm) |
| PetalLengthCm | Length of petal (cm) |
| PetalWidthCm | Width of petal (cm) |
| **Species (Target)** | Iris-setosa, Iris-versicolor, Iris-virginica |

**Dataset Characteristics:**
- ✅ Clean & well-balanced
- ✅ No missing values
- ✅ Well-separated classes
- ⚠️ Small dataset (150 samples)

---

## 🤖 Models Implemented

### 1. **K-Nearest Neighbors (KNN)**
- Simple distance-based classifier
- Finds k nearest training samples and votes on the class
- Fast prediction, no training phase required

### 2. **Logistic Regression**
- Probabilistic classifier for multi-class problems
- Learns decision boundaries
- Interpretable and efficient

### 3. **Naive Bayes**
- Probabilistic model based on Bayes' theorem
- Assumes feature independence
- Fast training and prediction

---

## 📈 Results

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| KNN | 100% | 100% | 100% |
| Logistic Regression | 98.67% | 98.67% | 98.67% |
| Naive Bayes | 100% | 100% | 100% |

**Best Model:** KNN & Naive Bayes (tied at 100% accuracy)

---

## 🛠️ Installation

### Prerequisites
- Python 3.7+
- pandas
- numpy
- scikit-learn

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/iris-flower-classification.git
cd iris-flower-classification

# Install dependencies
pip install -r requirements.txt
```

### Requirements.txt
```
pandas==1.3.5
numpy==1.21.6
scikit-learn==1.0.2
```

---

## 🚀 Usage

### 1. Load & Prepare Data
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("Iris.csv")

# Encode species to numbers
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Split features and target
X = df.drop(columns='Species')
Y = df['Species']

# Split train-test (50-50)
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.5, random_state=12
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 2. Train Models

#### KNN
```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X_train_scaled, Y_train)
y_pred_knn = knn.predict(X_test_scaled)
```

#### Logistic Regression
```python
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()
lr.fit(X_train_scaled, Y_train)
y_pred_lr = lr.predict(X_test_scaled)
```

#### Naive Bayes
```python
from sklearn.naive_bayes import GaussianNB

nb = GaussianNB()
nb.fit(X_train_scaled, Y_train)
y_pred_nb = nb.predict(X_test_scaled)
```

### 3. Evaluate Models
```python
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Accuracy
print(f"KNN Accuracy: {accuracy_score(Y_test, y_pred_knn)}")
print(f"LR Accuracy: {accuracy_score(Y_test, y_pred_lr)}")
print(f"NB Accuracy: {accuracy_score(Y_test, y_pred_nb)}")

# Confusion Matrix
print(confusion_matrix(Y_test, y_pred_knn))

# Detailed Report
print(classification_report(Y_test, y_pred_knn))
```

---

## 📁 Project Structure

```
iris-flower-classification/
│
├── Iris.csv                    # Dataset
├── iris_classification.ipynb   # Jupyter notebook with full code
├── iris_classification.py      # Python script version
├── README.md                   # This file
├── requirements.txt            # Dependencies
│
└── results/
    ├── model_comparison.png    # Performance comparison chart
    └── confusion_matrices.png  # Confusion matrices visualization
```

---

## 🔑 Key Steps in Data Preparation

1. **Load Data** — Read CSV file into pandas DataFrame
2. **Encode Target** — Convert text labels to numeric (0, 1, 2)
3. **Separate Features & Target** — X = features, Y = target
4. **Train-Test Split** — 50% train, 50% test
5. **Feature Scaling** — StandardScaler for model performance
6. **Train Models** — Fit all three classifiers
7. **Evaluate** — Compare accuracy, precision, recall, F1-score

---

## 📚 What I Learned

✅ Data preprocessing (encoding, scaling, splitting)  
✅ Training multiple classification algorithms  
✅ Model evaluation metrics (accuracy, confusion matrix, classification report)  
✅ Feature engineering importance  
✅ Handling imbalanced data (if present)  

---

## ⚠️ Important Notes

- This dataset is **very clean and balanced**, so models perform exceptionally well (100% accuracy)
- **Real-world datasets** are messier and require more preprocessing
- Always scale features before using distance-based models (KNN)
- Use train-test split to avoid data leakage
- Compare multiple metrics, not just accuracy

---

## 📖 References

- [Scikit-learn Documentation](https://scikit-learn.org)
- [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
- [Supervised Learning Algorithms](https://en.wikipedia.org/wiki/Supervised_learning)

---

## 👤 Author

**Your Name**  
Supervised Machine Learning - Assignment 3  
Contact: your_email@gmail.com

---

## 📄 License

This project is open source and available under the MIT License.

---

## 🤝 Contributing

Feel free to fork, modify, and submit pull requests!

---

**Last Updated:** August 2026  
**Status:** Complete ✅
