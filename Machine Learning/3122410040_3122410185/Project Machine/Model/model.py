import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import pickle
mydata=pd.read_csv("text.csv")
print(mydata.info())
from sklearn.model_selection import train_test_split

# Chia dữ liệu thành X (văn bản) và y (nhãn)
X = mydata['text']
y = mydata['label']

# Chia thành tập huấn luyện và tập kiểm tra (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer

# Chuyển đổi dữ liệu văn bản thành ma trận TF-IDF
vectorizer = TfidfVectorizer(max_features=2000)  # Giới hạn số từ quan trọng
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)  # Chỉ cần transform cho tập kiểm tra
print('hoàn thành')

# Lưu vectorizer vào file vectorizer.pkl
with open("../Api/vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("Vectorizer đã được lưu vectorizer.pkl")

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

model_lr = LogisticRegression(max_iter=10000)
model_lr.fit(X_train_tfidf, y_train)

# Dự đoán và tính độ chính xác
y_pred_lr = model_lr.predict(X_test_tfidf)
accuracy_lr = accuracy_score(y_test, y_pred_lr)
print(f"Logistic Regression Accuracy: {accuracy_lr}")



model_pkl_file = "../Api/emotion_md1.pkl"

with open(model_pkl_file, 'wb') as file:

        pickle.dump(model_lr, file)
