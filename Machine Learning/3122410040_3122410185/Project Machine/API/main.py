from fastapi import FastAPI
from pydantic import BaseModel
import pickle
from fastapi.middleware.cors import CORSMiddleware

# Tải vectorizer và mô hình đã lưu
with open("../Model/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("../Model/emotion_md_LR.pkl", "rb") as f:
    model = pickle.load(f)

# Tạo từ điển ánh xạ nhãn cảm xúc
index_to_emotion = {
    0: 'Sadness',
    1: 'Joy',
    2: 'Love',
    3: 'Anger',
    4: 'Fear',
    5: 'Surprise'
}

# Tạo ứng dụng FastAPI
app = FastAPI()

# Cấu hình CORS để cho phép các yêu cầu từ trình duyệt
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả nguồn gốc (có thể giới hạn theo domain nếu cần)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.post("/predict")
async def predict_emotion(request: TextRequest):
    text = request.text.strip()

    # Kiểm tra đầu vào hợp lệ
    if len(text) < 3:
        return {
            "sentence": text,
            "emotion": "Invalid Input",
            "confidence": 0
        }

    # Chuyển văn bản thành ma trận vector TF-IDF
    transformed_text = vectorizer.transform([text])

    # Lấy xác suất dự đoán
    probabilities = model.predict_proba(transformed_text)[0]
    max_probability = max(probabilities)
    prediction = model.predict(transformed_text)[0]  # Dự đoán nhãn cảm xúc
    emotion = index_to_emotion[int(prediction)]

    # Kiểm tra ngưỡng tin cậy


    return {
        "sentence": text,
        "emotion": emotion,
        "confidence": max_probability
    }
