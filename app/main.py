from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import torch
import torch.nn as nn
from torchvision import transforms
from typing import List, Annotated
from pydantic import WithJsonSchema
import base64
from datetime import datetime
import os

UploadFileX = Annotated[UploadFile, WithJsonSchema({"type": "string", "format": "binary"})]
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

from app.model import DEVICE, load_model

app = FastAPI()

model = load_model()

# classes = ["healthy", "ulcer"]

@app.get("/")
def root():
    return {"message": "DFU Classification API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()

    image = Image.open(io.BytesIO(contents)).convert("RGB")
    # image = transforms(image).unsqueeze(0).to(DEVICE)
    image = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(image)
        prob = output.item()

    prediction = "ulcer" if prob > 0.5 else "normal"
    confidence = prob if prob > 0.5 else 1 - prob

    return {
        "prediction": prediction,
        "confidence": round(confidence, 4)
    }

@app.post("/predict-batch")
async def predict_batch(files: List[UploadFileX] = File(...)):
    results = []
    log_path = "inference_log.txt"

    for file in files:
        contents = await file.read()

        # ---------- IMAGE ----------
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)

        # ---------- MODEL ----------
        with torch.no_grad():
            output = model(image_tensor)
            prob = output.item()

        prediction = "ulcer" if prob > 0.5 else "normal"
        confidence = prob if prob > 0.5 else 1 - prob

        # ---------- BASE64 PREVIEW ----------
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode()

        # ---------- LOG ----------
        log_line = f"{datetime.now().isoformat()} | {file.filename} | {prediction} | {round(confidence,4)}\n"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_line)

        results.append({
            "filename": file.filename,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "image_base64": img_base64
        })

    return results