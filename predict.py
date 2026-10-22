
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

MODEL_PATH = "output"

texts = [
    "我每天都能跟她一起上學，我好開心！",
    "最好的朋友要離開臺灣了，以後可能不容易再見面...",
    "我覺得我快不行了...",
    "剛剛收到研究所錄取的通知書！",
    "今年的冬天好像比較晚來。"
]

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model.to(device)
model.eval()

for text in texts:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=-1)

    predicted_id = torch.argmax(probs, dim=-1).item()
    confidence = probs[0, predicted_id].item()

    emotion = model.config.id2label[predicted_id]

    print(f"{text} => {emotion} ({confidence:.4f})")
