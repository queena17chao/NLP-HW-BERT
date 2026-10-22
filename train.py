
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from datasets import Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    set_seed,
)

# =====================================
# 1. 基本設定
# =====================================

set_seed(42)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "data.csv"
OUTPUT_DIR = BASE_DIR / "output"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"

MODEL_NAME = "google-bert/bert-base-chinese"

print("使用裝置：", "GPU" if torch.cuda.is_available() else "CPU")

# =====================================
# 2. 讀取資料
# =====================================

df = pd.read_csv(DATA_PATH)

df = df.dropna(subset=["text", "emotion"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df["emotion"] = df["emotion"].astype(str).str.strip()

df = df[(df["text"] != "") & (df["emotion"] != "")]

print("資料筆數：", len(df))

# =====================================
# 3. 情緒標籤轉換：文字 -> 整數
# =====================================

emotion_list = sorted(df["emotion"].unique())

label2id = {
    emotion: idx
    for idx, emotion in enumerate(emotion_list)
}

id2label = {
    idx: emotion
    for emotion, idx in label2id.items()
}

df["labels"] = df["emotion"].map(label2id).astype(int)

print("\n情緒標籤對照：")
for label, idx in label2id.items():
    print(f"{idx}: {label}")

# =====================================
# 4. 切分訓練集與驗證集
# =====================================

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df["labels"],
)

print("\n訓練集：", len(train_df))
print("驗證集：", len(val_df))

train_dataset = Dataset.from_pandas(
    train_df[["text", "labels"]],
    preserve_index=False,
)

val_dataset = Dataset.from_pandas(
    val_df[["text", "labels"]],
    preserve_index=False,
)

# =====================================
# 5. Tokenization
# =====================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,
    )

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

val_dataset = val_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"],
)

# =====================================
# 6. 載入 BERT 多元分類模型
# =====================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(emotion_list),
    label2id=label2id,
    id2label=id2label,
)

# =====================================
# 7. 評估指標
# =====================================

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    predictions = np.argmax(logits, axis=-1)

    accuracy = accuracy_score(labels, predictions)
    f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    return {
        "accuracy": accuracy,
        "f1_macro": f1,
    }

# =====================================
# 8. 訓練參數
# =====================================

training_args = TrainingArguments(
    output_dir=str(CHECKPOINT_DIR),
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,

    eval_strategy="epoch",
    save_strategy="epoch",

    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,

    save_total_limit=1,
    logging_strategy="epoch",
    fp16=torch.cuda.is_available(),
    report_to="none",
    seed=42,
)

# =====================================
# 9. 建立 Trainer
# =====================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    processing_class=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# =====================================
# 10. 開始訓練
# =====================================

print("\n開始 BERT 微調...")
trainer.train()

# =====================================
# 11. 模型評估
# =====================================

results = trainer.evaluate()

print("\n模型評估結果：")
print(results)

# =====================================
# 12. 儲存最佳模型
# =====================================

trainer.save_model(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

print("\n最佳模型已儲存至 output/")
