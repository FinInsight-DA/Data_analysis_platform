# sentiment_labeling.py
import os
import random
import numpy as np
import pandas as pd
import torch
import json
from io import TextIOWrapper
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, TrainerCallback

os.environ["WANDB_DISABLED"] = "true"

# ----------------------------
# 0️⃣ Seed 고정
# ----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# ----------------------------
# 1️⃣ 감성사전 로드
# ----------------------------
def load_domain_sentiment_dict(file_input):
    POS_BASE, NEG_BASE, ASPECT_POS, ASPECT_NEG = [], [], {}, {}
    filename = getattr(file_input, "name", str(file_input)).lower()
    try:
        if filename.endswith(".json"):
            if isinstance(file_input, str):
                with open(file_input, "r", encoding="utf-8") as f:
                    d = json.load(f)
            else:
                raw = file_input.read()
                if isinstance(raw, bytes):
                    raw = raw.decode("utf-8")
                d = json.loads(raw)
            POS_BASE = d.get("POS_BASE", [])
            NEG_BASE = d.get("NEG_BASE", [])
            ASPECT_POS = d.get("ASPECT_POS", {})
            ASPECT_NEG = d.get("ASPECT_NEG", {})
        elif filename.endswith(".txt"):
            if isinstance(file_input, str):
                with open(file_input, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            else:
                if isinstance(file_input, TextIOWrapper):
                    lines = file_input.readlines()
                else:
                    lines = file_input.read().decode("utf-8").splitlines()
            current_list = POS_BASE
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                if line.startswith("#POS"):
                    current_list = POS_BASE
                    continue
                elif line.startswith("#NEG"):
                    current_list = NEG_BASE
                    continue
                current_list.append(line)
        else:
            raise ValueError("지원하지 않는 파일 형식입니다. (.json 또는 .txt만 가능)")
    except Exception as e:
        print(f"[ERROR] 감성사전 로드 실패: {e}")
    return POS_BASE, NEG_BASE, ASPECT_POS, ASPECT_NEG

# ----------------------------
# 2️⃣ 규칙 기반 라벨링
# ----------------------------
def rule_label_sentence_vectorized(sentences, aspects, POS_BASE, NEG_BASE, ASPECT_POS, ASPECT_NEG):
    pos_base_set = set(POS_BASE)
    neg_base_set = set(NEG_BASE)
    aspect_pos_sets = {k: set(v) for k, v in ASPECT_POS.items()}
    aspect_neg_sets = {k: set(v) for k, v in ASPECT_NEG.items()}

    results = []
    for s, a in zip(sentences, aspects):
        s_str = str(s)
        pos_hit = any(word in s_str for word in pos_base_set)
        neg_hit = any(word in s_str for word in neg_base_set)
        if a in aspect_pos_sets:
            pos_hit |= any(word in s_str for word in aspect_pos_sets[a])
            neg_hit |= any(word in s_str for word in aspect_neg_sets.get(a, set()))
        if pos_hit and not neg_hit:
            results.append("positive")
        elif neg_hit and not pos_hit:
            results.append("negative")
        else:
            results.append("neutral")
    return results

def apply_rule_labeling(df, POS_BASE, NEG_BASE, ASPECT_POS, ASPECT_NEG, text_col="sentence",
                        aspect_col="aspect_category", progress_callback=None):
    total = len(df)
    df["sentiment"] = ""
    df["sentiment_source"] = ""
    df["sentiment_confidence"] = 0.0

    for idx, row in enumerate(df.itertuples(), 1):
        df.at[row.Index, "sentiment"] = rule_label_sentence_vectorized(
            [getattr(row, text_col)],
            [getattr(row, aspect_col, None)],
            POS_BASE, NEG_BASE, ASPECT_POS, ASPECT_NEG
        )[0]
        df.at[row.Index, "sentiment_source"] = "rule" if df.at[row.Index, "sentiment"] in ["positive","negative"] else "rule-neutral"
        df.at[row.Index, "sentiment_confidence"] = 1.0 if df.at[row.Index, "sentiment"] in ["positive","negative"] else 0.0

        if progress_callback:
            progress_callback(stage="rule", current=idx, total=total)
    return df

# ----------------------------
# 3️⃣ TrainingArguments 생성 (4.x 호환)
# ----------------------------
def _make_training_args(output_dir, lr, batch_size, num_epochs):
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        logging_strategy="steps",
        logging_steps=10,
        save_strategy="no",
        weight_decay=0.01,
        disable_tqdm=True,
        report_to=[],
        eval_steps=None,
    )


# ----------------------------
# Streamlit용 Trainer Callback
# ----------------------------
class StreamlitTrainerCallback(TrainerCallback):
    def __init__(self, progress_callback, total_steps):
        self.progress_callback = progress_callback
        self.total_steps = total_steps
        self.current = 0

    def on_step_end(self, args, state, control, **kwargs):
        self.current += 1
        if self.progress_callback:
            self.progress_callback(
                stage="model_train",
                current=self.current,
                total=self.total_steps
            )

# ----------------------------
# 4️⃣ 모델 학습 및 중립 문장 라벨링
# ----------------------------
def train_and_label_neutral_sentences(df, POS_BASE, NEG_BASE, ASPECT_POS, ASPECT_NEG,
                             model_name="monologg/koelectra-base-v3-discriminator",
                             text_col="sentence", num_epochs=3, lr=5e-5, batch_size=16,
                             val_size=0.1, seed=42, pred_threshold=0.5, progress_callback=None,
                             stage_name="neutral_labeling"):

    set_seed(seed)
    MODEL_OUTPUT_DIR_TEMP = "./temp_model"

    train_df = df[df["sentiment"].isin(["positive","negative"])][[text_col,"sentiment"]].copy()
    if len(train_df) < 50:
        print("[WARN] 학습 데이터가 너무 적어 중립 문장 라벨링 건너뜀")
        return df

    label2id = {"positive":0,"neutral":1,"negative":2}
    id2label = {v:k for k,v in label2id.items()}

    tr_df, va_df = train_test_split(train_df, test_size=val_size, random_state=seed, stratify=train_df["sentiment"])
    tr_ds = Dataset.from_pandas(tr_df.reset_index(drop=True))
    va_ds = Dataset.from_pandas(va_df.reset_index(drop=True))

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tr_ds = tr_ds.map(lambda x: tokenizer(x[text_col], truncation=True, padding="max_length", max_length=128), batched=True)
    va_ds = va_ds.map(lambda x: tokenizer(x[text_col], truncation=True, padding="max_length", max_length=128), batched=True)
    tr_ds = tr_ds.map(lambda x: {"labels": label2id[x["sentiment"]]}, batched=False)
    va_ds = va_ds.map(lambda x: {"labels": label2id[x["sentiment"]]}, batched=False)
    tr_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])
    va_ds.set_format(type="torch", columns=["input_ids","attention_mask","labels"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    model.to(device)

    # ----------------------------
    # Trainer + Streamlit callback
    # ----------------------------
    training_args = _make_training_args(MODEL_OUTPUT_DIR_TEMP, lr, batch_size, num_epochs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tr_ds,
        eval_dataset=va_ds,
        tokenizer=tokenizer
    )

    # Trainer 기준 총 step
    total_steps = trainer.state.max_steps or ((len(tr_ds) // batch_size + int(len(tr_ds) % batch_size != 0)) * num_epochs)
    # callback 연결
    trainer.add_callback(StreamlitTrainerCallback(progress_callback, total_steps))

    trainer.train()

    # ----------------------------
    # 중립 문장 예측
    # ----------------------------
    target_mask = df["sentiment"].eq("neutral")
    target_df = df[target_mask].copy()
    if target_df.empty:
        return df

    model.eval()
    batch_size_pred = 64
    total = len(target_df)
    preds, probs_all = [], []

    for i in range(0, total, batch_size_pred):
        batch = target_df[text_col].iloc[i:i+batch_size_pred].tolist()
        enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128)
        enc = {k:v.to(device) for k,v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds.extend(np.argmax(probs, axis=1))
        probs_all.extend(probs.tolist())

        if progress_callback:
            progress_callback(stage="neutral_labeling", current=i+len(batch), total=total)

    probs_all = np.array(probs_all)
    pred_labels = [id2label[p] for p in preds]
    max_probs = probs_all.max(axis=1)
    accept = max_probs >= pred_threshold

    df.loc[target_mask,"sentiment"] = [pred_labels[i] if accept[i] else df.loc[target_mask].iloc[i]["sentiment"]
                                        for i in range(len(pred_labels))]
    df.loc[target_mask,"sentiment_source"] = ["neutral_model" if accept[i] else df.loc[target_mask].iloc[i]["sentiment_source"]
                                              for i in range(len(pred_labels))]
    df.loc[target_mask,"sentiment_confidence"] = [max_probs[i] if accept[i] else df.loc[target_mask].iloc[i]["sentiment_confidence"]
                                                  for i in range(len(pred_labels))]
    return df

# ----------------------------
# 5️⃣ 통합 실행
# ----------------------------
def run_sentiment_labeling(data_file, sentiment_file, progress_callback=None):
    POS_BASE, NEG_BASE, ASPECT_POS, ASPECT_NEG = load_domain_sentiment_dict(sentiment_file)
    df = pd.read_csv(data_file)
    total = len(df)

    # 1️⃣ 규칙 기반 라벨링
    if progress_callback:
        progress_callback(stage="rule", current=0, total=total)
    df = apply_rule_labeling(df, POS_BASE, NEG_BASE, ASPECT_POS, ASPECT_NEG, progress_callback=progress_callback)

    # 2️⃣ positive/negative 학습 및 neutral 라벨링
    train_df = df[df["sentiment"].isin(["positive","negative"])].copy()
    if len(train_df) >= 50:
        df = train_and_label_neutral_sentences(df, POS_BASE, NEG_BASE, ASPECT_POS, ASPECT_NEG,
                                               progress_callback=progress_callback, stage_name="neutral_labeling")
    else:
        print("[WARN] 학습 데이터가 너무 적어 중립 문장 라벨링 건너뜀")

    return df
