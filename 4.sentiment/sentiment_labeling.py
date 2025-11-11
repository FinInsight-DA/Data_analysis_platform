import os
import random
import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
import json
from io import TextIOWrapper
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, IntervalStrategy, TrainingArguments


# =========================
# 환경 설정
# =========================
os.environ["WANDB_DISABLED"] = "true"


# =========================
# 1️⃣ 감성사전 로드 함수 (도메인 제거 버전)
# =========================
def load_domain_sentiment_dict(file_input):
    """
    감성사전 로드 함수
    - file_input: 파일 경로(str) 또는 업로드된 파일 객체(BytesIO/StringIO)
    반환: POS_BASE, NEG_BASE, ASPECT_POS, ASPECT_NEG
    """
    POS_BASE, NEG_BASE, ASPECT_POS, ASPECT_NEG = [], [], {}, {}
    filename = getattr(file_input, "name", str(file_input)).lower()

    try:
        # -------- JSON 파일 --------
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

        # -------- TXT 파일 --------
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


# =========================
# 2️⃣ 규칙 기반 라벨링
# =========================
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
        elif pos_hit and neg_hit:
            results.append("neutral")
        else:
            results.append("neutral")
    return results

def apply_rule_labeling(df, POS_BASE, NEG_BASE, ASPECT_POS, ASPECT_NEG, text_col="sentence", aspect_col="aspect_category"):
    print("규칙 기반 라벨링 적용 중...")
    tqdm.pandas(desc="Rule-based labeling")
    df["sentiment"] = df.progress_apply(
        lambda r: rule_label_sentence_vectorized([r[text_col]], [r.get(aspect_col, None)], POS_BASE, NEG_BASE, ASPECT_POS, ASPECT_NEG)[0],
        axis=1
    )
    df["sentiment_source"] = np.where(df["sentiment"].isin(["positive", "negative"]), "rule", "rule-neutral")
    df["sentiment_confidence"] = np.where(df["sentiment"].isin(["positive", "negative"]), 1.0, 0.0)
    return df



# =========================
# 3️⃣ 학습용 모델 보조 라벨링
# =========================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _make_training_args(output_dir, lr, batch_size, num_epochs, weight_decay=0.01, logging_steps=50):
    return TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        learning_rate=lr,
        weight_decay=weight_decay,
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=logging_steps,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=4,
        run_name="koelectra-sentiment",
    )

def train_and_fill_with_model(df, POS_BASE, NEG_BASE, ASPECT_POS, ASPECT_NEG,
                             model_name="monologg/koelectra-base-v3-discriminator",
                             text_col="sentence", num_epochs=3, lr=5e-5, batch_size=16,
                             val_size=0.1, seed=42, pred_threshold=0.5):
    set_seed(seed)
    MODEL_OUTPUT_DIR_TEMP = "./temp_model"
    MODEL_OUTPUT_DIR_DRIVE = "./koelectra_absa"

    train_df = df[df["sentiment"].isin(["positive", "negative"])][[text_col, "sentiment"]].copy()
    if len(train_df) < 50:
        print("[WARN] 학습 데이터가 너무 적어 모델 보조 라벨링을 건너뜁니다.")
        return df

    # 라벨 매핑
    label2id = {"positive": 0, "neutral": 1, "negative": 2}
    id2label = {v: k for k, v in label2id.items()}

    # 데이터 분할
    tr_df, va_df = train_test_split(train_df, test_size=val_size, random_state=seed, stratify=train_df["sentiment"])
    tr_ds = Dataset.from_pandas(tr_df.reset_index(drop=True))
    va_ds = Dataset.from_pandas(va_df.reset_index(drop=True))

    # 토크나이저
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tr_ds = tr_ds.map(lambda x: tokenizer(x[text_col], truncation=True, padding="max_length", max_length=128), batched=True)
    va_ds = va_ds.map(lambda x: tokenizer(x[text_col], truncation=True, padding="max_length", max_length=128), batched=True)
    tr_ds = tr_ds.map(lambda x: {"labels": label2id[x["sentiment"]]}, batched=False)
    va_ds = va_ds.map(lambda x: {"labels": label2id[x["sentiment"]]}, batched=False)
    tr_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    va_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=3)
    args = _make_training_args(MODEL_OUTPUT_DIR_TEMP, lr, batch_size, num_epochs)
    trainer = Trainer(model=model, args=args, train_dataset=tr_ds, eval_dataset=va_ds,
                    tokenizer=tokenizer,
                    compute_metrics=lambda eval_pred: {
                        "acc": accuracy_score(eval_pred.label_ids, np.argmax(eval_pred.predictions, axis=1)),
                        "f1_macro": f1_score(eval_pred.label_ids, np.argmax(eval_pred.predictions, axis=1), average="macro")
                    })
    print("모델 학습 시작...") 
    trainer.train()

    # neutral 샘플만 예측
    target_mask = df["sentiment"].eq("neutral")
    target_df = df[target_mask].copy()
    if target_df.empty:
        print("모델로 예측할 neutral 샘플이 없습니다.")
        return df

    model.eval()
    preds, probs_all = [], []
    for i in tqdm(range(0, len(target_df), 64), desc="Model prediction"):
        batch = target_df[text_col].iloc[i:i+64].tolist()
        enc = tokenizer(batch, return_tensors="pt", truncation=True, padding=True, max_length=128)
        enc = {k: v.cuda() for k, v in enc.items()} if torch.cuda.is_available() else enc
        with torch.no_grad():
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds.extend(np.argmax(probs, axis=1))
        probs_all.extend(probs.tolist())

    probs_all = np.array(probs_all)
    pred_labels = [id2label[p] for p in preds]
    max_probs = probs_all.max(axis=1)
    accept = max_probs >= pred_threshold

    df.loc[target_mask, "sentiment"] = [pred_labels[i] if accept[i] else df.loc[target_mask].iloc[i]["sentiment"] for i in range(len(pred_labels))]
    df.loc[target_mask, "sentiment_source"] = ["model" if accept[i] else df.loc[target_mask].iloc[i]["sentiment_source"] for i in range(len(pred_labels))]
    df.loc[target_mask, "sentiment_confidence"] = [max_probs[i] if accept[i] else df.loc[target_mask].iloc[i]["sentiment_confidence"] for i in range(len(pred_labels))]

    return df

# =========================
# 4️⃣ 통합 실행 함수
# =========================
def run_sentiment_labeling(
    data_file, sentiment_file, output_csv="./output_labeled.csv"
):
    POS_BASE, NEG_BASE, ASPECT_POS, ASPECT_NEG = load_domain_sentiment_dict(
        sentiment_file
    )
    df = pd.read_csv(data_file)

    df = apply_rule_labeling(df, POS_BASE, NEG_BASE, ASPECT_POS, ASPECT_NEG)
    df = train_and_fill_with_model(df, POS_BASE, NEG_BASE, ASPECT_POS, ASPECT_NEG)


    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    print(f"✅ 결과 저장 완료: {output_csv}")
    return df


# =========================
# 5️⃣ 테스트용 main (Streamlit 아님)
# =========================
if __name__ == "__main__":
    DATA_PATH = "../data/lda_20_topics_selected_20251111_083817.csv"
    SENTI_PATH = "sentiment_dict.json"
    OUTPUT_PATH = "../data/sentiment_labeled_20251111_083817.csv"

    run_sentiment_labeling(DATA_PATH, SENTI_PATH, OUTPUT_PATH)
