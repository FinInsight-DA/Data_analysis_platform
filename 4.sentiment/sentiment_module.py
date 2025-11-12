# sentiment_module.py
import os
import io
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset as HFDataset

# ======================
# 전통 ML
# ======================
def run_traditional_ml(model_name, df, progress_callback=None):
    X = df['sentence']
    y = df['sentiment']

    vectorizer = TfidfVectorizer(max_features=5000)
    X_tfidf = vectorizer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    if model_name == "RF":
        model = RandomForestClassifier(random_state=42)
    elif model_name == "SVM":
        model = LinearSVC(random_state=42)
    elif model_name == "NB":
        model = MultinomialNB()
    else:
        raise ValueError(f"Unknown ML model: {model_name}")

    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')

    if progress_callback:
        for i in range(5):
            progress_callback(i + 1, 5, stage_name=f"{model_name} 진행 중")

    return {
        "Model": model_name,
        "Type": "Traditional ML",
        "Accuracy": round(acc, 4),
        "F1": round(f1, 4),
    }

# ======================
# 딥러닝 HF 모델
# ======================
def run_hf_model(model_name, short_name, df, progress_callback=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_train, X_test, y_train, y_test = train_test_split(
        df['sentence'], df['sentiment'], test_size=0.2, random_state=42
    )
    label2id = {"negative": 0, "neutral": 1, "positive": 2}
    id2label = {0: "negative", 1: "neutral", 2: "positive"}

    train_ds = HFDataset.from_dict({"text": X_train.tolist(), "labels": [label2id[v] for v in y_train]})
    test_ds = HFDataset.from_dict({"text": X_test.tolist(), "labels": [label2id[v] for v in y_test]})

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    def tokenize(batch):
        return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)
    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)
    train_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_ds.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=3, id2label=id2label, label2id=label2id,
        trust_remote_code=True, use_safetensors=True
    ).to(device)

    training_args = TrainingArguments(
        output_dir=f"./results_{short_name}",
        do_train=True,
        do_eval=True,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        num_train_epochs=1,
        logging_steps=10,
        save_total_limit=2,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    class ProgressCallback:
        def __init__(self, user_callback):
            self.user_callback = user_callback
            self.global_step = 0
            self.total_steps = training_args.num_train_epochs * len(train_ds) // training_args.per_device_train_batch_size

        def __call__(self, step=None):
            if self.user_callback:
                self.global_step += 1
                self.user_callback(self.global_step, self.total_steps, stage_name=f"{short_name} 학습 중")

    progress_cb = ProgressCallback(progress_callback)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds
    )

    if progress_callback:
        progress_callback(0, 1, stage_name=f"{short_name} 학습 시작")

    trainer.train()
    if progress_callback:
        progress_cb()

    preds_output = trainer.predict(test_ds)
    preds = torch.argmax(torch.tensor(preds_output.predictions), dim=1)
    y_test_encoded = [label2id[v] for v in y_test]
    acc = accuracy_score(y_test_encoded, preds)
    f1 = f1_score(y_test_encoded, preds, average='macro')

    return {
        "Model": short_name,
        "Type": "Deep Learning",
        "Accuracy": round(acc, 4),
        "F1": round(f1, 4),
    }

# ======================
# 선택 모델 실행
# ======================
def run_selected_models(selected_ml=None, selected_dl=None, input_csv=None, progress_callback=None):
    selected_ml = selected_ml or []
    selected_dl = selected_dl or []

    if isinstance(input_csv, (io.BytesIO, io.TextIOWrapper)):
        df = pd.read_csv(input_csv)
    elif isinstance(input_csv, str) and os.path.exists(input_csv):
        df = pd.read_csv(input_csv)
    else:
        raise ValueError("CSV 파일 필요")

    results = []

    # 전통 ML 모델 실행
    for ml in selected_ml:
        results.append(run_traditional_ml(ml, df, progress_callback=progress_callback))

    # 딥러닝 모델 실행
    for dl in selected_dl:
        if dl == "KoELECTRA":
            results.append(run_hf_model("monologg/koelectra-base-v3-discriminator", "KoELECTRA", df, progress_callback=progress_callback))
        elif dl == "KoBERT":
            results.append(run_hf_model("skt/kobert-base-v1", "KoBERT", df, progress_callback=progress_callback))
        elif dl == "BERT":
            results.append(run_hf_model("bert-base-uncased", "BERT", df, progress_callback=progress_callback))
        elif dl == "KoRoBERTa":
            results.append(run_hf_model("klue/roberta-base", "KoRoBERTa", df, progress_callback=progress_callback))

    return pd.DataFrame(results)
