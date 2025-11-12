# sentiment_absa.py
"""
ABSA 감성 분석 파이프라인
- 모델 로드
- 문장 단위 감성 분석
- CSV 결과 저장
"""

import torch
import torch.nn.functional as F
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import argparse


class ABSAModel:
    def __init__(self, model_path: str):
        print(f"🔹 모델 로드 중... ({model_path})")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        print("✅ 모델 로드 완료")

    def analyze_sentiment(self, sentence: str):
        """
        단일 문장 감성 분석
        """
        inputs = self.tokenizer(sentence, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1)
            label_id = torch.argmax(probs, dim=1).item()
            confidence = probs[0][label_id].item()
        return label_id, confidence


def run_absa(model_path, input_csv, output_csv):
    """
    전체 파이프라인 실행
    """
    # 데이터 로드
    df = pd.read_csv(input_csv)
    print(f"📄 입력 데이터 로드 완료: {len(df)}개 문장")

    # 모델 로드
    model = ABSAModel(model_path)

    # 감성 분석 수행
    sentiments, confidences = [], []
    for sentence in tqdm(df['sentence'], desc="감성 분석 중"):
        label, conf = model.analyze_sentiment(sentence)
        sentiments.append(label)
        confidences.append(conf)

    df['pred_label'] = sentiments
    df['confidence'] = confidences


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABSA 감성 분석 실행 스크립트")
    parser.add_argument("--model_path", type=str, required=True, help="모델 디렉토리 경로")
    parser.add_argument("--input_csv", type=str, required=True, help="입력 CSV 파일 경로")
    parser.add_argument("--output_csv", type=str, default="absa_results.csv", help="출력 파일 경로")
    args = parser.parse_args()

    run_absa(args.model_path, args.input_csv, args.output_csv)
