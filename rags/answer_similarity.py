#!/usr/bin/env python3
# compute_answer_similarity_from_csv.py

import os
import pandas as pd
from datasets import Dataset
from dotenv import load_dotenv

from openai import OpenAI
from langchain_openai import ChatOpenAI
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import answer_similarity

# ───────────────────────────────────────────────────
# 0) 환경변수 로드 & 클라이언트 초기화
# ───────────────────────────────────────────────────
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise RuntimeError("OPENAI_API_KEY 환경변수를 설정하세요.")
openai = OpenAI(api_key=openai_api_key)
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

# ───────────────────────────────────────────────────
# 1) embeddings 래퍼 정의 (동기 + 비동기)
# ───────────────────────────────────────────────────
class MyEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        resp = openai.embeddings.create(
            model="text-embedding-3-small",
            input=text,
            encoding_format="float",
        )
        return resp.data[0].embedding

    def embed_documents(self, docs: list[str]) -> list[list[float]]:
        return [self.embed_query(d) for d in docs]

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    async def aembed_documents(self, docs: list[str]) -> list[list[float]]:
        return self.embed_documents(docs)

embeddings = MyEmbeddings()

# ───────────────────────────────────────────────────
# 2) 질문/답변/정답 쌍 로드 (CSV에서)
# ───────────────────────────────────────────────────
# UTF-8 BOM 인코딩된 CSV 파일 사용
csv_path = "upstage_answer_similarity_only.csv"
df = pd.read_csv(csv_path, encoding="utf-8-sig")

# 컬럼명 변환: input → question, response → answer
df = df.rename(columns={
    "user_input":    "question",
    "response": "answer",
    # 'reference' 컬럼은 그대로 사용
})

# Dataset 생성: question, answer, reference 세 컬럼 필요
eval_ds = Dataset.from_pandas(df[["question", "answer", "reference"]])

# ───────────────────────────────────────────────────
# 3) RAGAS Answer Similarity 평가 실행
# ───────────────────────────────────────────────────
ragas_llm = LangchainLLMWrapper(llm)
ragas_emb = LangchainEmbeddingsWrapper(embeddings)

scores = evaluate(
    dataset    = eval_ds,
    metrics    = [answer_similarity],
    llm        = ragas_llm,
    embeddings = ragas_emb,
)

# ───────────────────────────────────────────────────
# 4) 결과 출력 및 저장
# ───────────────────────────────────────────────────
df_scores = scores.to_pandas()
pd.set_option("display.precision", 3)

# 터미널 출력
print(df_scores)

# CSV 저장
output_csv = "upstage_answer_similarity_only_res.csv"
df_scores.to_csv(output_csv, index=False, encoding="utf-8-sig")
print(f"Answer Similarity 점수를 '{output_csv}'에 저장했습니다.")
