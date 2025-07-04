# search_local_json.py
"""
pip install openai>=1.3 numpy python-dotenv
.env → OPENAI_API_KEY 만 있으면 됨
"""

import os, json, numpy as np
from dotenv import load_dotenv
import openai

# ─────────────────────────────
# 0) 환경 변수 & OpenAI 설정
# ─────────────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")
MODEL = "text-embedding-3-small"

# ─────────────────────────────
# 1) 로컬 임베딩 로드 & 정규화
# ─────────────────────────────
with open("new_question_embeddings2.json", encoding="utf-8") as f:
    DOCS = json.load(f)

# (n, 1536) 행렬로 변환
EMB_MATRIX = np.asarray([d["embedding"] for d in DOCS], dtype=np.float32)
EMB_MATRIX /= np.linalg.norm(EMB_MATRIX, axis=1, keepdims=True)   # L2 정규화

# ─────────────────────────────
# 2) 쿼리 문장 → 임베딩
# ─────────────────────────────
def get_embedding(text: str) -> np.ndarray:
    resp = openai.embeddings.create(
        model=MODEL,
        input=text,
        encoding_format="float",
    )
    vec = np.asarray(resp.data[0].embedding, dtype=np.float32)
    return vec / np.linalg.norm(vec)            # 코사인용 정규화

# ─────────────────────────────
# 3) Top-k 코사인 유사도 검색
# ─────────────────────────────
def search_similar(query: str, k: int = 5):
    q_vec   = get_embedding(query)                         # (1536,)
    scores  = EMB_MATRIX @ q_vec                           # dot = cosine
    top_idx = scores.argsort()[-k:][::-1]                  # 상위 k개 내림차순
    return [
        {
            "rank": r + 1,
            "id":   DOCS[i]["id"],
            "score": float(scores[i]),
            "Q": DOCS[i]["input"],
            "A": DOCS[i]["output"],
        }
        for r, i in enumerate(top_idx)
    ]

# ─────────────────────────────
# 4) 사용 예시
# ─────────────────────────────
if __name__ == "__main__":
    query = "판결서의 비실명 처리에 대한 규정은 무엇인가요?"
    print(f"origin Q: {query}\n")
    for hit in search_similar(query, k=3):
        print(f"[score {hit['score']:.4f}] id={hit['id']}\nQ: {hit['Q']}\nA: {hit['A']}\n")
