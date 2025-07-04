import pandas as pd, time
from datasets import Dataset
from ragas import evaluate
from ragas.llms       import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics    import (
    context_precision, context_recall,
    faithfulness,      answer_relevancy,
)
import os
import json
import psycopg2
from dotenv import load_dotenv             # ← 추가
from openai import OpenAI
from types import SimpleNamespace

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence
from tqdm import tqdm       
# ————————— 환경 변수 & 클라이언트 셋업 —————————
load_dotenv()                              # ← dotenv 로드
DB = {
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT"),
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}
openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ————————— 1) 임베딩 조회 함수 —————————
def get_embedding(text: str) -> list[float]:
    resp = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text,
        encoding_format="float",
    )
    return resp.data[0].embedding

# ————————— 2) pgvector SQL 검색 함수 —————————
def search_similar(text: str, k: int = 5):
    vec = json.dumps(get_embedding(text))
    sql = """
    WITH q AS (SELECT %s::vector AS v)
    SELECT id, input, output, 1 - (embedding <=> q.v) AS score
      FROM question_embeddings, q
     ORDER BY embedding <=> q.v
     LIMIT %s;
    """
    with psycopg2.connect(**DB) as conn, conn.cursor() as cur:
        cur.execute(sql, (vec, k))
        return cur.fetchall()

# ————————— retriever 래퍼 정의 —————————
class SQLRetriever:
    def __init__(self, top_k: int = 5):
        self.top_k = top_k

    def invoke(self, query: str, top_k: int | None = None):
        k = top_k or self.top_k
        rows = search_similar(query, k=k)
        return [SimpleNamespace(page_content=output) for _, _, output, _ in rows]

retriever = SQLRetriever(top_k=3)

# ————————— LangChain LLM & Prompt —————————
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

prompt = PromptTemplate.from_template(
    """당신은 민법 전문가입니다. 다음 reference를 참고하여 Question에 답변해주세요.

#Question:
{question}

#reference:
{reference}

#Answer:"""
)

chain = RunnableSequence(
    {"question": RunnablePassthrough(), "reference": RunnablePassthrough()},
    prompt,
    llm,
)

# ————————— embeddings 래퍼 정의 —————————
class MyEmbeddings:
    def embed_query(self, text: str) -> list[float]:
        return get_embedding(text)

    def embed_documents(self, docs: list[str]) -> list[list[float]]:
        return [get_embedding(d) for d in docs]

embeddings = MyEmbeddings()

# 1) 질문 리스트 + 실제 정답(ground truths)
# 1) JSON 로드
with open("eval_dataset_j.json", "r", encoding="utf-8") as f:
    qna = json.load(f)

questions      = [item["input"]  for item in qna]
ground_truths  = [item["output"] for item in qna]

# 2) 질의 → (answer, contexts, reference) 수집
records = []
for q, gt in tqdm(zip(questions, ground_truths), total=len(questions),  # ★ tqdm 래핑
                  desc="Generating", ncols=80):
    ctx_docs = retriever.invoke(q)
    ctx_list = [d.page_content for d in ctx_docs]

    result  = chain.invoke({
        "question":  q,
        "reference": "\n\n---\n\n".join(ctx_list)
    })
    answer = result.content

    records.append({
        "question":  q,
        "answer":    answer,
        "contexts":  ctx_list,
        "reference": gt,
    })
    time.sleep(0.4)

eval_ds = Dataset.from_pandas(pd.DataFrame(records))

# 3) RAGAS 4종 지표 계산
ragas_llm = LangchainLLMWrapper(llm)
ragas_emb = LangchainEmbeddingsWrapper(embeddings)

scores = evaluate(
    dataset    = eval_ds,
    metrics    = [context_precision, context_recall, faithfulness, answer_relevancy],
    llm        = ragas_llm,
    embeddings = ragas_emb,
)

# DataFrame으로 변환
df_scores = scores.to_pandas()
pd.set_option("display.precision", 3)


# 2) CSV로 저장 (선택)
df_scores.to_csv("ragas_scores_j2.csv", index=False, encoding="utf-8-sig")
print("RAGAS 점수를 ragas_scores.csv에 저장했습니다.")