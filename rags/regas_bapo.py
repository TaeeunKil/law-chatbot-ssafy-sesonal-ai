"""
ragas_sql_ragsystem_style.py
----------------------------
SQLRetriever + gpt-4o-mini 를 이용해 RAGAS 5메트릭 평가
(rag_system 모듈은 불러오지 않음)
"""

import os, json, time, logging
import pandas as pd
import psycopg2
from types import SimpleNamespace
from datasets import Dataset
from dotenv import load_dotenv
from tqdm import tqdm; tw = tqdm.write

from openai import OpenAI
from langchain_openai     import ChatOpenAI
from langchain_core.prompts   import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

from ragas import evaluate
from ragas.metrics import (
    context_precision, context_recall,
    faithfulness,      answer_relevancy,
    answer_similarity,
)
from ragas.llms       import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper

# ─────────── 0) ENV & 로깅 ───────────
load_dotenv()
logging.basicConfig(level=logging.WARNING)

DB = dict(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
)

openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ─────────── 1) Embedding + SQL 검색 ───────────
EMBED_MODEL = "text-embedding-3-small"      ### CHANGED

def get_embedding(text: str) -> list[float]:
    return openai.embeddings.create(
        model=EMBED_MODEL,
        input=text,
        encoding_format="float",
    ).data[0].embedding

def search_similar(text: str, k: int = 4):   ### CHANGED (k=4, rag_system 기본)
    vec = json.dumps(get_embedding(text))
    sql = """
      WITH q AS (SELECT %s::vector AS v)
      SELECT input, output
        FROM question_embeddings, q
   ORDER BY embedding <=> q.v              -- cosine 거리
       LIMIT %s;
    """
    with psycopg2.connect(**DB) as conn, conn.cursor() as cur:
        cur.execute(sql, (vec, k))
        return cur.fetchall()

class SQLRetriever:
    def __init__(self, top_k=4): self.top_k = top_k
    def invoke(self, query: str, **_):
        return [SimpleNamespace(page_content=out)
                for _, out in search_similar(query, self.top_k)]

retriever = SQLRetriever()

# ─────────── 2) LLM Prompt (rag_system 과 동일) ───────────
LLM_MODEL = "gpt-4o-mini"                  ### CHANGED
llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0.2)

prompt_text = """당신은 민법에 대해서 잘 알고 있는 변호사입니다.
고객의 질문에 상세하게 답을 해줄 수 있다면, 일반인이 알아들을 수 있게 풀어서 설명하세요.
해당 되는 법조항이 있다면 법조항을 포함하여 답변을 해주세요.
만약 질문에 대한 답변이 없다면 "해당 경우에 대한 판례가 없기 때문에 답변을 드리기 어렵습니다" 라고 답하세요.
판례를 찾을 수 있다면 판례를 포함하여 답변을 해주세요.

#Context:
{reference}

#Question:
{question}

#Answer:"""                                          ### CHANGED (rag_system prompt)

prompt = PromptTemplate.from_template(prompt_text)

chain = RunnableSequence(
    {"question": RunnablePassthrough(),
     "reference": RunnablePassthrough()},
    prompt,
    llm,
)

# ─────────── 3) Embedding 래퍼 (RAGAS용) ───────────
class LocalEmbeddings:
    def embed_query(self, text):      return get_embedding(text)
    def embed_documents(self, docs):  return [get_embedding(d) for d in docs]
    async def aembed_query(self, text):
        return self.embed_query(text)

    async def aembed_documents(self, docs):
        return self.embed_documents(docs)


ragas_llm = LangchainLLMWrapper(llm)
ragas_emb = LangchainEmbeddingsWrapper(LocalEmbeddings())

# ─────────── 4) 평가용 데이터 로드 ───────────
with open("eval_dataset_j.json", encoding="utf-8") as f:
    qna = json.load(f)
tw(f"질문 {len(qna)}개 로드 완료")

records = []
for item in tqdm(qna, desc="Generating", ncols=80):
    q, ref = item["input"], item["output"]

    ctx_docs = retriever.invoke(q)
    ctx_list = [d.page_content for d in ctx_docs]

    answer = chain.invoke({
        "question":  q,
        "reference": "\n\n---\n\n".join(ctx_list)
    }).content

    records.append({
        "question":  q,
        "answer":    answer,
        "contexts":  ctx_list,
        "reference": ref,          # RAGAS ground-truth
    })
    time.sleep(0.25)

# ─────────── 5) RAGAS 평가 ───────────
eval_ds = Dataset.from_pandas(pd.DataFrame(records))

scores = evaluate(
    dataset    = eval_ds,
    metrics    = [
        context_precision,
        context_recall,
        faithfulness,
        answer_relevancy,
        answer_similarity,
    ],
    llm        = ragas_llm,
    embeddings = ragas_emb,
    raise_exceptions=False,
)

df_scores = scores.to_pandas()
df_scores.to_csv("ragas_scores_bapo.csv", index=False, encoding="utf-8-sig")
tw("✅ RAGAS 메트릭 저장 → ragas_scores_bapo3.csv")
