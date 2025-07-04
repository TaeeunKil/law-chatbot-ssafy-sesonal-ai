"""
ragas_sql_ragsystem_style_upstage.py
------------------------------------
SQLRetriever + Upstage Solar Pro 2 + RAGAS 5 메트릭 평가
(동시성 제한·재시도 백오프 포함)
"""
import os, json, logging, psycopg2
import pandas as pd
from types import SimpleNamespace
from datasets import Dataset
from dotenv import load_dotenv
from tqdm import tqdm; tw = tqdm.write

from openai import OpenAI                         # 임베딩용
from langchain_upstage.chat_models import ChatUpstage
from langchain_core.prompts   import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

from ragas import evaluate
from ragas.metrics import (
    context_precision, context_recall,
    faithfulness,      answer_relevancy,
    answer_similarity,
)
from ragas.llms        import LangchainLLMWrapper
from ragas.embeddings  import LangchainEmbeddingsWrapper
from ragas.run_config  import RunConfig            # 동시성 컨트롤러
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
EMBED_MODEL = "text-embedding-3-small"
def get_embedding(text: str) -> list[float]:
    return openai.embeddings.create(
        model=EMBED_MODEL, input=text, encoding_format="float"
    ).data[0].embedding

def search_similar(text: str, k: int = 1):
    import json
    vec = json.dumps(get_embedding(text))
    sql = """
      WITH q AS (SELECT %s::vector AS v)
      SELECT input, output
        FROM question_embeddings, q
   ORDER BY embedding <=> q.v
       LIMIT %s;
    """
    with psycopg2.connect(**DB) as conn, conn.cursor() as cur:
        cur.execute(sql, (vec, k))
        return cur.fetchall()

class SQLRetriever:
    def __init__(self, top_k=1): self.top_k = top_k
    def invoke(self, query: str, **_):
        return [SimpleNamespace(page_content=out)
                for _, out in search_similar(query, self.top_k)]
retriever = SQLRetriever()

# ─────────── 2) LLM + 프롬프트 ───────────
llm = ChatUpstage(
    model_name  = "solar-pro2-preview",
    temperature = 0.2,
    max_retries = 6,          # 자동 백오프
)
prompt = PromptTemplate.from_template("""당신은 민법에 대해서 잘 알고 있는 변호사입니다.
고객의 질문에 상세하게 답을 해줄 수 있다면, 일반인이 알아들을 수 있게 풀어서 설명하세요.
해당 되는 법조항이 있다면 법조항을 포함하여 답변을 해주세요.
만약 질문에 대한 답변이 없다면 "해당 경우에 대한 판례가 없기 때문에 답변을 드리기 어렵습니다" 라고 답하세요.
판례를 찾을 수 있다면 판례를 포함하여 답변을 해주세요.

#Context:
{reference}

#Question:
{question}

#Answer:
""")
chain = RunnableSequence(
    {"question": RunnablePassthrough(),
     "reference": RunnablePassthrough()},
    prompt,
    llm,
)

# ─────────── 3) RAGAS 래퍼 ───────────
class LocalEmbeddings:
    def embed_query(self, t):      return get_embedding(t)
    def embed_documents(self, ds): return [get_embedding(d) for d in ds]
    async def aembed_query(self, t):      return self.embed_query(t)
    async def aembed_documents(self, ds): return self.embed_documents(ds)

ragas_llm = LangchainLLMWrapper(llm)
ragas_emb = LangchainEmbeddingsWrapper(LocalEmbeddings())

# ─────────── 4) 데이터 생성 ───────────
with open("eval_dataset_j.json", encoding="utf-8") as f:
    qna = json.load(f)
tw(f"질문 {len(qna)}개 로드 완료")

records = []
for item in tqdm(qna, desc="Generating", ncols=80):
    q, ref = item["input"], item["output"]
    ctx_docs = retriever.invoke(q)
    ctx      = "\n\n---\n\n".join(d.page_content for d in ctx_docs)
    answer   = chain.invoke({"question": q, "reference": ctx}).content
    records.append({
        "question":  q,
        "answer":    answer,
        "contexts":  [d.page_content for d in ctx_docs],
        "reference": ref,
    })

# ─────────── 5) RAGAS 평가 ───────────
run_cfg = RunConfig(max_workers=3, timeout=120, max_wait=60)
eval_ds = Dataset.from_pandas(pd.DataFrame(records))
scores  = evaluate(
    dataset     = eval_ds,
    metrics     = [
        context_precision, context_recall,
        faithfulness, answer_relevancy, answer_similarity,
    ],
    llm         = ragas_llm,
    embeddings  = ragas_emb,
    run_config  = run_cfg,
    raise_exceptions=False,
)

scores.to_pandas().to_csv("ragas_scores_upstage4.csv",
                          index=False, encoding="utf-8-sig")
tw("✅ RAGAS 메트릭 저장 → ragas_scores_upstage3.csv")
