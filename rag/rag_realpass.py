import os
import json
import psycopg2
from dotenv import load_dotenv
from openai import OpenAI

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence

# ————————— 환경 변수 & 클라이언트 셋업 —————————
load_dotenv()
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
def search_similar(text: str, k: int = 3):
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
        return cur.fetchall()   # [(id, input, output, score), …]

# ————————— 3) LangChain 체인 준비 (RunnableSequence) —————————
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

# ————————— 4) 실행 예시 —————————
if __name__ == "__main__":
    q = "민사집행법에서 집행관의 권한은 무엇인가요?"
    print("\n# 질문\n", q)

    # 4-1) SQL로 유사도 검색
    rows = search_similar(q, k=3)
    print("\n# 검색된 QnA")
    for id_, inp, outp, score in rows:
        print(f"id={id_} score={score:.4f}")
    
    # 4-2) reference 문자열 조립
    reference = "\n\n".join(f"[{score:.4f}] {inp} → {outp}" for _, inp, outp, score in rows)
    print("\n# reference\n", reference)

    # 4-3) RunnableSequence.invoke() 로 최종 답변 생성
    result = chain.invoke({"question": q, "reference": reference})
    print("\n# 답변\n", result.content)
