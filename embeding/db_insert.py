import json
import os
import logging
import psycopg2
from dotenv import load_dotenv
from pgvector.psycopg2 import register_vector
from tqdm import tqdm

load_dotenv()

# ——— 로깅 설정 ———
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# ——— DB 연결 & pgvector 설정 ———
conn = psycopg2.connect(
    host=os.getenv("DB_HOST"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"),
    port=os.getenv("DB_PORT"),
)
register_vector(conn)
cur = conn.cursor()

# ——— 테이블 생성 (확인용) ———
cur.execute("""
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE IF NOT EXISTS question_embeddings (
    id        TEXT PRIMARY KEY,
    input     TEXT,
    output    TEXT,
    embedding vector(1536)
);
""")
conn.commit()
logging.info("테이블 확인/생성 완료")

# ——— JSON 로드 & 데이터 준비 ———
with open("_question_embeddings.json", encoding="utf-8") as f:
    rows = json.load(f)

insert_sql = """
INSERT INTO question_embeddings (id, input, output, embedding)
VALUES (%s, %s, %s, %s)
ON CONFLICT (id) DO NOTHING;
"""

data = [(row["id"], row["input"], row["output"], row["embedding"]) for row in rows]
total = len(data)
logging.info(f"총 {total}건의 레코드 준비 완료")

# ——— 진행 로그를 보여주며 일괄 삽입 ———
for i, record in enumerate(tqdm(data, desc="Inserting", unit="rows"), start=1):
    cur.execute(insert_sql, record)
    # 필요하다면 중간 커밋
    if i % 1000 == 0:
        conn.commit()
        logging.info(f"{i}/{total}건 커밋 완료")

# 마지막 커밋
conn.commit()
logging.info(f"모든 삽입 완료: {total}/{total}건")

# ——— 정리 ———
cur.close()
conn.close()
logging.info("DB 연결 종료")
