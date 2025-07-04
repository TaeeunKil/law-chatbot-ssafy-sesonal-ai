'''
비동기식으로 임베딩 빨리하기
사용한 데이터는 QnA를 input, output 쌍으로 구성된 json
'''



# fast_embed_save_dict.py
import os, json, pickle, asyncio, numpy as np, openai
from tqdm import tqdm
from dotenv import load_dotenv                 # pip install python-dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

MODEL        = "text-embedding-3-small"
BATCH_SIZE   = 256
MAX_CONC     = 20
RETRY_LIMIT  = 5

# ---------- 1. 원본 데이터 읽기 -------------------------------------------
with open("preprocessed_data.json", encoding="utf-8") as f:
    rows = json.load(f)                        # [{input, output, casename}, ...]

texts = [r["input"] for r in rows]             # 질문만 추출
total = len(texts)

# ---------- 2. 배치 + 비동기로 임베딩 -------------------------------------
import asyncio, math, time, openai
from openai import RateLimitError

def chunk(lst, size):
    """size 단위로 리스트 쪼개기 (len ≤ size 마지막 배치 포함)"""
    for i in range(0, len(lst), size):
        yield i, lst[i : i + size]          # (시작 인덱스, 배치)

async def embed_once(cli, batch):
    """단일 배치를 embedding → list[list[float]] 반환, 자동 재시도"""
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            res = await cli.embeddings.create(
                model=MODEL,
                input=batch,
            )
            # OpenAI v1.x: res.data 는 순서가 유지됨
            return [d.embedding for d in res.data]

        except (RateLimitError) as e:
            wait = 2 ** (attempt - 1)       # 1, 2, 4, 8, 16…
            if attempt == RETRY_LIMIT:
                raise RuntimeError(f"Embedding failed after {RETRY_LIMIT} tries") from e
            await asyncio.sleep(wait)

async def get_all_embeddings(texts):
    async with openai.AsyncOpenAI() as cli:         # v1.x 비동기 클라이언트
        sem         = asyncio.Semaphore(MAX_CONC)   # 동시 배치 수 제한
        embeddings  = [None] * len(texts)           # 자리 확보

        async def worker(start_idx, batch):
            async with sem:                         # acquire / release
                vecs = await embed_once(cli, batch)
                embeddings[start_idx : start_idx + len(batch)] = vecs

        tasks = [asyncio.create_task(worker(i, b))
                 for i, b in chunk(texts, BATCH_SIZE)]

        for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            await f                                 # 예외 전파

    return embeddings


embeds = asyncio.run(get_all_embeddings(texts))   # list[list[float]]

# ---------- 3. dict 형태로 재조합 -----------------------------------------
embedded_data = []
for i, (row, vec) in enumerate(zip(rows, embeds)):
    embedded_data.append({
        "id": f"q_{i}",
        "input": row["input"],
        "output": row.get("output", ""),
        "embedding": vec,                 # 1536-D list[float]
    })

# ---------- 4. Pickle 저장 -------------------------------------------------
# with open("fast_q_embeddings2.pkl", "wb") as f:
    # pickle.dump(embedded_data, f, protocol=pickle.HIGHEST_PROTOCOL)
with open("new_question_embeddings2.json", "w", encoding="utf-8") as f:
    json.dump(embedded_data, f, ensure_ascii=False, indent='\t')

print(f"✅ saved {len(embedded_data)} items → qfast_q_embeddings2.pkl")
