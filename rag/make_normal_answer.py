"""
baseline_no_rag.py
------------------
RAG 없이 동일한 LLM(gpt-4o-mini)만으로 답변을 받아
RAG 결과와 직접 비교할 때 쓰는 최소 코드
(환경 변수, OpenAI 키 등은 기존 스크립트와 동일하게 가정)
"""
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

# ─────────── 0) 환경 변수 ───────────
load_dotenv()                 # .env에서 OPENAI_API_KEY 로드

# ─────────── 1) LLM 셋업 ───────────
LLM_MODEL = "gpt-4o-mini"
llm = ChatOpenAI(model_name=LLM_MODEL, temperature=0.2)

# ─────────── 2) 프롬프트 (Context 부분만 제거) ───────────
prompt_text = """당신은 민법에 대해서 잘 알고 있는 변호사입니다.
고객의 질문에 상세하게 답을 해줄 수 있다면, 일반인이 알아들을 수 있게 풀어서 설명하세요.
해당 되는 법조항이 있다면 법조항을 포함하여 답변을 해주세요.
만약 질문에 대한 답변이 없다면 "해당 경우에 대한 판례가 없기 때문에 답변을 드리기 어렵습니다" 라고 답하세요.
판례를 찾을 수 있다면 판례를 포함하여 답변을 해주세요.

#Question:
{question}

#Answer:"""

prompt  = PromptTemplate.from_template(prompt_text)
chain   = prompt | llm        # Prompt → LLM 파이프라인

# ─────────── 3) 한 번 호출해서 결과 확인 ───────────
if __name__ == "__main__":
    query = "명예훼손죄에서 '공연성'이 인정되는 요건은 무엇인가요?"
    answer = chain.invoke({"question": query}).content
    print("Q:", query)
    print("A:", answer)
