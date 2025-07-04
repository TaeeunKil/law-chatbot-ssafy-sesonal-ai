from typing import List, Dict, Any, Optional
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.pgvector import PGVector
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
import json, os
import numpy as np
from dotenv import load_dotenv   
from langchain.docstore.document import Document

load_dotenv()
DB = {
    "host":     os.getenv("DB_HOST"),
    "port":     os.getenv("DB_PORT"),
    "dbname":   os.getenv("DB_NAME"),
    "user":     os.getenv("DB_USER"),
    "password": os.getenv("DB_PASSWORD"),
}
import psycopg2    

class RAGSystem:
    def __init__(self):
        # Model configurations
        self.embedding_model_name = 'text-embedding-3-small'  # Default embedding model
        self.llm_model_name = 'gpt-4o-mini'  # Default LLM model
        self.llm_temperature = 0.2  # Default temperature
        
        # Initialize the embedding model
        self.embedding_model = OpenAIEmbeddings(
            model=self.embedding_model_name,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        # HuggingFaceModel
        # self.embedding_model = HuggingFaceEmbeddings(
        #     model_name=self.embedding_model_name,
        #     model_kwargs={
        #         "device": "cuda" if torch.cuda.is_available() else "cpu"
        #     }
        # )
        
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model_name=self.llm_model_name,
            temperature=self.llm_temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Initialize the LLM for compression
        compressor_llm = ChatOpenAI(
            model_name=self.llm_model_name,
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Set up the connection string for PGVector
        self.connection_string = PGVector.connection_string_from_db_params(
            driver="psycopg2",
            host=DB['host'],
            port=DB['port'],
            database=DB['dbname'],
            user=DB['user'],
            password=DB['password']
        )
        
        # Initialize the vector store with the correct table name 'question_embeddings'
        # and specify the custom schema to match our database structure
        self.vector_store = PGVector(
            collection_name="question_embeddings",  # Matches the table name in the database
            connection_string=self.connection_string,
            embedding_function=self.embedding_model,
            collection_metadata={
                "table_name": "question_embeddings",  # Explicitly set the table name
                "content_column": "input",  # The column containing the text
                "metadata_column": "output",  # The column containing additional metadata
                "embedding_column": "embedding"  # The column containing the vector
            },
            use_jsonb=False  # Since we're not using JSONB in our schema
        )
        
        # Set up the retriever with MMR (Maximal Marginal Relevance)
        self.retriever = self.vector_store.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 4, "fetch_k": 20}
        )
        
        # Set up the prompt template
        self.prompt_template = """
        당신은 민법에 대해서 잘 알고 있는 변호사입니다.
        고객의 질문에 상세하게 답을 해줄 수 있다면, 일반인이 알아들을 수 있게 풀어서 설명하세요.
        해당 되는 법조항이 있다면 법조항을 포함하여 답변을 해주세요.
        만약 질문에 대한 답변이 없다면 "해당 경우에 대한 판례가 없기 때문에 답변을 드리기 어렵습니다" 라고 답하세요.
        판례를 찾을 수 있다면 판례를 포함하여 답변을 해주세요.
        
        Context: {context}
        
        Question: {question}
        
        Answer:
        """
        
        self.prompt = ChatPromptTemplate.from_template(self.prompt_template)
        
        # Set up the RAG chain
        self.rag_chain = (
            {"context": self.retriever, "question": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
    
    async def query(self, question: str, similar_docs: List[Dict[str, Any]] = None) -> str:
        """
        Query the RAG system with a question and optional similar documents.
        
        Args:
            question: The user's question
            similar_docs: List of similar documents to include in the context
            
        Returns:
            str: Generated response
        """
        try:
            if similar_docs and len(similar_docs) > 0:
                # Format the similar documents as context
                context = "\n\n".join([
                    f"Document {i+1}:\nInput: {doc.get('input', '')}\n"
                    f"Output: {doc.get('output', '')}"
                    for i, doc in enumerate(similar_docs)
                ])
                
                # Create a custom prompt with the similar documents as context
                custom_prompt = f"""
                다음은 사용자의 질문과 관련된 참고 문서들입니다. 이 문서들을 참고하여 질문에 답변해주세요.
                
                참고 문서:
                {context}
                
                사용자 질문: {question}
                
                답변:
                """
                
                # Use the LLM to generate a response with the custom prompt
                messages = [
                    ("system", "당신은 민법에 정통한 변호사입니다. 주어진 참고 문서를 바탕으로 사용자의 질문에 정확하고 상세하게 답변해주세요."),
                    ("user", custom_prompt)
                ]
                
                response = await self.llm.ainvoke(messages)
                return response.content
            else:
                # Fall back to the standard RAG chain if no similar docs provided
                result = await self.rag_chain.ainvoke(question)
                return result
                
        except Exception as e:
            return f"Error processing your query: {str(e)}"
    
    async def add_document(self, input_text: str, output_text: str) -> bool:
        """Add a new document to the vector store."""
        try:
            # Create a document with the input and output
            doc = Document(
                page_content=input_text,
                metadata={
                    "output": output_text,
                    "input": input_text  # Store input in metadata for retrieval
                }
            )
            
            # Add the document to the vector store
            self.vector_store.add_documents([doc])
            return True
        except Exception as e:
            print(f"Error adding document: {e}")
            return False
    
    async def search_similar(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents to the query."""
        try:
            # Get the embedding for the query
            query_embedding = await self.get_embedding(query)
            if not query_embedding:
                return []
                
            # Convert the query embedding to a string format for SQL query
            embedding_str = ','.join(map(str, query_embedding))
            
            # Perform similarity search using raw SQL to match our schema
            with psycopg2.connect(**DB) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT 
                            input, 
                            output,
                            1 - (embedding <=> %s) as similarity
                        FROM question_embeddings
                        ORDER BY embedding <=> %s
                        LIMIT %s;
                    """, (f"[{embedding_str}]", f"[{embedding_str}]", k))
                    
                    results = []
                    for row in cur.fetchall():
                        results.append({
                            'input': row[0],
                            'output': row[1],
                            'score': float(row[2])  # Convert numpy.float32 to Python float
                        })
                    
                    return results
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            import traceback
            traceback.print_exc()
            return []
            
    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """Get the embedding for a given text."""
        try:
            # Get the embedding for the text
            embedding = self.embedding_model.embed_query(text)
            return embedding.tolist() if hasattr(embedding, 'tolist') else embedding
        except Exception as e:
            print(f"Error getting embedding: {e}")
            return None

# Singleton instance
rag_system = RAGSystem()
