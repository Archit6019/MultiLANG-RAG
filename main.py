from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from FileProcessor import Processor
from groq import Groq
import os 
import uuid
import torch
from sentence_transformers import CrossEncoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#QDRANT_URL = os.environ.get("QDRANT_URL", None)
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

class DocumentProcessor:
    def __init__(self, embedder=None):
        self.embedder = embedder or SentenceTransformer("embaas/sentence-transformers-multilingual-e5-base", device=device)
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2", device=device)
        self.processor = Processor(embedder=self.embedder)
        self.groq_client = Groq(api_key=GROQ_API_KEY)
        self.qdrant_client = QdrantClient(":memory:")
        self.context = []
        self.context.append({
            "role": "system",
            "content": ("You are a helpful assistant, tasked with answering questions based on the provided context. "
                    "You will receive both the conversation history and additional relevant information, "
                    "referred to as Relevant_context, alongside the user's question. Relevant information will be provided as chunks of data. "
                    "The information you get might not be coherent, still try to answer the question by grasping the main idea from the information. "
                    "If you cannot answer the question based on the information provided in both the conversation history and relevant data, simply state "
                    "'I don't have enough information to answer this question accurately', do not answer questions outside of the scope of provided information. "
                    "For greetings also return 'I don't have enough information to answer this question accurately'. "
                    "Do not reference the context or the Relevant_context in your response; the users should not be aware that you have access to this information.")
        })

    def create_collection(self, name, vector_size):
        try:
            self.qdrant_client.create_collection(
                collection_name=name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE
                )
            )

            self.qdrant_client.create_payload_index(
                collection_name=name,
                field_name="document_name",
                field_type=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.PREFIX,
                    min_token_len=2,
                    max_token_len=256,
                )
            )
            return {"message": "Collection created successfully"}
        
        except Exception as e:
            return {"error": str(e)}

    def upload_document(self, collection_name, file_path, file_name, doc_type):

        try:
            with open(file_path, "rb") as f:
                file_content = f.read()

            output, chunk_len = self.processor.Preprocess(file_content=file_content)
            
            if output is None:
                return {"error": "Failed to process the document"}
            
            for i, doc_chunk in enumerate(output):
                unique_chunk_id = str(uuid.uuid4())
                self.qdrant_client.upsert(
                    collection_name=collection_name,
                    points=[
                        models.PointStruct(
                            id=unique_chunk_id,
                            vector=doc_chunk["embedding"],
                            payload={
                                "text": doc_chunk["text"],
                                "document_id": doc_chunk["document_id"],
                                "document_name": file_name,
                                "doc_type": doc_type
                            }
                        )
                    ]
                )
            
            return {"message": f"Document uploaded successfully, {len(output)} chunks generated"}
        
        except Exception as e:
            return {"error": str(e)}

    def query_decompose(self, current_query):

        system_prompt = """
        You are an AI assistant tasked with reformulating user queries based on conversation history. 
        Your goal is to create a standalone question that incorporates the context from previous messages if the current query is related to them.
        If the current query is unrelated to the previous conversation, simply restate the query as it is.
        The reformulated question should be clear, concise, and fully capture the user's intent, whether or not it is related to the prior context.
        Do not add any explanation or commentary. Only output the reformulated question.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Given this conversation history {self.context[-5:]} , rewrite the following query into a standalone question: '{current_query}'"}
        ]

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=messages,
                model="llama3-70b-8192"
            )

            return chat_completion.choices[0].message.content
        
        except Exception as e:
            return f"Error decomposing query: {str(e)}"

    def qdrant_search(self, collection_name, query):

        try:
            output = []
            result = self.qdrant_client.query_points(
                collection_name=collection_name,
                query=query,
                score_threshold=0.25,
                limit=25
            )

            if not result.points:
                return []
            
            for point in result.points:
                output.append(point.payload)
            

            return output
        
        except Exception as e:
            return {"error": str(e)}

    def chat(self, collection_name, user_message):

        try:
            decomposed_query = self.query_decompose(user_message)
            decomposed_query = f"query: {decomposed_query}"

            query_embeddings = self.embedder.encode(decomposed_query).tolist()

            output = self.qdrant_search(collection_name, query_embeddings)

            # Cross Encoder Logic
            pairs = [(decomposed_query, item['text']) for item in output]
            scores = self.cross_encoder.predict(pairs)
            scored_results = sorted(zip(scores, output), key=lambda x: x[0], reverse=True)
            output = [item for score, item in scored_results[:5]]

            if not output:
                text_context = ["No results found"]
            else:
                text_context = "\n".join(f"Chunk {i+1}:\n{item['text'].strip()}" for i, item in enumerate(output))

            self.context.append({
                "role": "user",
                "content": f"User query: {user_message}\n\nText context: {text_context}"
            })

            search_results = [
                {k: v for k, v in item.items() if k != 'text'} for item in output
            ]

            response = self.groq_client.chat.completions.create(
                model="llama3-70b-8192",
                messages=self.context,
                max_tokens=500
            )

            self.context.append({
                "role": "assistant",
                "content": response.choices[0].message.content
            })

            return {
                "ai_response": response.choices[0].message.content,
                "search_results": search_results
            }
        
        except Exception as e:
            return {"error": str(e)}


