import os
import io
import base64
import PyPDF2
import logging
import uuid
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()

class embeddingModel_openai:
    def __init__(self):
        
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        
  
       
        self.connect_to_pinecone()
        self.initialize_index()
    
    def connect_to_pinecone(self):
        """Connect to Pinecone vector database."""
        try:
            self.pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
            logger.info("Connected to Pinecone")
        except Exception as e:
            logger.error(f"Error connecting to Pinecone: {e}")
            #st.error(f"Error connecting to Pinecone: {e}")
    
    def initialize_index(self):
        """Initialize Pinecone index if it doesn't exist."""
        index_name = "cs510_project"
        self.namespace = "cs510_project"
        
        try:
          
            indexes = self.pc.list_indexes()
            
           
            if index_name not in [index.name for index in indexes]:
               
                self.pc.create_index(
                    name=index_name,
                    dimension=1536,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
                logger.info(f"Created index {index_name}")
            
            
            self.index = self.pc.Index(index_name)
            logger.info(f"Connected to index {index_name}")
        except Exception as e:
            logger.error(f"Error initializing Pinecone index: {e}")
            
    

    
    def get_embedding(self, text):
        """Get embedding for text using OpenAI's API."""
        try:
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=text
            )
            output_response = response.data[0].embedding
            print(output_response)
            return output_response
        except Exception as e:
            logger.error(f"Error getting embedding: {e}")
            
            return [0] * 1536  # Return zero vector as fallback
    
    def store_document(self, file_name, content, email_subject, email_body):
        """Store document in Pinecone with its embedding."""
        try:
           
            doc_id = str(uuid.uuid4())
           
            embedding = self.get_embedding(content)
         
            metadata = {
                "file_name": file_name,
                "content": content[:8000],  
                "email_subject": email_subject,
                "email_body": email_body[:8000],  
                "created_at": datetime.now().isoformat()
            }
            
            self.index.upsert(
                vectors=[
                    {
                        "id": doc_id,
                        "values": embedding,
                        "metadata": metadata
                    }
                ],
                namespace=self.namespace
            )
            logger.info(f"Stored document {file_name} with ID {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Error storing document: {e}")
           
            return None
    
    def search_similar_documents(self, query_text, limit=5):
        """Search for similar documents in Pinecone."""
        try:
           
            query_embedding = self.get_embedding(query_text)
            results = self.index.query(
                namespace=self.namespace,
                vector=query_embedding,
                top_k=limit,
                include_metadata=True
            )
            
            similar_docs = []
            for match in results.matches:
                similar_docs.append({
                    'id': match.id,
                    'file_name': match.metadata.get('file_name', 'Unknown'),
                    'content': match.metadata.get('content', ''),
                    'email_subject': match.metadata.get('email_subject', ''),
                    'email_body': match.metadata.get('email_body', ''),
                    'similarity': match.score
                })
            
            return similar_docs
        except Exception as e:
            logger.error(f"Error searching similar documents: {e}")
           
            return []
        

if __name__ == "__main__":
    embedding_model = embeddingModel_openai()
    embedding_model.get_embedding("Hello, how are you?")

    
  