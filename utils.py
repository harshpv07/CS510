import os
import io
import base64
import PyPDF2
import logging
import uuid
import time
from datetime import datetime, timedelta
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


load_dotenv()

class embeddingModel_openai:
    def __init__(self):
        self.client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        self.pc = None  # Initialize Pinecone client
        self.index = None  # Initialize index to None
        self.namespace = "cs510-project"  # Define namespace here
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
            self.pc = None # Ensure pc is None if connection fails
    
    def initialize_index(self):
        """Initialize Pinecone index if it doesn't exist."""
        if not self.pc:
            logger.error("Pinecone client not initialized. Cannot initialize index.")
            return

        index_name = "cs510-project"
        # self.namespace = "cs510_project" # Moved to __init__
        
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
            self.index = None # Ensure index is None if initialization fails
    

    
    def get_embedding(self, text):
        """Get embedding for text using OpenAI's API."""
        if self.index is None:
            logger.error("Pinecone index is not initialized. Cannot get embedding.")
            return [0] * 1536  # Return zero vector as fallback

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
    
    def chunk_text(self, text, chunk_size=1000, chunk_overlap=200):
        """
        Split a long text into smaller chunks using recursive character text splitter.
        
        Args:
            text (str): The text to be chunked
            chunk_size (int): Maximum size of each chunk
            chunk_overlap (int): Number of characters to overlap between chunks
            
        Returns:
            list: List of text chunks
        """
        if self.index is None:
            logger.error("Pinecone index is not initialized. Cannot chunk text.")
            return [text]  # Return original text as a single chunk if chunking fails

        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", " ", ""]
            )
            
            chunks = text_splitter.split_text(text)
            logger.info(f"Split text into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Error chunking text: {e}")
            return [text]  # Return original text as a single chunk if chunking fails


    def store_document(self, user_prompt, face_description, base_emotions):
        """Store document in Pinecone with its embedding."""
        if self.index is None:
            logger.error("Pinecone index is not initialized. Cannot store document.")
            return None

        try:
            doc_id = str(uuid.uuid4())
            embedding = self.get_embedding(user_prompt)
         
            metadata = {  
                "content": user_prompt,
                "face_description": face_description,
                "base_emotions" : base_emotions,
                "created_at": datetime.now().timestamp()
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
            
            logger.info(f"Stored document {doc_id} with ID {doc_id}")
            return doc_id
        except Exception as e:
            logger.error(f"Error storing document: {e}")
           
            return None
    

    def chunk_and_embed_text(self, text, chunk_size=1000, chunk_overlap=200):
        """
        Chunk a text document and store each chunk in the vector database with embeddings.
        
        Args:
            text (str): The text to be chunked and embedded
            chunk_size (int): Maximum size of each chunk
            chunk_overlap (int): Number of characters to overlap between chunks
            
        Returns:
            list: List of document IDs for the stored chunks
        """
        if self.index is None:
            logger.error("Pinecone index is not initialized. Cannot chunk and embed text.")
            return []

        try:
            # Split the text into chunks
            chunks = self.chunk_text(text, chunk_size, chunk_overlap)
            
            # Store each chunk with its embedding
            doc_ids = []
            for i, chunk in enumerate(chunks):
                doc_id = str(uuid.uuid4())
                embedding = self.get_embedding(chunk)
                
                metadata = {
                    "content": chunk,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "created_at": datetime.now().timestamp()
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
                
                doc_ids.append(doc_id)
                logger.info(f"Stored chunk {i+1}/{len(chunks)} with ID {doc_id}")
            
            return doc_ids
        except Exception as e:
            logger.error(f"Error chunking and embedding text: {e}")
            return []

    def retrieve_recent_documents(self, hours_ago=2, limit=1000):
        """Retrieve documents created within the last specified hours."""
        if self.index is None:
            logger.error("Pinecone index is not initialized. Cannot retrieve documents.")
            return []

        try:
            # Calculate the timestamp for N hours ago
            time_threshold = datetime.now() - timedelta(hours=hours_ago)
            time_threshold_unix = time_threshold.timestamp()

            # Define the metadata filter
            metadata_filter = {
                "created_at": {"$gte": time_threshold_unix}
            }

            # Query Pinecone using a zero vector and the metadata filter
            # Pinecone query requires a vector, even if filtering is the main goal.
            # Use a zero vector and a high top_k to get candidates, then filter.
            zero_vector = [0.0] * 1536 # Assuming embedding dimension is 1536
            results = self.index.query(
                namespace=self.namespace,
                vector=zero_vector,
                filter=metadata_filter,
                top_k=limit,
                include_metadata=True
            )

            recent_docs = []
            for match in results.matches:
                # Optionally, double-check the timestamp client-side if needed,
                # though the server-side filter should handle it.
                recent_docs.append({
                    'id': match.id,
                    'metadata': match.metadata,
                    'score': match.score # Similarity score relative to the zero vector
                })

            logger.info(f"Retrieved {len(recent_docs)} documents created since {time_threshold.isoformat()}")
            return recent_docs
        except Exception as e:
            logger.error(f"Error retrieving recent documents: {e}")
            return []

    
    def search_similar_documents(self, query_text, limit=5):
        """Search for similar documents in Pinecone."""
        if self.index is None:
            logger.error("Pinecone index is not initialized. Cannot search for similar documents.")
            return []

        try:
            query_embedding = self.get_embedding(query_text)
            results = self.index.query(
                namespace=self.namespace,
                vector=query_embedding,
                top_k=limit
            )

            similar_docs = []
            for match in results.matches:
                similar_docs.append({
                    'id': match.id,
                    'score': match.score
                })

            logger.info(f"Found {len(similar_docs)} similar documents")
            return similar_docs
        except Exception as e:
            logger.error(f"Error searching for similar documents: {e}")
            return []


if __name__ == "__main__":
    embedding_model = embeddingModel_openai()
    embedding_model.get_embedding("Hello, how are you?")
    chunks = embedding_model.chunk_text("Hello, how are you?. This could be a long text and we need to chunk it into smaller chunks. Could you also chunking this text into a more word document ")
    embedding_model.store_document("how do i deal with stress? ",  "The face of the user is sad and confused.", "sad, confused")
    
    # Example usage for chunk_and_embed_text
    sample_text_to_chunk = "This is a longer document that needs to be broken down into several pieces. Each piece will be embedded and stored separately in the vector database for efficient retrieval later on. We are testing the chunk_and_embed_text functionality."
    doc_ids = embedding_model.chunk_and_embed_text(sample_text_to_chunk)
    if doc_ids:
        print(f"Successfully chunked and embedded document. Chunk IDs: {doc_ids}")
    else:
        print("Failed to chunk and embed document.")
  
    # Add a delay to allow for indexing
    print("\nWaiting for indexing...")
    time.sleep(5) 

    print("\nRetrieving documents from the last 2 hours:")
    recent_documents = embedding_model.retrieve_recent_documents(hours_ago=2)
    if recent_documents:
        print(f"Found {len(recent_documents)} recent documents:")
        for doc in recent_documents:
            created_at_ts = doc['metadata'].get('created_at')
            created_at_str = datetime.fromtimestamp(created_at_ts).isoformat() if created_at_ts else "N/A"
            print(f"  ID: {doc['id']}, Created At: {created_at_str}")
            print(f"  Content: {doc['metadata'].get('content', '[Content not found in metadata]')}")
    else:
        print("No recent documents found or an error occurred.")
  