from typing import Dict, Any, List
import os
import logging
import tempfile
import shutil
import subprocess
import sys
import time
from dotenv import load_dotenv
import pinecone
from pinecone import ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from pydantic import BaseModel
from PyPDF2 import PdfReader
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set custom temp directory
tempfile.tempdir = "C:\\Users\\mide\\Documents\\MarkMusk\\temp"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

from utils.config import get_pinecone_config, get_app_config

class BotService:
    def __init__(self, fastapi_url: str):
        self.fastapi_url = fastapi_url

        # Load configurations
        try:
            self.pinecone_config = get_pinecone_config()
            self.app_config = get_app_config()
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            raise

        # Apply Pinecone fallbacks
        self.pinecone_config["api_key"] = self.pinecone_config.get("api_key", os.getenv("PINECONE_API_KEY"))
        self.pinecone_config["PINECONE_INDEX"] = self.pinecone_config.get("PINECONE_INDEX", os.getenv("PINECONE_INDEX"))
        self.pinecone_config["namespace"] = self.pinecone_config.get("namespace", os.getenv("PINECONE_NAMESPACE", None))

        # Validate Pinecone config
        required_pinecone_keys = ["api_key", "PINECONE_INDEX"]
        logger.info(f"Pinecone config: {self.pinecone_config | {'api_key': '***'}}")
        for key in required_pinecone_keys:
            if not self.pinecone_config.get(key):
                logger.error(f"Missing Pinecone configuration key: {key}")
                raise ValueError(f"Missing required Pinecone configuration key: {key}")

        # Set Google API key
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

        # Initialize embeddings
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            logger.info(f"Using embedding model: text-embedding-004 (768 dimensions)")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini embeddings: {e}")
            raise

        # Initialize Pinecone
        try:
            self.pinecone_instance = pinecone.Pinecone(api_key=self.pinecone_config["api_key"])
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise

        # Create Pinecone index if needed
        index_name = self.pinecone_config["PINECONE_INDEX"]
        try:
            existing_indexes = [index["name"] for index in self.pinecone_instance.list_indexes()]
            if index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {index_name}")
                self.pinecone_instance.create_index(
                    name=index_name,
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
        except Exception as e:
            logger.error(f"Failed to create Pinecone index: {e}")
            raise

        # Initialize vector store
        try:
            self.vector_store = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=self.embeddings,
                namespace=self.pinecone_config.get("namespace", None)
            )
        except Exception as e:
            logger.error(f"Failed to initialize PineconeVectorStore: {e}")
            raise

        # Initialize LLM
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.7
            )
            logger.info(f"Using LLM: gemini-1.5-flash")
        except Exception as e:
            logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
            raise

        # Initialize memory
        self.memory = ConversationBufferMemory(
            input_key="question",
            memory_key="chat_history"
        )

        # Define prompt template with strict formatting rules
        self.qa_template = """You are Mark Musk, an AI assistant specialized in CreditChek API and SDK integration.
Provide detailed answers with these formatting rules:
1. Use **bold** ONLY for section headers (e.g., **Authentication** or **Troubleshooting**)
2. Never use bold for regular text
3. Use bullet points (-) for lists
4. Use code blocks (```) for examples
5. Separate sections with blank lines
6. Keep technical explanations clear and precise

Context: {context}

Question: {question}

Answer:"""
        self.qa_prompt = PromptTemplate(
            template=self.qa_template,
            input_variables=["context", "question"]
        )

        # Setup QA chain
        self.qa_chain = LLMChain(
            llm=self.llm,
            prompt=self.qa_prompt,
            memory=self.memory,
            output_key="answer",
            verbose=True
        )

        # Initialize document store
        self.init_document_store()

    def clear_index(self):
        """Clear all documents in the Pinecone index."""
        try:
            logger.info(f"Clearing Pinecone index {self.pinecone_config['PINECONE_INDEX']}")
            index = self.pinecone_instance.Index(self.pinecone_config["PINECONE_INDEX"])
            stats = index.describe_index_stats()
            logger.info(f"Index stats before clearing: {stats}")
            self.vector_store.delete(delete_all=True, namespace=self.pinecone_config.get("namespace", None))
            stats = index.describe_index_stats()
            logger.info(f"Index stats after clearing: {stats}")
        except Exception as e:
            logger.error(f"Failed to clear Pinecone index: {e}")
            raise

    def init_document_store(self):
        """Initialize document store with sample CreditChek docs if not already indexed."""
        try:
            logger.info("Checking Pinecone index for existing documents")
            index = self.pinecone_instance.Index(self.pinecone_config["PINECONE_INDEX"])
            stats = index.describe_index_stats()
            total_vectors = stats.get("total_vector_count", 0)
            logger.info(f"Index contains {total_vectors} vectors")
            
            if total_vectors > 0:
                logger.info("Index is not empty, skipping sample document initialization")
                return
            
            self.sample_docs = [
                Document(
                    page_content="**Authentication**\n\nTo authenticate with CreditChek API:\n- Obtain API key from dashboard\n- Include in headers: 'Authorization: Bearer YOUR_API_KEY'\n- All requests must use HTTPS",
                    metadata={"source": "sample_auth"}
                ),
                Document(
                    page_content="**Identity Verification**\n\nEndpoint: /api/v1/identity\nRequired parameters:\n- first_name\n- last_name\n- dob\n- id_number",
                    metadata={"source": "sample_identity"}
                ),
                Document(
                    page_content="**Webhooks**\n\nSetup process:\n1. Register URL in dashboard\n2. Handle POST requests\n3. Process JSON payload containing:\n- transaction_id\n- status\n- amount",
                    metadata={"source": "sample_webhooks"}
                )
            ]
            logger.info("Indexing sample documents")
            self._upload_documents_in_batches(self.sample_docs, batch_size=5)
        except Exception as e:
            logger.error(f"Failed to index sample documents: {e}")
            raise

    def _upload_documents_in_batches(self, documents: List[Document], batch_size: int = 50) -> None:
        """Upload documents to Pinecone in batches with parallel processing."""
        def embed_batch(batch: List[Document]) -> List[List[float]]:
            try:
                return self.embeddings.embed_documents([doc.page_content for doc in batch])
            except Exception as e:
                logger.error(f"Failed to embed batch: {e}")
                raise

        for doc in documents:
            content_size = len(doc.page_content.encode('utf-8')) / 1024
            if content_size > 500:
                logger.warning(f"Skipping large document: {doc.metadata.get('source', 'unknown')}")
                continue
        
        with ThreadPoolExecutor(max_workers=4) as executor:
            batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
            for i, batch in enumerate(batches):
                try:
                    embeddings = executor.submit(embed_batch, batch).result()
                    vectors = [
                        (f"doc_{i*batch_size+j}", emb, doc.metadata)
                        for j, (emb, doc) in enumerate(zip(embeddings, batch))
                    ]
                    index = self.pinecone_instance.Index(self.pinecone_config["PINECONE_INDEX"])
                    index.upsert(vectors=vectors, namespace=self.pinecone_config.get("namespace", None))
                    logger.info(f"Uploaded batch {i+1}/{len(batches)}")
                except Exception as e:
                    logger.error(f"Failed to upload batch {i+1}: {e}")
                    raise

    def process_github_repo(self, repo_url: str, branch: str = "master"):
        """Clone GitHub repo and index its documentation."""
        temp_dir = tempfile.mkdtemp()
        try:
            self.clear_index()
            logger.info(f"Cloning {repo_url} (branch: {branch})")
            subprocess.run(["git", "clone", "--branch", branch, repo_url, temp_dir], check=True)
            
            docs = []
            for root, _, files in os.walk(os.path.join(temp_dir, "docs")):
                for file in files:
                    if file.endswith((".md", ".mdx")):
                        file_path = os.path.join(root, file)
                        with open(file_path, "r", encoding="utf-8") as f:
                            docs.append(Document(
                                page_content=f.read(),
                                metadata={"source": file, "repo": repo_url}
                            ))
            
            if not docs:
                logger.warning("No Markdown files found in docs directory")
                return
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = splitter.split_documents(docs)
            self._upload_documents_in_batches(split_docs)
            logger.info(f"Indexed {len(split_docs)} document chunks")
            
        except Exception as e:
            logger.error(f"Failed to process GitHub repo: {e}")
            raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def process_uploaded_doc(self, uploaded_file):
        """Process and index uploaded PDF or text documents."""
        try:
            if uploaded_file.type == "application/pdf":
                text = "\n".join(page.extract_text() for page in PdfReader(uploaded_file).pages)
            elif uploaded_file.type == "text/plain":
                text = uploaded_file.read().decode("utf-8")
            else:
                raise ValueError("Unsupported file type")
            
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = splitter.split_text(text)
            docs = [Document(page_content=chunk, metadata={"source": uploaded_file.name}) for chunk in chunks]
            self._upload_documents_in_batches(docs)
            logger.info(f"Indexed {len(docs)} chunks from {uploaded_file.name}")
            
        except Exception as e:
            logger.error(f"Failed to process document: {e}")
            raise

    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate a properly formatted response with bold headers only."""
        query_lower = query.lower()

        # Handle greetings
        if any(greeting in query_lower for greeting in ["hello", "hi", "hey"]):
            return {
                "text": """**Welcome to CreditChek Assistant**\n\nHello! I'm Mark Musk, your dedicated CreditChek API and SDK integration expert.\n\n**How can I assist you today?**"""
            }

        # Handle example requests
        for lang in self.app_config["supported_languages"]:
            if f"show {lang.lower()} example" in query_lower:
                return {
                    "text": f"**{lang} Code Example**\n\nHere's a sample implementation:",
                    "code_snippets": {lang: f"# {lang} code example\n..."}
                }

        # Perform search and generate response
        try:
            index = self.pinecone_instance.Index(self.pinecone_config["PINECONE_INDEX"])
            query_embedding = self.embeddings.embed_query(query)
            results = index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True,
                namespace=self.pinecone_config.get("namespace", None)
            )
            context = "\n".join(
                match.metadata.get("content", "") 
                for match in results.matches
            ) if results.matches else "\n".join(doc.page_content for doc in self.sample_docs[:3])
            
            response = self.qa_chain.invoke({
                "context": context,
                "question": query
            })["answer"]
            
            # Ensure only headers are bold
            response = self._format_response(response)
            
            return {"text": response}
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "text": "**Error**\n\nI encountered an issue processing your request. Please try again."
            }

    def _format_response(self, text: str) -> str:
        """Ensure only section headers are bold."""
        lines = text.split('\n')
        formatted_lines = []
        
        for line in lines:
            stripped = line.strip()
            # Identify headers (lines ending with : or short title-like lines)
            if (stripped.endswith(':') or 
                (len(stripped.split()) <= 3 and 
                 stripped and 
                 stripped[0].isupper() and
                 not any(c in stripped for c in ['`', '*', '-']))):
                formatted_lines.append(f"**{stripped}**")
            else:
                formatted_lines.append(line)
        
        return '\n'.join(formatted_lines)