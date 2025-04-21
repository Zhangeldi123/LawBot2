import os
import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional

# Telegram Bot
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, MessageHandler, CallbackQueryHandler, ContextTypes, filters

# Vector Database and Embeddings
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

# LLM Integration
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import uvicorn
from fastapi import FastAPI, Request

# Caching and Rate Limiting
import redis
import time
import hashlib
from functools import lru_cache

app = FastAPI()


@app.post("/telegram")
async def webhook(request: Request):
    try:
        json_data = await request.json()
        update = Update.de_json(json_data, bot=None)  # Бот будет инициализирован позже

        # Здесь должна быть обработка обновления
        logger.info(f"Received update: {update.update_id}")
        return {"status": "ok"}

    except Exception as e:
        logger.error(f"Webhook error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


async def get_application():
    """Инициализация приложения один раз"""
    if not hasattr(app, "tg_application"):
        app.tg_application = await main()
    return app.tg_application


# Logging configuration
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Environment variables (would be set in production)
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
REDIS_URL = os.environ.get("REDIS_URL")

# Initialize Redis client for caching
redis_client = redis.from_url(REDIS_URL)

# Constants
CACHE_TTL = 3600  # Cache TTL in seconds (1 hour)
MAX_TOKENS_PER_REQUEST = 4000  # Maximum tokens to send to LLM in one request
RATE_LIMIT_REQUESTS = 20  # Number of requests allowed per minute
RATE_LIMIT_WINDOW = 60  # Rate limit window in seconds (1 minute)
BATCH_SIZE = 5  # Number of documents to process in a batch

SUPPORTED_MIME_TYPES = {
    'text/plain': '.txt',
    'application/pdf': '.pdf',
    'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
    'text/csv': '.csv',
    'text/markdown': '.md',
    'application/json': '.json'
}

TEMP_DIR = "temp_downloads"
os.makedirs(TEMP_DIR, exist_ok=True)

@app.get("/health")
async def health_check():
    return {"status": "healthy"}


class DocumentProcessor:
    """Handles document processing, chunking, and embedding."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100,
            length_function=len,
        )
        # Use a smaller, faster model for embeddings to reduce costs
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vector_store = None

    async def load_documents(self) -> None:
        """Load documents from the data directory and create vector store."""
        logger.info("Loading documents and creating vector store...")

        documents = []
        # Process files in batches to handle large datasets
        for root, _, files in os.walk(self.data_dir):
            for i, file in enumerate(files):
                if i % 100 == 0:
                    logger.info(f"Processed {i} files so far...")

                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Create document with metadata
                    doc = Document(
                        page_content=content,
                        metadata={"source": file_path, "filename": file}
                    )
                    documents.append(doc)

                    # Process in batches to avoid memory issues
                    if len(documents) >= BATCH_SIZE:
                        await self._process_document_batch(documents)
                        documents = []

                except Exception as e:
                    logger.error(f"Error processing file {file_path}: {e}")

        # Process any remaining documents
        if documents:
            await self._process_document_batch(documents)

        logger.info("Document loading and vector store creation complete.")

    async def _process_document_batch(self, documents: List[Document]) -> None:
        """Process a batch of documents: chunk and add to vector store."""
        chunks = self.text_splitter.split_documents(documents)

        # Create or update vector store
        if self.vector_store is None:
            self.vector_store = FAISS.from_documents(chunks, self.embeddings)
        else:
            self.vector_store.add_documents(chunks)

    async def similarity_search(self, query: str, k: int = 5) -> List[Document]:
        """Perform similarity search on the vector store."""
        if self.vector_store is None:
            logger.error("Vector store not initialized. Call load_documents first.")
            return []

        # Generate cache key based on query and k
        cache_key = f"similarity_search:{hashlib.md5(f'{query}:{k}'.encode()).hexdigest()}"

        # Check cache first
        cached_result = redis_client.get(cache_key)
        if cached_result:
            logger.info("Using cached search results")
            return json.loads(cached_result)

        # Perform search
        try:
            results = self.vector_store.similarity_search(query, k=k)

            # Cache results
            serialized_results = json.dumps([
                {"content": doc.page_content, "metadata": doc.metadata}
                for doc in results
            ])
            redis_client.setex(cache_key, CACHE_TTL, serialized_results)

            return results
        except Exception as e:
            logger.error(f"Error in similarity search: {e}")
            return []


class RateLimiter:
    """Handles rate limiting for API calls."""

    def __init__(self, max_requests: int = RATE_LIMIT_REQUESTS, window: int = RATE_LIMIT_WINDOW):
        self.max_requests = max_requests
        self.window = window
        self.request_timestamps = []

    def can_make_request(self) -> bool:
        """Check if a request can be made without exceeding rate limits."""
        current_time = time.time()

        # Remove timestamps older than the window
        self.request_timestamps = [ts for ts in self.request_timestamps
                                   if current_time - ts < self.window]

        # Check if we're under the limit
        return len(self.request_timestamps) < self.max_requests

    def add_request(self) -> None:
        """Record a new request."""
        self.request_timestamps.append(time.time())


class RAGBot:
    """Main RAG bot class that handles Telegram integration and RAG logic."""

    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.rate_limiter = RateLimiter()

        # Initialize LLM with a smaller model to reduce costs
        self.llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model_name="gpt-3.5-turbo",  # Use a smaller model for cost efficiency
            temperature=0.7,
        )

        # Create prompt templates
        self.qa_prompt = PromptTemplate(
            input_variables=["question", "context"],
            template="""
            You are a helpful assistant that answers questions based on the provided context.

            Context:
            {context}

            Question: {question}

            Answer the question based only on the provided context. If the context doesn't contain 
            the information needed to answer the question, say "I don't have enough information to 
            answer that question." Be concise and accurate.
            """
        )

        self.summarization_prompt = PromptTemplate(
            input_variables=["documents"],
            template="""
            Summarize the following information concisely:

            {documents}

            Summary:
            """
        )

        # Create chains
        self.qa_chain = LLMChain(llm=self.llm, prompt=self.qa_prompt)
        self.summarization_chain = LLMChain(llm=self.llm, prompt=self.summarization_prompt)

    async def initialize(self) -> None:
        """Initialize the bot by loading documents."""
        await self.document_processor.load_documents()

    @lru_cache(maxsize=100)
    async def _get_context(self, query: str, max_docs: int = 5) -> str:
        """Get relevant context for a query from the vector store."""
        docs = await self.document_processor.similarity_search(query, k=max_docs)

        if not docs:
            return "No relevant information found."

        # Combine document contents
        context = "\n\n".join([f"Document: {doc.metadata.get('filename', 'Unknown')}\n{doc.page_content}"
                               for doc in docs])

        # If context is too large, summarize it
        if len(context) > MAX_TOKENS_PER_REQUEST * 4:  # Rough character estimate
            context = await self._summarize_documents(docs)

        return context

    async def _summarize_documents(self, docs: List[Document]) -> str:
        """Summarize a list of documents to reduce token count."""
        # Check rate limits
        if not self.rate_limiter.can_make_request():
            logger.warning("Rate limit exceeded for summarization")
            # Return truncated content instead of summary when rate limited
            return "\n\n".join([doc.page_content[:500] + "..." for doc in docs[:3]])

        # Record the API call
        self.rate_limiter.add_request()

        # Prepare documents for summarization
        docs_text = "\n\n".join([f"Document: {doc.metadata.get('filename', 'Unknown')}\n{doc.page_content}"
                                 for doc in docs])

        try:
            # Generate summary
            summary = await self.summarization_chain.arun(documents=docs_text)
            return summary
        except Exception as e:
            logger.error(f"Error in summarization: {e}")
            # Fallback to truncated content
            return "\n\n".join([doc.page_content[:500] + "..." for doc in docs[:3]])

    async def answer_question(self, query: str) -> str:
        """Generate an answer for a user query using RAG."""
        # Generate cache key
        cache_key = f"answer:{hashlib.md5(query.encode()).hexdigest()}"

        # Check cache first
        cached_answer = redis_client.get(cache_key)
        if cached_answer:
            logger.info("Using cached answer")
            return cached_answer.decode('utf-8')

        # Check rate limits
        if not self.rate_limiter.can_make_request():
            return "I'm currently processing too many requests. Please try again in a minute."

        # Get context for the query
        context = await self._get_context(query)

        # Record the API call
        self.rate_limiter.add_request()

        try:
            # Generate answer
            answer = await self.qa_chain.arun(question=query, context=context)

            # Cache the answer
            redis_client.setex(cache_key, CACHE_TTL, answer)

            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I encountered an error while processing your question. Please try again later."


# Telegram bot handlers
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /start is issued."""
    await update.message.reply_text(
        "👋 Welcome to the RAG Telegram Bot!\n\n"
        "I can answer questions based on a large knowledge base. "
        "Just send me your question, and I'll do my best to help you."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a message when the command /help is issued."""
    await update.message.reply_text(
        "Here's how to use this bot:\n\n"
        "1. Simply type your question and send it to me\n"
        "2. I'll search through my knowledge base and provide an answer\n"
        "3. For best results, be specific in your questions\n\n"
        "Commands:\n"
        "/start - Start the bot\n"
        "/help - Show this help message"
    )


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle user messages and generate responses."""
    if not update.message or not update.message.text:
        return

    query = update.message.text
    user_id = update.effective_user.id

    # Send typing indicator
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")

    # Get the RAG bot instance
    rag_bot = context.bot_data.get("rag_bot")
    if not rag_bot:
        await update.message.reply_text("The bot is still initializing. Please try again in a moment.")
        return

    # Generate answer
    answer = await rag_bot.answer_question(query)

    # Send response
    await update.message.reply_text(answer)


async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle errors in the telegram bot."""
    logger.error(f"Error: {context.error} caused by {update}")

    # Notify user of error
    if update and update.effective_chat:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text="Sorry, something went wrong. Please try again later."
        )


async def main() -> None:
    """Start the bot."""
    # Create the Application
    application = Application.builder().token(TELEGRAM_TOKEN).build()

    # Initialize RAG bot
    rag_bot = RAGBot()
    await rag_bot.initialize()

    # Store RAG bot in application context
    application.bot_data["rag_bot"] = rag_bot

    # Add handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Add error handler
    application.add_error_handler(error_handler)

    # Start the Bot
    await application.run_polling()


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
