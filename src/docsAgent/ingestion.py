import logging
import os
import ssl

import certifi
from dotenv import load_dotenv
import certifi
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_tavily import TavilyCrawl, TavilyExtract, TavilyMap

load_dotenv()


ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small",
    show_progress_bar=False,
    chunk_size=50,
    retry_min_seconds=10,
)
# vectorstore = Chroma(persist_directory="chroma_db", embedding_function=embeddings)
vectorstore = PineconeVectorStore(
    index_name="langchain-doc-agent", embedding=embeddings
)
tavily_extract = TavilyExtract()
tavily_map = TavilyMap(max_depth=5, max_breadth=20, max_pages=1000)
tavily_crawl = TavilyCrawl()


def main():
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting the Tavily Crawl and Map process...")

    res = tavily_crawl.invoke(
        {
            "url": "https://python.langchain.com/",
            "max_depth": 1,
            "extract_depth": "advanced",
        }
    )
    logging.info("Crawl and Map process completed.")
    allDocs = [
        Document(
            page_content=tavily_crawl_result_item["raw_content"],
            metadata={"source": tavily_crawl_result_item["url"]},
        )
        for tavily_crawl_result_item in res["result"]
    ]
    logging.info("Documents extracted and converted to Document objects.")


if __name__ == "__main__":
    main()
