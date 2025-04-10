import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import streamlit as st
import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer, util
from llama_index.llms.groq import Groq
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv

# --- Configuration & Initialization ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

# Initialize Sentence Transformer model (load once)
@st.cache_resource(show_spinner=False)
def load_sentence_transformer():
    try:
        model_local = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Sentence Transformer model loaded successfully.")
        return model_local
    except Exception as e:
        logger.error(f"Error loading Sentence Transformer model: {e}", exc_info=True)
        raise

model = load_sentence_transformer()

# Initialize DuckDuckGo Search client
ddgs = DDGS()

# --- Core Functions ---

def search_ddg(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Performs a search using DuckDuckGo and returns results."""
    try:
        results = ddgs.text(query, max_results=max_results * 2)
        valid_results = [res for res in results if res.get('href')]
        logger.info(f"DDG Search for '{query}' returned {len(valid_results)} valid results.")
        return valid_results[:max_results]
    except Exception as e:
        logger.error(f"Error during DuckDuckGo search for '{query}': {e}", exc_info=True)
        return []

def extract_full_content(url: str, timeout: int = 10) -> str:
    """
    Fetches and extracts full text content from a given URL.
    Attempts to extract content from <article> tags first,
    then from <p> tags, and finally from large <div> blocks.
    Returns an empty string if fetching or parsing fails.
    """
    try:
        headers = {
            'User-Agent': ('Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                           'AppleWebKit/537.36 (KHTML, like Gecko) '
                           'Chrome/91.0.4472.124 Safari/537.36')
        }
        response = requests.get(url, headers=headers, timeout=timeout, allow_redirects=True)
        response.raise_for_status()

        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            logger.warning(f"Skipping non-HTML content at {url} (Content-Type: {content_type})")
            return ""

        soup = BeautifulSoup(response.text, 'html.parser')
        articles = soup.find_all('article')
        if articles:
            article_text = ' '.join(article.get_text(strip=True) for article in articles if article.get_text(strip=True))
            if article_text:
                return article_text

        paragraphs = soup.find_all('p')
        p_text = ' '.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        if p_text:
            return p_text

        divs = soup.find_all('div')
        candidate_texts = []
        for div in divs:
            div_text = div.get_text(strip=True)
            if len(div_text) > 200:
                candidate_texts.append(div_text)
        if candidate_texts:
            return max(candidate_texts, key=len)

        logger.warning(f"No substantial text found at {url}")
        return ""
    except requests.exceptions.RequestException as e:
        logger.warning(f"Error fetching {url}: {e}")
        return ""
    except Exception as e:
        logger.error(f"Error processing {url}: {e}", exc_info=True)
        return ""

def perform_semantic_search(
    query: str,
    max_search_results: int = 10,
    similarity_threshold: float = 0.4,
    max_workers: int = 8
) -> List[Dict[str, Any]]:
    """
    Performs a semantic web search:
    1. Searches DuckDuckGo for the query.
    2. Concurrently fetches content from result URLs.
    3. Calculates semantic similarity between query and fetched content.
    4. Returns relevant content snippets exceeding the similarity threshold.
    """
    search_results = search_ddg(query, max_results=max_search_results)
    if not search_results:
        return []

    query_embedding = model.encode(query, convert_to_tensor=True)
    relevant_contents = []
    futures_to_url = {}

    logger.info(f"Starting parallel fetch for {len(search_results)} URLs...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in search_results:
            url = res.get('href')
            if url:
                future = executor.submit(extract_full_content, url)
                futures_to_url[future] = url

        for future in as_completed(futures_to_url):
            url = futures_to_url[future]
            try:
                content = future.result()
                if content:
                    content_embedding = model.encode(content, convert_to_tensor=True)
                    similarity = util.cos_sim(query_embedding, content_embedding).item()
                    logger.debug(f"Similarity with {url}: {similarity:.3f}")

                    if similarity >= similarity_threshold:
                        relevant_contents.append({
                            "url": url,
                            "similarity": similarity,
                            "content": content
                        })
                        logger.info(f"Found relevant content (sim: {similarity:.3f}) at {url}")
            except Exception as e:
                logger.error(f"Error processing result for {url}: {e}", exc_info=True)

    relevant_contents.sort(key=lambda x: x["similarity"], reverse=True)
    logger.info(f"Semantic search found {len(relevant_contents)} relevant results for '{query}'.")
    return relevant_contents

# --- LlamaIndex Tool Definition ---

def semantic_web_search_tool_wrapper(query: str) -> List[Dict[str, Any]]:
    """
    LlamaIndex tool wrapper for the semantic web search function.
    Returns a list of dictionaries with URL, similarity, and a content snippet.
    """
    logger.info(f"Tool 'semantic_web_search' called with query: '{query}'")
    results = perform_semantic_search(query)
    tool_results = [
        {
            "url": r["url"],
            "similarity": round(r["similarity"], 3),
            "content_snippet": r["content"][:500] + ("..." if len(r["content"]) > 500 else "")
        }
        for r in results
    ]
    max_results_for_llm = 5
    return tool_results[:max_results_for_llm]

semantic_search_function_tool = FunctionTool.from_defaults(
    fn=semantic_web_search_tool_wrapper,
    name="semantic_web_search",
    description=(
        "Performs a semantic web search for a given query. "
        "It fetches web pages related to the query, compares their content semantically, "
        "and returns the most relevant URLs, their similarity score, and a snippet of their content."
    )
)

# --- Agent Setup and Execution ---

def run_agent(query: str) -> str:
    """Initializes and runs the ReActAgent with the semantic search tool."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
         logger.error("GROQ_API_KEY not found in environment variables.")
         return "Error: GROQ API key is missing."

    try:
        llm = Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
        agent = ReActAgent.from_tools(
            tools=[semantic_search_function_tool],
            llm=llm,
            verbose=True
        )
        logger.info(f"Starting agent chat with query: '{query}'")
        response = agent.chat(query)
        logger.info("Agent chat finished.")
        return response.response
    except Exception as e:
        logger.error(f"Error during agent execution: {e}", exc_info=True)
        return f"An error occurred while processing your request: {e}"

# --- Streamlit App Interface ---

st.title("Semantic Web Search Agent")
st.write("Enter a query to perform a semantic web search using the agent.")

with st.form("search_form"):
    query_input = st.text_input("Enter your query:", "")
    submitted = st.form_submit_button("Search")
    
if submitted and query_input:
    st.info("Processing your query. This may take a moment...")
    result = run_agent(query_input)
    st.write("### Agent Response")
    st.text_area("Response:", result, height=300)
elif submitted and not query_input:
    st.error("Please enter a valid query.")
