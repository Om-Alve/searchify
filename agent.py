import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup  # May be used for further HTML cleanup if desired
from duckduckgo_search import DDGS
from readability import Document
from sentence_transformers import SentenceTransformer, util
from llama_index.llms.groq import Groq
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv
from llama_index.tools.tavily_research.base import TavilyToolSpec

# --- Configuration & Initialization ---

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables (like GROQ_API_KEY)
load_dotenv()

# Initialize Sentence Transformer model (load once)
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence Transformer model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Sentence Transformer model: {e}", exc_info=True)
    raise

# Initialize DuckDuckGo Search client
ddgs = DDGS()

# --- Helper Function to Filter Paragraphs ---
def split_text_into_chunks(content, chunk_size):
    """Splits text into fixed-size chunks.

    Args:
        content (str): The input text.
        chunk_size (int): The desired size of each chunk (in characters).

    Returns:
        list: A list of text chunks.
    """
    chunks = [content[i:i + chunk_size] for i in range(0, len(content), chunk_size)]
    return chunks

def get_relevant_chunks(content, query_embedding, model, similarity_threshold=0.65, chunk_size=800):
    """Splits text into fixed-size chunks and identifies relevant ones based on semantic similarity.

    Args:
        content (str): The input text.
        query_embedding (torch.Tensor): The embedding of the search query.
        model (SentenceTransformer): The sentence embedding model.
        similarity_threshold (float): The minimum cosine similarity to consider a chunk relevant.
        chunk_size (int): The desired size of each chunk (in characters).

    Returns:
        list: A list of dictionaries, where each dictionary contains a relevant text chunk and its similarity score.
    """
    chunks = split_text_into_chunks(content, chunk_size)
    relevant_chunks = []

    for chunk in chunks:
        try:
            chunk_embedding = model.encode(chunk, convert_to_tensor=True)
            similarity = util.cos_sim(query_embedding, chunk_embedding).item()
            if similarity >= similarity_threshold:
                relevant_chunks.append({
                    "chunk": chunk,
                    "similarity": similarity
                })
        except Exception as e:
            logger.error(f"Error encoding or processing chunk: {e}", exc_info=True)
            continue

    return relevant_chunks[:3] 

# --- Core Functions ---

def search_ddg(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Performs a search using DuckDuckGo and returns results."""
    try:
        results = ddgs.text(query, max_results=max_results * 2)  # Fetch extra for filtering later
        valid_results = [res for res in results if res.get('href')]
        logger.info(f"DDG Search for '{query}' returned {len(valid_results)} valid results.")
        return valid_results[:max_results]
    except Exception as e:
        logger.error(f"Error during DuckDuckGo search for '{query}': {e}", exc_info=True)
        return []

def extract_content_jina(url: str) -> str:
    headers = {
        'Authorization': f'Bearer {os.getenv("JINA_API_KEY")}',
    }
    response = requests.get(f'https://r.jina.ai/{url}', headers=headers)
    return response.text

def perform_semantic_search(
    query: str,
    max_search_results: int = 2,
    page_similarity_threshold: float = 0.2,  # Lower threshold for the full page to allow further paragraph filtering
    paragraph_similarity_threshold: float = 0.4,
    max_workers: int = 8
) -> List[Dict[str, Any]]:
    """
    Performs a semantic web search:
    1. Searches DuckDuckGo for the query.
    2. Concurrently fetches content from result URLs.
    3. Splits content into paragraphs and calculates semantic similarity between each paragraph and the query.
    4. Returns the paragraphs (with their original URL) that exceed the paragraph similarity threshold.
    """
    search_results = search_ddg(query, max_results=max_search_results)
    if not search_results:
        return []

    query_embedding = model.encode(query, convert_to_tensor=True)
    relevant_results = []
    futures_to_url = {}

    logger.info(f"Starting parallel fetch for {len(search_results)} URLs...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for res in search_results:
            url = res.get('href')
            if url:
                future = executor.submit(extract_content_jina, url)
                futures_to_url[future] = url

        for future in as_completed(futures_to_url):
            url = futures_to_url[future]
            try:
                content = future.result()
                if content:
                    # First, optionally check if the overall page text has any relevance.
                    # For example, you can encode the whole content.
                    content_embedding = model.encode(content, convert_to_tensor=True)
                    page_similarity = util.cos_sim(query_embedding, content_embedding).item()
                    
                    # If the overall page similarity is low, you might skip detailed paragraph processing.
                    if page_similarity < page_similarity_threshold:
                        logger.info(f"Skipping detailed analysis for {url} due to low page similarity ({page_similarity:.3f}).")
                        continue
                    
                    # Now split the content into paragraphs and filter individually.
                    relevant_paragraphs = get_relevant_chunks(content, query_embedding, model, paragraph_similarity_threshold)
                    if relevant_paragraphs:
                        relevant_results.append({
                            "url": url,
                            "page_similarity": round(page_similarity, 3),
                            "relevant_paragraphs": relevant_paragraphs
                        })
                        logger.info(f"Found {len(relevant_paragraphs)} relevant paragraphs at {url}")

            except Exception as e:
                logger.error(f"Error processing result for {url}: {e}", exc_info=True)

    # Optionally, sort results by the highest similarity found among the paragraphs
    relevant_results.sort(key=lambda x: max(p["similarity"] for p in x["relevant_paragraphs"]), reverse=True)
    logger.info(f"Semantic search found {len(relevant_results)} relevant results for '{query}'.")
    return relevant_results

# --- LlamaIndex Tool Definition ---

def semantic_web_search_tool_wrapper(query: str) -> List[Dict[str, Any]]:
    """
    LlamaIndex tool wrapper for the semantic web search function.
    Returns a list of dictionaries with URL, overall page similarity, and relevant content paragraphs.
    """
    logger.info(f"Tool 'semantic_web_search' called with query: '{query}'")
    results = perform_semantic_search(query)

    # Prepare results for the agent
    tool_results = [
        {
            "url": r["url"],
            "page_similarity": r["page_similarity"],
            "relevant_paragraphs": [
                {
                    "similarity": round(p["similarity"], 3),
                    "paragraph_snippet": p["chunk"][:500]
                }
                for p in r["relevant_paragraphs"]
            ]
        }
        for r in results
    ]
    # Limit the number of results returned to avoid overwhelming the agent
    max_results_for_llm = 2 
    return tool_results[:max_results_for_llm]

semantic_search_function_tool = FunctionTool.from_defaults(
    fn=semantic_web_search_tool_wrapper,
    name="semantic_web_search",
    description=(
        "Performs a semantic web search for a given query. "
        "It fetches web pages related to the query, splits them into paragraphs, "
        "compares their content semantically to the query, and returns the most relevant paragraphs with their source URLs."
    )
)

# --- Agent Setup and Execution ---

def run_agent(query: str):
    """Initializes and runs the ReActAgent with the semantic search tool."""
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        logger.error("GROQ_API_KEY not found in environment variables.")
        return "Error: GROQ API key is missing."

    try:
        llm = Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
        tools = TavilyToolSpec(
            api_key='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',
        ).to_tool_list()
        tools.append(semantic_search_function_tool)
        agent = ReActAgent.from_tools(
            system_prompt="Only use tools when you need them. Use the semantic_search tool for finding relevant content, if it doesn't give results on the first try then use the Tavily Tool",
            tools=tools,
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

# --- Main Execution Block ---

if __name__ == "__main__":
    test_query = "Where does Sonal Solaskar work?"
    final_answer = run_agent(test_query)
    print("\n" + "="*20 + " Final Answer " + "="*20)
    print(final_answer)
    print("="*54)

