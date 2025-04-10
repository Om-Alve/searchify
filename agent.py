import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup
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
    # Depending on the use case, you might want to exit or handle this differently
    raise  # Re-raise the exception if the model is critical

# Initialize DuckDuckGo Search client
ddgs = DDGS()

# --- Core Functions ---

def search_ddg(query: str, max_results: int = 10) -> List[Dict[str, Any]]:
    """Performs a search using DuckDuckGo and returns results."""
    try:
        # Note: ddgs.text doesn't have an explicit max_results in its direct signature easily,
        # but fetching more than ~20-30 is often unreliable/limited by the underlying API.
        # We limit how many we *process* later. Let's fetch a reasonable amount.
        results = ddgs.text(query, max_results=max_results * 2) # Fetch a bit more just in case some fail
        # Filter out results without 'href' early
        valid_results = [res for res in results if res.get('href')]
        logger.info(f"DDG Search for '{query}' returned {len(valid_results)} valid results.")
        return valid_results[:max_results] # Limit to the desired number for processing
    except Exception as e:
        logger.error(f"Error during DuckDuckGo search for '{query}': {e}", exc_info=True)
        return []

def extract_full_content(url: str, timeout: int = 10) -> str:
    """
    Fetches and extracts full text content from a given URL.
    Tries several fallback strategies:
    1. <article>
    2. <main>
    3. <p>
    4. Large <div>
    5. Meta description
    6. Readability
    7. Fallback to body text
    Returns an empty string if all fail.
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

        # Strategy 1: Article tags
        articles = soup.find_all('article')
        article_text = ' '.join(article.get_text(strip=True) for article in articles if article.get_text(strip=True))
        if article_text:
            return article_text

        # Strategy 2: Main tags
        mains = soup.find_all('main')
        main_text = ' '.join(main.get_text(strip=True) for main in mains if main.get_text(strip=True))
        if main_text:
            return main_text

        # Strategy 3: Paragraph tags
        paragraphs = soup.find_all('p')
        p_text = ' '.join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))
        if p_text:
            return p_text

        # Strategy 4: Large divs
        divs = soup.find_all('div')
        candidate_texts = [div.get_text(strip=True) for div in divs if len(div.get_text(strip=True)) > 200]
        if candidate_texts:
            return max(candidate_texts, key=len)

        # Strategy 5: Meta description
        meta_desc = soup.find('meta', attrs={'name': 'description'}) or soup.find('meta', attrs={'property': 'og:description'})
        if meta_desc and meta_desc.get('content'):
            return meta_desc['content'].strip()

        # Strategy 6: Readability (extracts the most relevant content block)
        try:
            doc = Document(response.text)
            readable_html = doc.summary()
            readable_soup = BeautifulSoup(readable_html, 'html.parser')
            readable_text = readable_soup.get_text(strip=True)
            if readable_text:
                return readable_text
        except Exception as e:
            logger.info(f"Readability parsing failed for {url}: {e}")

        # Strategy 7: Full body fallback
        body = soup.body
        if body:
            body_text = body.get_text(separator=' ', strip=True)
            if len(body_text) > 100:
                return body_text

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
    similarity_threshold: float = 0.4, # Slightly lower threshold might be better sometimes
    max_workers: int = 8 # Adjust based on your system and network
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
    # Use ThreadPoolExecutor for concurrent network I/O
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit fetch tasks
        for res in search_results:
            url = res.get('href')
            if url:
                # Submit the function call to the executor
                future = executor.submit(extract_full_content, url)
                futures_to_url[future] = url # Map future back to URL

        # Process completed futures as they finish
        for future in as_completed(futures_to_url):
            url = futures_to_url[future]
            try:
                content = future.result() # Get the result from the completed future
                if content:
                    # Calculate similarity only if content was successfully fetched
                    content_embedding = model.encode(content, convert_to_tensor=True)
                    similarity = util.cos_sim(query_embedding, content_embedding).item()
                    logger.debug(f"Similarity with {url}: {similarity:.3f}")

                    if similarity >= similarity_threshold:
                        relevant_contents.append({
                            "url": url,
                            "similarity": similarity,
                            "content": content # Store full content temporarily
                        })
                        logger.info(f"Found relevant content (sim: {similarity:.3f}) at {url}")

            except Exception as e:
                logger.error(f"Error processing result for {url}: {e}", exc_info=True)

    # Sort by similarity (highest first) before returning
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
    results = perform_semantic_search(query) # Use the optimized function

    # Prepare results for the agent (snippets)
    tool_results = [
        {
            "url": r["url"],
            "similarity": round(r["similarity"], 3),
            "content_snippet": r["content"]
        }
        for r in results
    ]
    # Limit the number of results returned to the LLM to avoid overwhelming it
    max_results_for_llm = 5
    return tool_results[:max_results_for_llm]


# Register as a FunctionTool
semantic_search_function_tool = FunctionTool.from_defaults(
    fn=semantic_web_search_tool_wrapper, # Use the wrapper
    name="semantic_web_search",
    description=(
        "Performs a semantic web search for a given query. "
        "It fetches web pages related to the query, compares their content semantically, "
        "and returns the most relevant URLs, their similarity score, and a snippet of their content."
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
        llm = Groq(model="meta-llama/llama-4-maverick-17b-128e-instruct", api_key=groq_api_key) # or llama3-70b-8192
        tools = TavilyToolSpec(
            api_key='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',
        ).to_tool_list()
        tools.append(semantic_search_function_tool) # Add the semantic search tool
        agent = ReActAgent.from_tools(
            system_prompt="Use the semantic_search tool for finding relevant content, if it doesn't give results on the first try then use the Tavily Tool",
            tools=tools,
            llm=llm,
            verbose=True # Set to False for cleaner production output
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
    test_query = "Where does Sonal Solaskar from Vidyalankar work currently?"
    final_answer = run_agent(test_query)
    print("\n" + "="*20 + " Final Answer " + "="*20)
    print(final_answer)
    print("="*54)
