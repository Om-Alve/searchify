import asyncio
import httpx  # Import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from duckduckgo_search import DDGS
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from llama_index.llms.groq import Groq
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from dotenv import load_dotenv
import os  # Import os for environment variables

load_dotenv()

# Ensure GROQ_API_KEY is loaded
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in environment variables. Please set it in your .env file.")

# --- Model Initialization ---
# Consider loading the model outside functions if it's used frequently
# (already done correctly)
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Async Search Functions ---

async def search_query_async(query: str, max_results: int = 5):
    """Asynchronously search DuckDuckGo."""
    # Use DDGS context manager for potential resource cleanup
    async with DDGS() as ddgs:
        results = [r async for r in ddgs.atext(query, max_results=max_results)]
        return results

async def extract_full_content_async(url: str, client: httpx.AsyncClient):
    """Asynchronously extract content from a URL using httpx."""
    try:
        # Use a shared httpx client for efficiency
        response = await client.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)
        soup = BeautifulSoup(response.text, 'html.parser')
        paragraphs = soup.find_all('p')
        full_text = ' '.join([p.get_text(strip=True) for p in paragraphs]) # Added strip=True
        # Basic filtering for empty or very short content
        if len(full_text) < 100:
             print(f"Skipping {url}: Content too short or mostly non-paragraph text.")
             return ""
        return full_text
    except httpx.RequestError as e:
        print(f"HTTP error fetching {url}: {e}")
        return ""
    except Exception as e:
        print(f"Error processing {url}: {e}")
        return ""

async def search_async(query: str):
    """
    Perform an asynchronous semantic web search and return relevant webpage contents.
    """
    print(f"Starting async search for: {query}")
    # Use the async search function
    results = await search_query_async(query, max_results=7) # Increased potential results slightly
    print(f"Got {len(results)} initial search results.")

    threshold = 0.4 # Slightly lower threshold might be needed if scraping is imperfect
    query_embedding = model.encode(query, convert_to_tensor=True)
    relevant_contents = []

    # Create a single httpx client for all requests in this search
    async with httpx.AsyncClient(follow_redirects=True) as client:
        tasks = []
        url_map = {} # To map task results back to URLs

        for i, res in enumerate(results):
            url = res.get('href')
            if not url:
                continue
            # Create an async task for each extraction
            task = asyncio.create_task(extract_full_content_async(url, client))
            tasks.append(task)
            url_map[i] = url # Store URL by original index

        # Gather results from all extraction tasks concurrently
        extracted_contents = await asyncio.gather(*tasks, return_exceptions=True)
        print(f"Finished fetching content for {len(extracted_contents)} URLs.")

        # Process results after gathering
        for i, content_or_exc in enumerate(extracted_contents):
            url = url_map.get(i)
            if url is None: continue # Should not happen

            if isinstance(content_or_exc, Exception):
                print(f"Task for {url} failed: {content_or_exc}")
                continue
            
            content = content_or_exc
            if not content: # Skip if content extraction returned empty string
                print(f"No usable content extracted from {url}")
                continue

            try:
                # Run potentially CPU-bound embedding in thread to avoid blocking event loop
                # (Though SentenceTransformer might release GIL, this is safer)
                content_embedding = await asyncio.to_thread(model.encode, content, convert_to_tensor=True)
                similarity = util.cos_sim(query_embedding, content_embedding).item()
                print(f"Similarity with {url}: {similarity:.3f}")
                if similarity >= threshold:
                    relevant_contents.append({
                        "url": url,
                        "similarity": similarity,
                        "content": content
                    })
            except Exception as e:
                 print(f"Error during embedding/similarity calculation for {url}: {e}")

    # Sort by similarity descending and take top N (e.g., top 3)
    relevant_contents.sort(key=lambda x: x['similarity'], reverse=True)
    print(f"Found {len(relevant_contents)} relevant contents above threshold {threshold}.")
    return relevant_contents[:3] # Return only the top 3 most relevant results

async def semantic_web_search_tool_async(query: str):
    """
    Async LlamaIndex tool: Performs semantic web search using async functions.
    """
    print(f"Tool called with query: {query}")
    # Call the main async search logic
    results = await search_async(query)
    print(f"Tool returning {len(results)} results.")
    # Return snippet for brevity in agent context
    return [
        {
            "url": r["url"],
            "similarity": round(r["similarity"], 3),
            "content_snippet": r["content"][:1000] + "..." if len(r["content"]) > 1000 else r["content"] # Longer snippet
        }
        for r in results
    ]

# --- FastAPI App Setup ---
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- LlamaIndex Setup ---

# Create the FunctionTool from the ASYNC function
# LlamaIndex usually detects async functions automatically
semantic_search_function_tool = FunctionTool.from_defaults(
    fn=semantic_web_search_tool_async, # Use the async version
    name="semantic_web_search",
    description="Performs an up-to-date semantic web search for a given query and returns the most relevant web page content snippets and their URLs. Use this for recent events or information not likely in the LLM's training data."
)

# Initialize LLM
# Make sure GROQ_API_KEY is set in your environment or .env file
llm = Groq(model="llama-3.1-70b-versatile") # Use updated model name if needed

# Initialize Agent - ReActAgent generally works well with async tools
agent = ReActAgent.from_tools(
    tools=[semantic_search_function_tool],
    llm=llm,
    verbose=True,
    # Consider adding max_iterations to prevent infinite loops
    # max_iterations=10
)

# --- API Endpoints ---

# Request model
class QueryRequest(BaseModel):
    query: str

@app.post("/chat")
async def chat(request: QueryRequest):
    """
    Handles chat requests, invoking the ReActAgent asynchronously.
    """
    try:
        print(f"Received chat request: {request.query}")
        # Use agent.achat() for native async execution if available and preferred
        # If agent.chat handles async tools correctly internally (common),
        # wrapping it in to_thread is still a valid fallback, but achat is cleaner.
        # Let's try agent.achat() first
        if hasattr(agent, 'achat'):
             print("Using agent.achat()")
             response = await agent.achat(request.query)
        else:
             # Fallback if achat is not available or if you prefer this pattern
             # Note: This runs the *entire* agent reasoning loop in a thread.
             # The I/O within the tool is async, making the thread more efficient.
             print("Using asyncio.to_thread(agent.chat)")
             response = await asyncio.to_thread(agent.chat, request.query)

        print(f"Agent response: {response}")
        # Ensure the response is serializable (LlamaIndex response objects often are)
        # If response is a complex object, extract the string part: response.response
        response_content = getattr(response, 'response', str(response))
        return {"response": response_content}
    except Exception as e:
        print(f"Error during chat processing: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

@app.get("/")
def read_root():
    return {"message": "Groq-powered agent with async search is running!"}

# --- Main execution (for running with uvicorn directly) ---
if __name__ == "__main__":
    import uvicorn
    # It's better to run via `uvicorn main:app --reload` from the terminal
    # but this allows running `python your_script_name.py`
    uvicorn.run(app, host="0.0.0.0", port=8000)
