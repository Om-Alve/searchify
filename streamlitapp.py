import os
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any

import requests
from bs4 import BeautifulSoup
from duckduckgo_search import DDGS
from sentence_transformers import SentenceTransformer, util
from llama_index.llms.groq import Groq
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import ReActAgent
from llama_index.tools.tavily_research.base import TavilyToolSpec
from dotenv import load_dotenv

import streamlit as st
import pandas as pd
import torch
import time
import soundfile as sf
from io import BytesIO
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from st_audiorec import st_audiorec

# --- Configuration & Initialization ---
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize SentenceTransformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence Transformer model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading Sentence Transformer model: {e}", exc_info=True)
    raise

# Initialize DuckDuckGo Search client
ddgs = DDGS()

# --- Helper Functions ---

def filter_content_paragraphs(content: str, query_embedding, similarity_threshold: float = 0.4) -> List[Dict[str, Any]]:
    paragraphs = [p.strip() for p in content.split('\n') if p.strip()]
    relevant = []
    for p in paragraphs:
        try:
            emb = model.encode(p, convert_to_tensor=True)
            sim = util.cos_sim(query_embedding, emb).item()
            if sim >= similarity_threshold:
                relevant.append({"paragraph": p, "similarity": sim})
        except Exception as e:
            logger.error(f"Error processing paragraph: {e}", exc_info=True)
    return relevant


def search_ddg(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    try:
        results = ddgs.text(query, max_results=max_results * 2)
        valid = [r for r in results if r.get('href')]
        logger.info(f"DDG Search for '{query}' returned {len(valid)} valid results.")
        return valid[:max_results]
    except Exception as e:
        logger.error(f"Error during DDG search: {e}", exc_info=True)
        return []


def extract_content_jina(url: str) -> str:
    headers = {'Authorization': f"Bearer {os.getenv('JINA_API_KEY')}"}
    resp = requests.get(url, headers=headers)
    if resp.status_code != 200:
        logger.warning(f"Failed to fetch {url}: {resp.status_code}")
        return ""
    soup = BeautifulSoup(resp.text, 'html.parser')
    texts = soup.find_all(['p', 'div'])
    return "\n".join(t.get_text() for t in texts)


def perform_semantic_search(
    query: str,
    max_search_results: int = 10,
    page_similarity_threshold: float = 0.2,
    paragraph_similarity_threshold: float = 0.4,
    max_workers: int = 8
) -> List[Dict[str, Any]]:
    search_results = search_ddg(query, max_results=max_search_results)
    if not search_results:
        return []

    query_emb = model.encode(query, convert_to_tensor=True)
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(extract_content_jina, r['href']): r['href'] for r in search_results}
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                content = future.result()
                if not content:
                    continue
                page_emb = model.encode(content, convert_to_tensor=True)
                page_sim = util.cos_sim(query_emb, page_emb).item()
                if page_sim < page_similarity_threshold:
                    logger.info(f"Skipping {url}, low page similarity {page_sim:.3f}")
                    continue
                paras = filter_content_paragraphs(content, query_emb, paragraph_similarity_threshold)
                if paras:
                    results.append({"url": url, "relevant_paragraphs": paras})
                    logger.info(f"Found {len(paras)} relevant paragraphs at {url}")
            except Exception as e:
                logger.error(f"Error processing {url}: {e}", exc_info=True)
    results.sort(key=lambda r: max(p['similarity'] for p in r['relevant_paragraphs']), reverse=True)
    return results


def semantic_web_search_tool_wrapper(
    query: str,
    max_results: int = 5,
    threshold: float = 0.4
) -> List[Dict[str, Any]]:
    logger.info(f"Tool 'semantic_web_search' called with query: '{query}', max_results={max_results}, threshold={threshold}")
    raw = perform_semantic_search(
        query,
        max_search_results=max_results,
        paragraph_similarity_threshold=threshold
    )
    tool_results = []
    for r in raw:
        best = max(r['relevant_paragraphs'], key=lambda p: p['similarity'])
        snippet = best['paragraph']
        sim = round(best['similarity'], 3)
        tool_results.append({"url": r['url'], "similarity": sim, "content_snippet": snippet})
    return tool_results[:max_results]


semantic_search_function_tool = FunctionTool.from_defaults(
    fn=semantic_web_search_tool_wrapper,
    name="semantic_web_search",
    description=(
        "Performs a semantic web search for a given query. "
        "Fetches pages, ranks paragraphs by semantic similarity, and returns top snippets."
    )
)


def run_agent(query: str) -> str:
    groq_api_key = os.getenv('GROQ_API_KEY')
    if not groq_api_key:
        logger.error("GROQ_API_KEY missing")
        return "Error: GROQ API key missing."
    try:
        llm = Groq(model="llama-3.3-70b-versatile", api_key=groq_api_key)
        tools = TavilyToolSpec(api_key=os.getenv('TAVILY_API_KEY')).to_tool_list()
        tools.append(semantic_search_function_tool)
        agent = ReActAgent.from_tools(
            system_prompt="Use the semantic_search tool for finding relevant content.",
            tools=tools,
            llm=llm,
            verbose=True
        )
        logger.info(f"Agent query: {query}")
        res = agent.chat(query)
        return res.response
    except Exception as e:
        logger.error(f"Agent error: {e}", exc_info=True)
        return f"Error: {e}"

# --- Streamlit App ---

device = "cuda:0" if torch.cuda.is_available() else "cpu"
 torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
model_id = "openai/whisper-large-v3-turbo"

asr_model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    low_cpu_mem_usage=True,
    use_safetensors=True
).to(device)
asr_processor = AutoProcessor.from_pretrained(model_id)
asr_pipe = pipeline(
    "automatic-speech-recognition",
    model=asr_model,
    tokenizer=asr_processor.tokenizer,
    feature_extractor=asr_processor.feature_extractor,
    device=device,
    torch_dtype=torch_dtype,
)

if not os.getenv('GROQ_API_KEY'):
    st.error("⚠️ GROQ_API_KEY not found in .env")

st.set_page_config(page_title="Semantic Web Search 🔍", layout="wide")
st.title("🔍 Semantic Web Search")
st.markdown(
    """
This app performs semantic web searches by:
1. Searching the web  
2. Analyzing content  
3. Returning the most relevant info  
You can talk to it or type your query!
"""
)

# --- Audio Input Section ---
st.subheader("🎙️ Record Your Query")
wav_audio = st_audiorec()
query = None
if wav_audio is not None:
    st.audio(wav_audio, format='audio/wav')
    with st.spinner("Transcribing…"):
        data, sr = sf.read(BytesIO(wav_audio))
        asr_result = asr_pipe({"array": data, "sampling_rate": sr})
    transcript = asr_result["text"].strip()
    st.success("Transcription:")
    st.write(f"> {transcript}")
    query = transcript
else:
    st.subheader("🎤 Upload Audio or Type Query")
    audio_file = st.file_uploader(
        "Upload an audio clip (wav/mp3) or record externally", 
        type=["wav", "mp3", "m4a"],
        key="uploader"
    )
    if audio_file:
        with st.spinner("Transcribing…"):
            data, sr = sf.read(BytesIO(audio_file.read()))
            asr_result = asr_pipe({"array": data, "sampling_rate": sr})
        transcript = asr_result["text"].strip()
        st.success("Transcription:")
        st.write(f"> {transcript}")
        query = transcript
    else:
        query = st.text_input("Or type your search query", key="text_input")

# --- Settings & Search ---
with st.sidebar:
    st.header("⚙️ Settings")
    max_results = st.slider("Max Search Results", 3, 20, 10)
    sim_thresh = st.slider("Similarity Threshold", 0.1, 0.9, 0.4, 0.05)
    mode = st.radio(
        "Search Mode",
        ["Direct Semantic Search", "AI-Assisted Search (LLM)"]
    )
    st.markdown("---")
    st.markdown("Built with Streamlit + Whisper + Groq LLM")

if query and st.button("🔎 Search"):
    with st.spinner("Searching the web…"):
        start = time.time()
        try:
            if mode == "Direct Semantic Search":
                results = semantic_web_search_tool_wrapper(
                    query, max_results=max_results, threshold=sim_thresh
                )
                duration = time.time() - start
                st.success(f"Found {len(results)} results in {duration:.2f}s")
                for i, r in enumerate(results, 1):
                    with st.expander(f"{i}. {r['url']} (Sim: {r['similarity']})"):
                        st.markdown(r['content_snippet'])
                df = pd.DataFrame(results)
                df = df.rename(columns={"url":"URL","similarity":"Similarity","content_snippet":"Snippet"})
                st.dataframe(df[["URL","Similarity","Snippet"]], use_container_width=True)
            else:
                ai_resp = run_agent(query)
                duration = time.time() - start
                st.success(f"AI search done in {duration:.2f}s")
                st.subheader("AI Analysis")
                st.markdown(
                    f"<div style='padding:20px;border-radius:8px;background:#f0f2f6;'>{ai_resp}</div>",
                    unsafe_allow_html=True,
                )
                st.info("Verify with original sources!")
        except Exception as e:
            logger.error("Search error", exc_info=True)
            st.error(f"Error: {e}")

# --- Sample Queries & Footer ---
st.markdown("---")
st.subheader("Sample Queries")
for sample in [
    "Latest advances in quantum computing",
    "What is semantic search?",
    "Environmental impact of EVs"
]:
    if st.button(sample):
        st.session
