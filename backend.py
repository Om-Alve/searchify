from __future__ import annotations
import json
import os
import io
import logging
from typing import Any, Dict

from flask import Flask, request
from flask_cors import CORS
from dotenv import load_dotenv
from werkzeug.datastructures import FileStorage
from flask import Response, stream_with_context
from openai import OpenAI
from agent import run_agent

# ---------------------------------------------------------------------------
# Environment & logging ------------------------------------------------------
# ---------------------------------------------------------------------------
load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("semantic_search_api")

# Optional audio transcription client ---------------------------------------
AUDIO_BASE_URL = os.getenv("OPENAI_AUDIO_BASE_URL", "http://localhost:8000/v1/")
AUDIO_API_KEY = os.getenv("OPENAI_AUDIO_API_KEY", "cant-be-empty")

audio_client = OpenAI(api_key=AUDIO_API_KEY, base_url=AUDIO_BASE_URL)  # type: ignore[arg-type]
logger.info("OpenAI‑compatible audio backend initialised (%s)", AUDIO_BASE_URL)

# ---------------------------------------------------------------------------
# Flask app ------------------------------------------------------------------
# ---------------------------------------------------------------------------
app = Flask(__name__)
CORS(app)


@app.get("/health")
def health() -> tuple[str, int]:
    return "OK", 200


@app.post("/search")
def search() -> tuple[Dict[str, Any], int]:
    if not request.is_json:
        return {"error": "application/json required"}, 415
    query = (request.json or {}).get("query", "").strip()  # type: ignore[attr-defined]
    if not query:
        return {"error": "'query' must be non‑empty"}, 400
    logger.info("/search query: %s", query)
    answer = run_agent(query)
    return {"query": query, "answer": answer}, 200



@app.route("/search_from_audio_stream")
def search_from_audio_stream():
    def generate():
        file = request.files.get("audio")
        raw = file.read()

        # 1) Transcription start
        yield "event: progress\ndata: " + json.dumps({"step": "transcribing", "message": "Uploading & transcribing…"}) + "\n\n"
        transcript_rsp = audio_client.audio.transcriptions.create(
            model="Systran/faster-distil-whisper-large-v3",
            file=(file.filename, raw),
        )
        transcript = transcript_rsp.text.strip()

        # 2) Send the transcript as soon as it’s ready
        yield "event: transcript\ndata: " + json.dumps({"transcript": transcript}) + "\n\n"

        # 3) Agent processing start
        yield "event: progress\ndata: " + json.dumps({"step": "searching", "message": "Generating the answer…"}) + "\n\n"
        answer = run_agent(transcript)

        # 4) Final answer
        yield "event: answer\ndata: " + json.dumps({"answer": answer}) + "\n\n"

    return Response(
        stream_with_context(generate()),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    debug = bool(int(os.getenv("FLASK_DEBUG", "0")))
    logger.info("Starting Flask app on :%d (debug=%s)", port, debug)
    app.run("0.0.0.0", port, debug=debug)

