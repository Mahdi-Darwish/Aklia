
"""
FastAPI backend wrapping the existing four-agent LangGraph pipeline.

Local dev:
    pip install fastapi uvicorn
    uvicorn api.analyze:app --reload --port 8000

Deploying on Vercel:
    The FastAPI app is exposed through /api/analyze, while the root
    route serves the frontend index.html.
"""
import os
import sys
from pathlib import Path

# Make the project root importable.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from ai.graph import build_graph
from ai.state import FoodState

app = FastAPI()
# CORS
ALLOWED_ORIGINS = [
    "https://aklia-v64u-seven.vercel.app",
    "http://localhost:3000",
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)


class AnalyzeRequest(BaseModel):
    goal: str = ""
    meals: str


# Frontend
@app.get("/")
def home():
    """Serve the frontend when visiting the Vercel root URL."""
    return FileResponse(PROJECT_ROOT / "index.html")

# API
@app.post("/api/analyze")
def analyze(req: AnalyzeRequest):
    try:
        print("=== ANALYZE START ===")
        print("GROQ_API_KEY exists:", bool(os.environ.get("GROQ_API_KEY")))
        print("USDA_API_KEY exists:", bool(os.environ.get("USDA_API_KEY")))

        graph_app = build_graph()

        initial_state: FoodState = {
            "goal": req.goal,
            "meals": req.meals,
            "parser_output": [],
            "analyzer_output": {},
            "goal_analysis_content": "",
            "report_content": "",
            "retry_count": 0,
            "parser_retry_count": 0,
            "error": None,
        }

        print("Invoking graph...")

        final_state = graph_app.invoke(initial_state)

        print("=== GRAPH FINISHED ===")
        print("Final state:", final_state)

        analyzer_output = final_state.get("analyzer_output") or {}

        if not analyzer_output:
            return {
                "ok": False,
                "message": (
                    final_state.get("error")
                    or final_state.get("report_content")
                    or "Graph completed but analyzer_output is empty."
                ),
                "debug": {
                    "parser_output": final_state.get("parser_output"),
                    "analyzer_output": final_state.get("analyzer_output"),
                    "goal_analysis": final_state.get("goal_analysis_content"),
                },
            }

        return {
            "ok": True,
            "items": analyzer_output.get("items", []),
            "totals": analyzer_output.get("totals", {}),
            "macro_percentages": analyzer_output.get(
                "macro_percentages",
                {},
            ),
            "goal_analysis": final_state.get(
                "goal_analysis_content"
            ) or None,
        }

    except Exception as e:
        import traceback

        traceback.print_exc()

        return {
            "ok": False,
            "message": f"{type(e).__name__}: {str(e)}",
        }


