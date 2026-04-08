"""
AMPIDE FastAPI Server
Exposes the OpenEnv-compliant REST API for Hugging Face Spaces.

Endpoints:
  POST /reset          — Reset environment (optionally specify task)
  POST /step           — Take one action
  GET  /state          — Get full hidden state (for grading)
  GET  /grade          — Get deterministic grade for current episode
  GET  /tasks          — List available tasks
  GET  /health         — Health check
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ampide_env import AMPIDEAction, AMPIDEEnv, ActionType
from grader import grade
from tasks import TASKS

app = FastAPI(
    title="AMPIDE — Adversarial Multi-Turn Prompt Injection Defense Environment",
    description="OpenEnv-compliant RL environment for AI safety research.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://huggingface.co",
        "*.huggingface.co",
        "http://localhost:3000",
        "http://localhost:7860",
        "http://127.0.0.1:7860",
    ],
    allow_methods=["GET", "POST"],
    allow_credentials=True,
    allow_headers=["Content-Type"],
)

# ---------------------------------------------------------------------------
# In-memory environment (single-session for HF Space demo)
# ---------------------------------------------------------------------------

_env: Optional[AMPIDEEnv] = None
DEFAULT_TASK = os.getenv("AMPIDE_DEFAULT_TASK", "easy_direct_injection")


def get_env() -> AMPIDEEnv:
    global _env
    if _env is None:
        _env = AMPIDEEnv(task_name=DEFAULT_TASK)
    return _env


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_name: Optional[str] = None
    seed: Optional[int] = None

class ActionRequest(BaseModel):
    action_type: str          # ActionType value
    content: Optional[str] = ""
    tool_name: Optional[str] = None
    tool_args: Optional[Dict[str, Any]] = None

class ObservationResponse(BaseModel):
    conversation: list
    available_tools: list
    task_goal: str
    turn: int
    trust_hints: Optional[str]

class StepResponse(BaseModel):
    observation: ObservationResponse
    reward: float
    done: bool
    info: Dict[str, Any]

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def obs_to_response(obs) -> ObservationResponse:
    return ObservationResponse(
        conversation=obs.conversation,
        available_tools=obs.available_tools,
        task_goal=obs.task_goal,
        turn=obs.turn,
        trust_hints=obs.trust_hints,
    )

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
def home():
    return {"status": "ok", "version": "1.0.0"}
  
@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0.0"}


@app.get("/tasks")
def list_tasks():
    return {
        "tasks": list(TASKS.keys()),
        "descriptions": {
            "easy_direct_injection":      "🟢 Easy — Direct prompt injection. Detect and refuse.",
            "medium_indirect_injection":  "🟡 Medium — Injection hidden in email. Identify and block.",
            "hard_multiturn_tool_attack": "🔴 Hard — Multi-turn attack + tool misuse. Full defense required.",
        },
    }


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    global _env
    task = req.task_name or DEFAULT_TASK
    if task not in TASKS:
        raise HTTPException(status_code=400, detail=f"Unknown task: {task}. Available: {list(TASKS.keys())}")

    seed = req.seed if req.seed is not None else 42
    _env = AMPIDEEnv(task_name=task, seed=seed)
    obs  = _env.reset()
    return obs_to_response(obs)


@app.post("/step")
def step(req: ActionRequest):
    env = get_env()
    try:
        action_type = ActionType(req.action_type)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid action_type '{req.action_type}'. "
                   f"Valid: {[a.value for a in ActionType]}",
        )

    action = AMPIDEAction(
        action_type=action_type,
        content=req.content or "",
        tool_name=req.tool_name,
        tool_args=req.tool_args,
    )

    try:
        result = env.step(action)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return StepResponse(
        observation=obs_to_response(result.observation),
        reward=result.reward,
        done=result.done,
        info=result.info,
    )


@app.get("/state")
def get_state():
    env = get_env()
    return env.state()


@app.get("/grade")
def get_grade():
    env = get_env()
    s = env.state()
    result = grade(s)
    return result


# ---------------------------------------------------------------------------
# Dev server entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860, reload=False)
