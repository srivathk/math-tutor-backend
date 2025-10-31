from __future__ import annotations

import json
import os
import re
from typing import Dict, Optional, Tuple

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, constr
from sympy import ConditionSet, S, Symbol, simplify, solveset
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)
from openai import OpenAI

# --- env / client ------------------------------------------------------------
load_dotenv()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

_client: Optional[OpenAI] = None
if OPENAI_API_KEY:
    try:
        _client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        _client = None  # handled at runtime

# --- FastAPI -----------------------------------------------------------------
app = FastAPI(title="Math Tutor Backend", version="1.2.0")

# CORS for Next.js at http://localhost:3000
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- SymPy parsing helpers ---------------------------------------------------
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)
SAFE_LOCALS: Dict[str, object] = {"x": Symbol("x")}  # extend later if you add y, z, ...

def _normalize_expr(s: str) -> str:
    # accept ^ for exponent
    return s.replace("^", "**").strip()

def _parse_equation(text: str) -> Tuple[object, object]:
    """
    Accepts either a clean equation 'LHS = RHS' or free text containing an equation.
    Uses extract_equation() first, then parses each side into SymPy expressions.
    """
    cleaned = extract_equation(text)
    if "=" not in cleaned:
        raise ValueError("Equation must contain '='.")
    lhs, rhs = cleaned.split("=", 1)
    lhs_expr = parse_expr(_normalize_expr(lhs.strip()), transformations=TRANSFORMS, local_dict=SAFE_LOCALS)
    rhs_expr = parse_expr(_normalize_expr(rhs.strip()), transformations=TRANSFORMS, local_dict=SAFE_LOCALS)
    return lhs_expr, rhs_expr

def _differences_equivalent(d0, d1) -> bool:
    """
    Return True if two equation differences represent the same constraint.
    Robust to scalar multiples and many algebraically equivalent forms.
    """
    d0_s = simplify(d0)
    d1_s = simplify(d1)

    # identical zeros → both sides equal
    if d0_s == 0 and d1_s == 0:
        return True
    # if the proposed difference simplifies to 0, it's at least correct
    if d1_s == 0:
        return True
    # original is 0 but proposed isn't → not equivalent
    if d0_s == 0:
        return False

    # direct equality check
    if simplify(d0_s - d1_s) == 0:
        return True

    # compare solution sets when single variable
    symbols = d0_s.free_symbols | d1_s.free_symbols
    if len(symbols) == 1:
        var = next(iter(symbols))
        try:
            sol0 = solveset(d0_s, var, domain=S.Reals)
            sol1 = solveset(d1_s, var, domain=S.Reals)
            if not isinstance(sol0, ConditionSet) and not isinstance(sol1, ConditionSet):
                if sol0 == sol1:
                    return True
        except Exception:
            pass

    # final attempt: constant multiple?
    try:
        ratio = simplify(d0_s / d1_s)
    except Exception:
        return False
    if ratio in (S.NaN, S.ComplexInfinity):
        return False
    if getattr(ratio, "free_symbols", set()):
        return False
    if ratio.is_zero or ratio.is_finite is False:
        return False
    return True

# --- Output cleaners (handle 'becomes', extra words, etc.) -------------------
CONNECTOR_PATTERN = re.compile(
    r"\b(?:becomes|then|so|thus|which|and|therefore|hence|giving|gives|resulting|results|"
    r"leading(?:\s+to)?|yielding|with|where)\b",
    re.IGNORECASE,
)

def _clean_equation_side(segment: str, prefer_right: bool = False) -> str:
    if not segment:
        return ""
    piece = segment.replace("\n", " ").strip()
    piece = re.sub(r"[:;]", "|", piece)
    piece = CONNECTOR_PATTERN.sub("|", piece)
    parts = [p.strip(" \t.,") for p in piece.split("|") if p.strip(" \t.,")]
    if not parts:
        parts = [piece.strip(" \t.,")]
    # choose the last meaningful chunk (often after 'becomes')
    chosen = ""
    for part in reversed(parts):
        if part:
            chosen = part
            break
    if not chosen:
        chosen = piece.strip()
    if "=" in chosen:
        chosen = chosen.split("=")[-1 if prefer_right else 0].strip()
    return re.sub(r"\s+", " ", chosen).strip(" \t.,")

def extract_equation(text: str) -> str:
    """
    From arbitrary text, extract a single 'LHS = RHS' equation.
    Prefers the LAST equation found and cleans both sides.
    If none found, returns the stripped original text.
    """
    if not text:
        return ""
    matches = re.findall(r"[^=]+=[^=]+", text)
    for candidate in reversed(matches):
        lhs_raw, rhs_raw = candidate.split("=", 1)
        lhs = _clean_equation_side(lhs_raw, prefer_right=False)
        rhs = _clean_equation_side(rhs_raw, prefer_right=True)
        if lhs and rhs and "=" not in lhs and "=" not in rhs:
            return f"{lhs} = {rhs}"
    return text.strip()

def clean_proposed_step(raw: str, original_problem: str) -> str:
    """
    Ensures we return a full equation. If the model omits RHS/LHS,
    keep the missing side from the original problem.
    """
    raw = (raw or "").strip()
    # Prefer content after 'becomes' or arrows
    mb = re.search(r"(?:becomes|->|=>|→)\s*([^\n=]+)", raw, flags=re.I)
    if mb:
        left = _clean_equation_side(mb.group(1), prefer_right=False)
        rhs_from_raw = raw.split("=")[-1].strip() if "=" in raw else None
        rhs = _clean_equation_side(rhs_from_raw, prefer_right=True) if rhs_from_raw else None
        if not rhs:
            rhs = original_problem.split("=", 1)[1].strip() if "=" in original_problem else ""
        return f"{left} = {rhs}".strip()

    # Or extract a normal equation from text
    eq = extract_equation(raw)
    if "=" in eq:
        return eq

    # No '=' → salvage expression and reuse RHS from original
    rhs0 = original_problem.split("=", 1)[1].strip() if "=" in original_problem else ""
    expr_chunks = re.findall(r"[0-9a-zA-ZxX\^\+\-\*/\(\)\s]+", raw)
    expr = expr_chunks[-1].strip() if expr_chunks else raw
    return f"{expr} = {rhs0}".strip()

# --- Schemas -----------------------------------------------------------------
class StepCheckRequest(BaseModel):
    problem: constr(strip_whitespace=True, min_length=3, max_length=400) = Field(
        ..., description="Equation, e.g. '3(x-2)=12'"
    )
    # Accept 'student_step' but also allow clients to send 'proposed_step'
    student_step: constr(strip_whitespace=True, min_length=1, max_length=400) = Field(
        ..., description="Transformed equation, e.g. '3x-6=12'", validation_alias="proposed_step"
    )

class TutorRequest(BaseModel):
    problem: constr(strip_whitespace=True, min_length=3, max_length=400)
    previous_step: Optional[constr(strip_whitespace=True, min_length=1, max_length=400)] = None

class TutorResponse(BaseModel):
    hint: str
    proposed_step: Optional[str]
    verified: bool

# --- Routes ------------------------------------------------------------------
@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}

@app.post("/check-step")
def check_step(req: StepCheckRequest):
    try:
        o_lhs, o_rhs = _parse_equation(req.problem)
        s_lhs, s_rhs = _parse_equation(req.student_step)
        ok = _differences_equivalent(o_lhs - o_rhs, s_lhs - s_rhs)
        return {"verified": bool(ok), "message": "Checked successfully"}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid math input")

@app.post("/tutor", response_model=TutorResponse)
def tutor(req: TutorRequest):
    if _client is None:
        raise HTTPException(status_code=500, detail="OpenAI not configured")

    # Very strict system prompt to keep JSON clean
    sys = (
        "You are a concise algebra tutor. Reply with STRICT JSON only, no extra text. "
        'Return: {"hint": "<one short sentence>", "proposed_step": "<single equation with exactly one = >"}. '
        "If only one side changes, keep the other side identical to the original problem. "
        "Do NOT use words like 'becomes' or explanatory text in proposed_step."
    )
    ask = {"problem": req.problem, "previous_step": req.previous_step}

    # Call OpenAI and parse JSON safely; fall back to text-cleaning if needed
    try:
        resp = _client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {"role": "system", "content": sys},
                {"role": "user", "content": json.dumps(ask)},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content
        try:
            data = json.loads(content)
            hint = str(data.get("hint", "")).strip()
            raw_step = str(data.get("proposed_step", ""))
        except Exception:
            # Model didn't send perfect JSON—salvage from text
            hint = "Make one valid algebraic transformation."
            raw_step = content or ""
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation error: {e}")

    # Clean/repair the step and verify
    proposed_step = clean_proposed_step(raw_step, req.problem)
    verified = False
    if "=" in proposed_step:
        try:
            o_lhs, o_rhs = _parse_equation(req.problem)
            s_lhs, s_rhs = _parse_equation(proposed_step)
            verified = _differences_equivalent(o_lhs - o_rhs, s_lhs - s_rhs)
        except Exception:
            verified = False

    return TutorResponse(hint=hint, proposed_step=proposed_step, verified=verified)