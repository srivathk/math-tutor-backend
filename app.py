from __future__ import annotations

from typing import Dict
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, constr
from sympy import Symbol, simplify
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
)

# Allow 3(x-2) as 3*(x-2)
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

# Only expose symbols we expect
SAFE_LOCALS: Dict[str, object] = {"x": Symbol("x")}

app = FastAPI(title="Math Tutor Backend", version="1.0.0")


class StepCheckRequest(BaseModel):
    """Request body for checking whether a student's algebraic step preserves an equation."""
    problem: constr(strip_whitespace=True, min_length=3, max_length=400) = Field(
        ..., description="Original equation, e.g. '3(x-2)=12'"
    )
    student_step: constr(strip_whitespace=True, min_length=1, max_length=400) = Field(
        ..., description="Student's transformed equation, e.g. '3x-6=12'"
    )


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def _parse_equation(text: str):
    """Parse 'lhs=rhs' into SymPy expressions with implicit multiplication allowed."""
    if "=" not in text:
        raise ValueError("Equation must contain '='.")
    lhs, rhs = text.split("=", 1)
    lhs_expr = parse_expr(lhs.replace("^", "**").strip(), transformations=TRANSFORMS, local_dict=SAFE_LOCALS)
    rhs_expr = parse_expr(rhs.replace("^", "**").strip(), transformations=TRANSFORMS, local_dict=SAFE_LOCALS)
    return lhs_expr, rhs_expr


@app.post("/check-step")
def check_step(req: StepCheckRequest):
    """
    Returns valid=true if the student's equation has the same solution set
    as the original (by comparing simplified differences).
    """
    try:
        o_lhs, o_rhs = _parse_equation(req.problem)
        s_lhs, s_rhs = _parse_equation(req.student_step)

        diff_original = simplify(o_lhs - o_rhs)
        diff_student = simplify(s_lhs - s_rhs)
        ok = simplify(diff_original - diff_student) == 0

        return {"valid": bool(ok), "message": "Checked successfully"}
    except Exception as e:
        # Keep error minimal

        raise HTTPException(status_code=400, detail="Invalid math input") from e
