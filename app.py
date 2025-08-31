from fastapi import FastAPI
from pydantic import BaseModel
from sympy import Eq, simplify
from sympy.parsing.sympy_parser import parse_expr
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application
)

# transformations that allow 3(x-2) instead of 3*(x-2)
TRANSFORMS = standard_transformations + (implicit_multiplication_application,)

app = FastAPI()

# Health check
@app.get("/health")
def health():
    return {"status": "ok"}

class StepCheckRequest(BaseModel):
    problem: str
    student_step: str

@app.post("/check-step")
def check_step(req: StepCheckRequest):
    try:
        # Parse the problem as an equation
        if "=" not in req.problem:
            return {"valid": False, "message": "Problem must contain an equation."}
        lhs, rhs = req.problem.split("=")
        lhs_expr = parse_expr(lhs.strip(), transformations=TRANSFORMS)
        rhs_expr = parse_expr(rhs.strip(), transformations=TRANSFORMS)

        # Parse student's step
        if "=" not in req.student_step:
            return {"valid": False, "message": "Step must also contain an equation."}
        s_lhs, s_rhs = req.student_step.split("=")
        s_lhs_expr = parse_expr(s_lhs.strip(), transformations=TRANSFORMS)
        s_rhs_expr = parse_expr(s_rhs.strip(), transformations=TRANSFORMS)

        # Compare by simplification
        diff1 = simplify(lhs_expr - rhs_expr)
        diff2 = simplify(s_lhs_expr - s_rhs_expr)

        return {"valid": simplify(diff1 - diff2) == 0, "message": "Checked successfully"}
    except Exception as e:
        return {"valid": False, "error": str(e)}