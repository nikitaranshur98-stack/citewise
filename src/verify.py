from typing import Literal

from langchain_google_genai import ChatGoogleGenerativeAI
from pydantic import BaseModel, Field

from src.config import MODEL_NAME
from src.retry import call_with_retry

MAX_EVIDENCE_CHARS = 12000


class ClaimCheck(BaseModel):
    claim_number: int = Field(description="Number of the claim being checked, starting at 1")
    verdict: Literal["supported", "partial", "unsupported"] = Field(
        description="supported: the evidence states it. partial: only some of it is stated. "
        "unsupported: the evidence does not state it."
    )
    reason: str = Field(description="One short sentence explaining the verdict")


class VerificationReport(BaseModel):
    checks: list[ClaimCheck]
    
def verify_briefing(briefing, evidence):
    if not briefing.claims:
        return []

    evidence_text = "\n\n---\n\n".join(evidence)[:MAX_EVIDENCE_CHARS]
    claims_text = "\n".join(
        f"{i}. {claim.statement}" for i, claim in enumerate(briefing.claims, start=1)
    )

    prompt = f"""You are a strict fact checker.
Judge each numbered claim ONLY against the evidence below.
Do not use your own knowledge. If the evidence does not state the claim, the verdict is "unsupported".

EVIDENCE:
{evidence_text}

CLAIMS:
{claims_text}

Return one check for every claim."""

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME, temperature=0, timeout=60, max_retries=2
    )
    checker = llm.with_structured_output(VerificationReport)
    report = call_with_retry(lambda: checker.invoke(prompt))
    return report.checks

