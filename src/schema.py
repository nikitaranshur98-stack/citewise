from typing import Literal
from pydantic import BaseModel, Field

class Source(BaseModel):
    title: str = Field(description="Title of the document, web page, or paper")
    location: str = Field(description="URL, or document filename with page number")
    source_type: Literal["document", "web", "arxiv", "wikipedia"] = Field(
        description="Where this source came from"
    )
    
class Claim(BaseModel):
    statement: str = Field(description="One factual statement that answers part of the question")
    sources: list[Source] = Field(description="Sources that directly support this statement")
    confidence: Literal["high", "medium", "low"] = Field(
        description="How strongly the sources support the statement"
    )

class Briefing(BaseModel):
    question: str = Field(description="The user's original question")
    summary: str = Field(description="A short 2-3 sentence overview of the answer")
    claims: list[Claim] = Field(description="Key factual claims, each with its sources")
    limitations: str = Field(
        description="What could not be found or verified; say 'None' if nothing"
    )
    
