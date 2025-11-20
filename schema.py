from typing import List, Optional

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Schema for a source used by the agent"""

    url: str = Field(description="The URL of the source")


class AgentResponse(BaseModel):
    """Schema for agent response with anser and sources"""

    answer: str = Field(description="The answer to the question")
    sources: List[Source] = Field(
        defdescription="The list of sources used to answer the question"
    )
