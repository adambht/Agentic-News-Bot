from pydantic import BaseModel, Field

class NewsItem(BaseModel):
    """Pydantic model for a news item."""
    title: str
    text: str
    subject: str
    date: str
    label: int = Field(default=0)  # dummy for testing with ML model
    
class VerificationResult(BaseModel):
    """Pydantic model for web search verification result."""
    verdict: int  # 1 for True, 0 for False
    url: str = ""  # optional supporting link
    
    
    
    