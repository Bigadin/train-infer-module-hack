from typing import Optional, Dict, Any
from pydantic import BaseModel, Field


class TrainRequest(BaseModel):
    model_type: str = Field(..., description="'yolo' or 'detr'")
    dataset_url: Optional[str] = None
    params: Dict[str, Any] = {}


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: int
    message: str
    weights_ready: bool
    created_at: float
    updated_at: float

