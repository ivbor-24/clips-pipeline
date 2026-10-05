from pydantic import BaseModel


class UploadResponse(BaseModel):
    path: str
    filename: str
    size: int
