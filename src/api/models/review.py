from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.sql import func

from src.api.db_base import Base


class ClipReview(Base):
    """The user's decision on one clip of a job: keep / reject / edit, with notes.

    Stored in the database rather than in ``<clip>.meta.json``, which rendering
    rewrites. The clip's time range is kept with the decision: a re-render of
    the same clip keeps its review, while a re-scoring that puts other content
    under the same clip id starts it as pending again.
    """

    __tablename__ = "clip_reviews"
    __table_args__ = (UniqueConstraint("job_id", "clip_id", name="uq_clip_review"),)

    id = Column(Integer, primary_key=True)
    job_id = Column(Integer, ForeignKey("jobs.id", ondelete="CASCADE"), nullable=False, index=True)
    clip_id = Column(String(64), nullable=False)
    start_time = Column(Float, nullable=True)
    end_time = Column(Float, nullable=True)
    status = Column(String(16), nullable=False)
    notes = Column(Text, nullable=True)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
