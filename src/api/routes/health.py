from fastapi import APIRouter

from src import __version__

router = APIRouter()


@router.get("/health")
async def health_check():
    # "shutting_down" tells the web UI container to stop once the API is gone
    # (scripts/web_entrypoint.sh).
    from src.api.shutdown import state

    status = "shutting_down" if state.requested_at else "ok"
    return {"status": status, "version": __version__}
