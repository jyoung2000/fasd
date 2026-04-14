"""HTTP endpoints under ``/api/cloud``.

The router is intentionally thin: it does input validation and delegates
every non-trivial operation to :mod:`backend.app.cloud.service` or a
concrete :class:`~backend.app.cloud.base.CloudProvider`.

Flow map:

- ``GET    /api/cloud/providers``                — list status per provider
- ``GET    /api/cloud/{provider}/authorize``     — returns {authorize_url, state}
- ``GET    /api/cloud/{provider}/callback``      — OAuth redirect target
- ``DELETE /api/cloud/accounts/{account_id}``    — disconnect
- ``GET    /api/cloud/{provider}/browse``        — folder listing
- ``GET    /api/cloud/{provider}/search``        — name search
- ``POST   /api/cloud/{provider}/import``        — streamed import → job_id
"""

from __future__ import annotations

import logging
import secrets
from typing import Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field

from backend.app.cloud import accounts
from backend.app.cloud.auth import current_user_id
from backend.app.cloud.providers import cloud_storage_enabled, get_provider, all_providers
from backend.app.cloud.service import (
    CloudServiceError,
    call_with_refresh,
    import_cloud_file,
)
from backend.services.ingest import IngestMetadata
from backend.services.pipeline import broadcast_ws

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/cloud", tags=["cloud-storage"])


# ── helpers ─────────────────────────────────────────────────────────────────


def _ensure_enabled() -> None:
    if not cloud_storage_enabled():
        raise HTTPException(
            status_code=503,
            detail="Cloud storage is disabled. Set CLIPAI_CLOUD_STORAGE_ENABLED=true.",
        )


def _raise_service_error(exc: CloudServiceError) -> None:
    raise HTTPException(status_code=exc.status_code, detail=str(exc))


def _provider_or_404(name: str):
    _ensure_enabled()
    provider = get_provider(name)
    if provider is None:
        raise HTTPException(status_code=404, detail=f"Unknown cloud provider: {name}")
    return provider


# ── schemas ─────────────────────────────────────────────────────────────────


class ProviderStatus(BaseModel):
    name: str
    configured: bool
    connected: bool
    account: Optional[dict] = None


class ProvidersResponse(BaseModel):
    enabled: bool
    providers: list[ProviderStatus]


class AuthorizeResponse(BaseModel):
    authorize_url: str
    state: str


class FolderPageResponse(BaseModel):
    folders: list[dict]
    files: list[dict]
    next_page_token: Optional[str]


class ImportRequest(BaseModel):
    file_id: str = Field(..., description="Provider-specific file id to import")
    language: str = ""
    subtitle_language: str = ""
    content_type_override: str = ""
    game_type: str = ""
    anime_subtype: str = ""
    music_subtype: str = ""
    sports_subtype: str = ""


class ImportResponse(BaseModel):
    job_id: str
    status: str
    filename: str


# ── providers listing ──────────────────────────────────────────────────────


@router.get("/providers", response_model=ProvidersResponse)
async def list_providers() -> ProvidersResponse:
    """Return every known provider and whether the current user is connected."""
    if not cloud_storage_enabled():
        return ProvidersResponse(enabled=False, providers=[])

    user_id = current_user_id()
    user_accounts = {a.provider: a for a in await accounts.list_for_user(user_id)}

    result: list[ProviderStatus] = []
    for name, provider in all_providers().items():
        account = user_accounts.get(name)
        result.append(
            ProviderStatus(
                name=name,
                configured=provider.is_configured(),
                connected=account is not None,
                account=account.to_public() if account else None,
            )
        )
    return ProvidersResponse(enabled=True, providers=result)


# ── OAuth ──────────────────────────────────────────────────────────────────


@router.get("/{provider}/authorize", response_model=AuthorizeResponse)
async def authorize(provider: str) -> AuthorizeResponse:
    prov = _provider_or_404(provider)
    if not prov.is_configured():
        raise HTTPException(
            status_code=503,
            detail=f"{provider} is not configured on the server. See docs/cloud-storage/SETUP.md.",
        )
    state = secrets.token_urlsafe(32)
    await accounts.remember_state(state, provider)
    return AuthorizeResponse(authorize_url=prov.build_authorize_url(state), state=state)


@router.get("/{provider}/callback")
async def callback(
    provider: str,
    code: str = Query(..., description="OAuth authorization code"),
    state: str = Query(..., description="CSRF token returned by /authorize"),
):
    prov = _provider_or_404(provider)
    expected = await accounts.consume_state(state)
    if expected != provider:
        # Don't leak which half was wrong (timing / enumeration safety).
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state")

    try:
        bundle = await prov.exchange_code(code)
        info = await prov.get_account_info(bundle.access_token)
    except Exception as exc:
        logger.exception("OAuth callback failed for %s", provider)
        raise HTTPException(status_code=502, detail=f"OAuth exchange failed: {exc}")

    await accounts.upsert_for_provider(
        user_id=current_user_id(),
        provider=provider,
        provider_user_id=info.provider_user_id,
        display_name=info.display_name,
        access_token=bundle.access_token,
        refresh_token=bundle.refresh_token,
        token_expires_at=bundle.expires_at.isoformat() if bundle.expires_at else None,
        scopes=bundle.scopes,
    )

    # Redirect back into the frontend Settings page, which polls
    # /providers on mount to refresh its state.
    return RedirectResponse(url=f"/settings?connected={provider}", status_code=302)


@router.delete("/accounts/{account_id}")
async def disconnect(account_id: str):
    _ensure_enabled()
    account = await accounts.get(account_id)
    if not account or account.user_id != current_user_id():
        raise HTTPException(status_code=404, detail="Account not found")
    provider = get_provider(account.provider)
    if provider is not None:
        try:
            await provider.revoke(account.access_token)
        except Exception as exc:
            logger.info("Provider revoke failed (non-fatal): %s", exc)
    removed = await accounts.delete(account_id)
    return {"removed": removed}


# ── browse / search ────────────────────────────────────────────────────────


async def _account_for(provider_name: str):
    _ensure_enabled()
    account = await accounts.get_for_provider(current_user_id(), provider_name)
    if account is None:
        raise HTTPException(
            status_code=401,
            detail=f"{provider_name} is not connected. Connect it in Settings first.",
        )
    return account


@router.get("/{provider}/browse", response_model=FolderPageResponse)
async def browse(
    provider: str,
    folder_id: Optional[str] = Query(None),
    page_token: Optional[str] = Query(None),
):
    prov = _provider_or_404(provider)
    account = await _account_for(provider)
    try:
        page = await call_with_refresh(
            account,
            lambda token: prov.list_folder(token, folder_id, page_token),
        )
    except CloudServiceError as exc:
        _raise_service_error(exc)
    except Exception as exc:
        logger.exception("browse failed for %s folder=%s", provider, folder_id)
        raise HTTPException(status_code=502, detail=f"browse failed: {exc}")

    return FolderPageResponse(
        folders=[f.__dict__ for f in page.folders],
        files=[f.__dict__ for f in page.files],
        next_page_token=page.next_page_token,
    )


@router.get("/{provider}/search", response_model=FolderPageResponse)
async def search(
    provider: str,
    q: str = Query(..., min_length=1),
    page_token: Optional[str] = Query(None),
):
    prov = _provider_or_404(provider)
    account = await _account_for(provider)
    try:
        page = await call_with_refresh(
            account,
            lambda token: prov.search(token, q, page_token),
        )
    except CloudServiceError as exc:
        _raise_service_error(exc)
    except Exception as exc:
        logger.exception("search failed for %s q=%r", provider, q)
        raise HTTPException(status_code=502, detail=f"search failed: {exc}")

    return FolderPageResponse(
        folders=[],
        files=[f.__dict__ for f in page.files],
        next_page_token=page.next_page_token,
    )


# ── import ─────────────────────────────────────────────────────────────────


@router.post("/{provider}/import", response_model=ImportResponse)
async def import_file(
    provider: str,
    payload: ImportRequest,
    background_tasks: BackgroundTasks,
):
    """Start a server-side import → returns the job_id immediately.

    The actual download happens in a background task so the HTTP response
    comes back before the (potentially large) file is on disk. The
    frontend navigates to the usual processing view and the existing
    WebSocket on ``/ws/jobs/{job_id}`` picks up the progress channel —
    we emit ``downloading_from_cloud`` status updates from the download
    loop so the existing progress UI renders without changes.
    """
    prov = _provider_or_404(provider)
    account = await _account_for(provider)

    # Resolve the filename up front so we can include it in the response
    # and have something reasonable for the pending job row.
    try:
        meta = await call_with_refresh(
            account,
            lambda token: prov.get_file_metadata(token, payload.file_id),
        )
    except CloudServiceError as exc:
        _raise_service_error(exc)
    except Exception as exc:
        logger.exception("import metadata fetch failed for %s file=%s", provider, payload.file_id)
        raise HTTPException(status_code=502, detail=f"metadata fetch failed: {exc}")

    metadata = IngestMetadata.from_form(
        language=payload.language,
        subtitle_language=payload.subtitle_language,
        content_type_override=payload.content_type_override,
        game_type=payload.game_type,
        anime_subtype=payload.anime_subtype,
        music_subtype=payload.music_subtype,
        sports_subtype=payload.sports_subtype,
        source=provider,
    )

    # We need to expose a job_id in the response BEFORE running the
    # import (the frontend immediately navigates to the processing
    # screen and opens a ws on /ws/jobs/{job_id}). import_cloud_file
    # generates its own job_id internally, so we have to wrap it in a
    # background task that uses a pre-allocated id and threads it
    # through. To keep the public surface small we pass the id by
    # closing over it here.
    import uuid

    pending_job_id = str(uuid.uuid4())

    last_percent = -1

    async def _on_progress(downloaded: int, total: Optional[int]) -> None:
        nonlocal last_percent
        percent = int(downloaded / total * 100) if total and total > 0 else 0
        if percent == last_percent:
            return
        last_percent = percent
        await broadcast_ws(
            pending_job_id,
            {
                "type": "status",
                "status": "downloading_from_cloud",
                "progress": percent,
                "message": (
                    f"Downloading from {provider}: {percent}%"
                    if total
                    else f"Downloading from {provider}..."
                ),
            },
        )

    async def _run() -> None:
        try:
            await broadcast_ws(
                pending_job_id,
                {
                    "type": "status",
                    "status": "downloading_from_cloud",
                    "progress": 0,
                    "message": f"Starting download from {provider}...",
                },
            )
            await import_cloud_file(
                account,
                payload.file_id,
                metadata=metadata,
                progress=_on_progress,
                pending_job_id=pending_job_id,
            )
        except Exception as exc:
            logger.exception(
                "Cloud import failed (job_id=%s provider=%s)",
                pending_job_id,
                provider,
            )
            await broadcast_ws(
                pending_job_id,
                {
                    "type": "status",
                    "status": "failed",
                    "progress": 0,
                    "message": f"Cloud import failed: {exc}",
                },
            )

    background_tasks.add_task(_run)

    return ImportResponse(
        job_id=pending_job_id,
        status="downloading_from_cloud",
        filename=meta.name,
    )
