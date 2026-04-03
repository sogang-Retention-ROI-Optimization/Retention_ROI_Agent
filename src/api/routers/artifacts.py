from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from src.api.dependencies import get_settings
from src.api.schemas import SavedResultsArtifactsResponse, TrainingArtifactsResponse
from src.api.services.artifacts import (
    ensure_saved_results_artifacts,
    ensure_training_artifacts,
    load_saved_results_payload,
    load_training_artifacts_payload,
)
from src.api.settings import ApiSettings

router = APIRouter(prefix='/artifacts', tags=['artifacts'])


@router.get('/training', response_model=TrainingArtifactsResponse)
def training_artifacts(
    rebuild: bool = Query(default=False),
    settings: ApiSettings = Depends(get_settings),
) -> TrainingArtifactsResponse:
    ensure_training_artifacts(settings, rebuild=rebuild)
    return TrainingArtifactsResponse(**load_training_artifacts_payload(settings))


@router.get('/saved-results', response_model=SavedResultsArtifactsResponse)
def saved_results_artifacts(
    budget: int = Query(default=5000000, ge=1),
    rebuild: bool = Query(default=False),
    settings: ApiSettings = Depends(get_settings),
) -> SavedResultsArtifactsResponse:
    ensure_saved_results_artifacts(settings, budget=budget, rebuild=rebuild)
    return SavedResultsArtifactsResponse(**load_saved_results_payload(settings))
