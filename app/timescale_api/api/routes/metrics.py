import datetime

from fastapi import APIRouter, Depends
from shared.schemas import (
    GenericResponse,
    MetricHistoryRangeRead,
    MetricHistoryRead,
    MetricRead,
    PredictData,
    RawCpuCreate,
)
from sqlalchemy import func, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from ..database import get_db
from ..models import MetricEntry, ModelRegistry, RawCpuReading

router = APIRouter(prefix="/metrics", tags=["metrics"])


@router.post("/raw", response_model=GenericResponse)
async def save_raw_cpu(
    data: RawCpuCreate, db: AsyncSession = Depends(get_db)
) -> GenericResponse:
    """[Internal] Store a raw CPU reading with its timestamp."""
    reading = RawCpuReading(ts=data.ts, cpu_value=data.cpu_value)
    db.add(reading)
    await db.commit()
    return GenericResponse(status="success", message="Raw reading saved")


@router.put("/{row_id}/actual", response_model=GenericResponse)
async def sync_actual_by_id(
    row_id: int, db: AsyncSession = Depends(get_db)
) -> GenericResponse:
    """
    [Internal] Find the raw_cpu_readings value closest to target_ts and write it
    as actual_cpu. Accurate even after a service restart.
    """
    row_result = await db.execute(select(MetricEntry).where(MetricEntry.id == row_id))
    row = row_result.scalar_one_or_none()
    if not row or row.actual_cpu is not None:
        return GenericResponse(status="success", message="Nothing to sync")

    closest_result = await db.execute(
        select(RawCpuReading)
        .order_by(func.abs(func.extract("epoch", RawCpuReading.ts - row.target_ts)))
        .limit(1)
    )
    closest = closest_result.scalar_one_or_none()
    if not closest:
        return GenericResponse(status="error", message="No raw readings available")

    stmt = (
        update(MetricEntry)
        .where(MetricEntry.id == row_id)
        .values(actual_cpu=closest.cpu_value)
    )
    await db.execute(stmt)
    await db.commit()
    return GenericResponse(status="success", message=f"Row {row_id} synced")


@router.post("/predict", response_model=MetricRead)
async def save_new_prediction(
    data: PredictData, db: AsyncSession = Depends(get_db)
) -> MetricRead:
    """
    [Internal] Persist a new GRU prediction from the collector.
    target_ts is computed as now + horizon_seconds, where horizon_seconds
    equals forecast_horizon × collection_interval as reported by the predictor.
    """
    query = select(ModelRegistry.version).filter(ModelRegistry.is_active == True)
    result = await db.execute(query)
    active_version = result.scalar_one_or_none()

    if not active_version:
        active_version = "unknown-model"

    current_time = datetime.datetime.now(datetime.timezone.utc)
    target_time = current_time + datetime.timedelta(seconds=data.horizon_seconds)

    new_entry = MetricEntry(
        ts=current_time,
        target_ts=target_time,
        input_cpu=data.input_cpu,
        input_ram=data.input_ram,
        input_rps=data.input_rps,
        predicted_cpu=data.predicted_cpu,
        horizon_seconds=data.horizon_seconds,
        model_version=active_version,
    )

    db.add(new_entry)
    await db.commit()
    await db.refresh(new_entry)

    return MetricRead.model_validate(new_entry)


@router.get("/unsynced", response_model=list[MetricRead])
async def get_unsynced(db: AsyncSession = Depends(get_db)) -> list[MetricRead]:
    """
    [Internal] Return rows where actual_cpu IS NULL and target_ts has already passed.
    Used by the collector on startup to restore pending_sync.
    """
    now = datetime.datetime.now(datetime.timezone.utc)
    query = (
        select(MetricEntry)
        .where(MetricEntry.actual_cpu.is_(None))
        .where(MetricEntry.target_ts < now)
        .order_by(MetricEntry.target_ts.asc())
    )
    result = await db.execute(query)
    return [MetricRead.model_validate(m) for m in result.scalars().all()]


@router.get("/history", response_model=list[MetricRead])
async def get_history(
    history_read: MetricHistoryRead, db: AsyncSession = Depends(get_db)
) -> list[MetricRead]:
    """Return recent metric history for the Dashboard and Grafana."""
    query = (
        select(MetricEntry).order_by(MetricEntry.ts.desc()).limit(history_read.limit)
    )
    result = await db.execute(query)
    orm_metrics = result.scalars().all()
    return [MetricRead.model_validate(metric) for metric in orm_metrics]


@router.get("/history/range", response_model=list[MetricRead])
async def get_history_by_range(
    history_range_read: MetricHistoryRangeRead,
    db: AsyncSession = Depends(get_db),
) -> list[MetricRead]:
    query = (
        select(MetricEntry)
        .filter(MetricEntry.actual_cpu.isnot(None))
        .filter(MetricEntry.ts >= history_range_read.start_time)
        .filter(MetricEntry.ts <= history_range_read.end_time)
        .order_by(MetricEntry.ts.asc())
    )

    result = await db.execute(query)
    orm_metrics = result.scalars().all()
    return [MetricRead.model_validate(metric) for metric in orm_metrics]
