"""
test_match_insert, test_duplicate_match_is_safe (Phase 5, раздел 20).

Запускается ПРОТИВ РЕАЛЬНОЙ PostgreSQL (не мока) — в этой среде разработки
поднят локальный кластер (docs/environment-constraints.md: сеть к внешним
источникам заблокирована, но локальная БД полностью доступна и это
качественно более сильная проверка, чем sqlite/мок). Если TEST_DATABASE_URL
недоступен (например, CI без Postgres), тесты пропускаются, а не падают.

Намеренно используется ОТДЕЛЬНАЯ БД (TEST_DATABASE_URL), не DATABASE_URL:
autouse-фикстура ниже очищает matches/teams перед и после каждого теста —
на Phase 5 live (реальный OpenDota ingestion) это буквально стёрло 942 уже
загруженных матча, когда тесты были запущены против той же БД, что и
ingestion (см. reports/live-data-verification.md, раздел "Инциденты").
Без TEST_DATABASE_URL тесты пропускаются, а НЕ падают обратно на
database_url — иначе это та же дыра под другим именем.
"""

from datetime import datetime, timezone

import pytest
from sqlalchemy import func, select

from src.config import load_settings
from src.db.engine import make_engine
from src.db.schema import matches, teams
from src.repositories import match_repository as repo

settings = load_settings()

try:
    _engine = make_engine(settings, use_test_database=True) if settings.test_database_url else None
    if _engine is not None:
        with _engine.connect():
            pass
    DB_AVAILABLE = _engine is not None
except Exception:
    DB_AVAILABLE = False

pytestmark = pytest.mark.skipif(not DB_AVAILABLE, reason="TEST_DATABASE_URL недоступен в этой среде")


@pytest.fixture
def engine():
    return make_engine(settings, use_test_database=True)


@pytest.fixture(autouse=True)
def _clean_tables(engine):
    with engine.begin() as conn:
        conn.execute(matches.delete())
        conn.execute(teams.delete())
    yield
    with engine.begin() as conn:
        conn.execute(matches.delete())
        conn.execute(teams.delete())


def test_match_insert(engine):
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        repo.upsert_team(conn, 100, "Team Alpha", "ALP", now)
        repo.upsert_team(conn, 200, "Team Beta", "BET", now)
        repo.upsert_match(
            conn, match_id=9001, start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            duration_seconds=1800, radiant_team_id=100, dire_team_id=200, radiant_win=True,
            league_id=None, patch_id=None, series_id=None, series_type=None,
            source="opendota", ingested_at=now,
        )

    with engine.connect() as conn:
        row = conn.execute(select(matches).where(matches.c.match_id == 9001)).first()
    assert row is not None
    assert row.radiant_team_id == 100
    assert row.radiant_win is True


def test_duplicate_match_is_safe(engine):
    """Phase 5, раздел 16-17: повторная вставка того же match_id (тот же
    natural key) не создаёт дубль и не падает — upsert, не INSERT."""
    now = datetime.now(timezone.utc)

    def insert_once():
        with engine.begin() as conn:
            repo.upsert_team(conn, 100, "Team Alpha", "ALP", now)
            repo.upsert_team(conn, 200, "Team Beta", "BET", now)
            repo.upsert_match(
                conn, match_id=9002, start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
                duration_seconds=1800, radiant_team_id=100, dire_team_id=200, radiant_win=True,
                league_id=None, patch_id=None, series_id=None, series_type=None,
                source="opendota", ingested_at=now,
            )

    insert_once()
    insert_once()
    insert_once()

    with engine.connect() as conn:
        count = conn.execute(
            select(func.count()).select_from(matches).where(matches.c.match_id == 9002)
        ).scalar()
    assert count == 1, f"три идентичных upsert должны дать ровно одну строку, получено {count}"


def test_upsert_updates_changed_fields(engine):
    """Upsert должен ОБНОВЛЯТЬ значения при повторной вставке с другими
    данными (не игнорировать, docs/data-pipeline.md: источник может
    прислать уточнённые данные для уже известного match_id)."""
    now = datetime.now(timezone.utc)
    with engine.begin() as conn:
        repo.upsert_team(conn, 100, "Team Alpha", "ALP", now)
        repo.upsert_team(conn, 200, "Team Beta", "BET", now)
        repo.upsert_match(
            conn, match_id=9003, start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            duration_seconds=1800, radiant_team_id=100, dire_team_id=200, radiant_win=True,
            league_id=None, patch_id=None, series_id=111, series_type=1,
            source="opendota", ingested_at=now,
        )
        # Повтор с изменённым series_id (источник уточнил данные)
        repo.upsert_match(
            conn, match_id=9003, start_time=datetime(2024, 1, 1, tzinfo=timezone.utc),
            duration_seconds=1800, radiant_team_id=100, dire_team_id=200, radiant_win=True,
            league_id=None, patch_id=None, series_id=222, series_type=1,
            source="opendota", ingested_at=now,
        )

    with engine.connect() as conn:
        row = conn.execute(select(matches).where(matches.c.match_id == 9003)).first()
    assert row.series_id == 222, "upsert должен обновить series_id, не сохранить первое значение"
