"""
Phase 8 — player/roster strength (R1, R3, R4, R6, R7).

Центральная гипотеза Phase 8: **current roster strength != roster stability**
(Phase 7 проверила только stability и не нашла сигнала). Здесь строится
величина, отвечающая на вопрос "насколько сильны ИМЕННО ЭТИ пять игроков",
а не "давно ли они вместе".

## Математика PlayerEloEngine

Матч = состязание двух МНОЖЕСТВ игроков, не двух team_id:

    R_A(t) = mean( r_p(t) : p in roster_A )        (сила пятёрки)
    E_A    = 1 / (1 + 10^((R_B - R_A)/400))        (ожидание, как у Elo)
    delta  = K * (S_A - E_A)
    r_p   += delta   для p in roster_A
    r_p   -= delta   для p in roster_B

Каждый игрок получает ОДИНАКОВЫЙ кредит за исход (shared credit) — мы не
пытаемся приписать индивидуальный вклад по KDA/GPM, потому что это
статистика ТЕКУЩЕГО матча (запрещена, раздел 11 задания) и потому что
индивидуальная статистика сильно смешана с силой команды (раздел 15).

Ключевое свойство, ради которого это построено: при СТАБИЛЬНОМ составе
mean(5 игроков) двигается ровно как team Elo (каждый из 5 сдвигается на
delta => среднее сдвигается на delta) — то есть player-Elo НЕ дублирует
team-Elo избыточно. Различие возникает ровно тогда, когда игрок меняет
команду: рейтинг едет ЗА ИГРОКОМ, чего team-Elo принципиально не умеет.
Именно это и есть проверяемая гипотеза.

K=16 взят тем же, что у team-Elo (Phase 6.5 подобрала его на validation).
Он НЕ подбирался заново под player-Elo — сознательно, чтобы не заниматься
hyperparameter hunting (раздел 71 задания) и чтобы разница с baseline не
объяснялась просто другим K.

## Leakage-семантика (раздел 55 задания)

| Величина | source_timestamp | Доступна на prediction time? |
|---|---|---|
| r_p (рейтинг игрока) | все матчи игрока СТРОГО < t | да (walk-forward) |
| состав пятёрки на матч t | заявка на матч (в реальности публикуется до игры) | да — то же соглашение, что Phase 7 |
| pair synergy | совместные матчи пары СТРОГО < t | да |
| matches_played | матчи игрока СТРОГО < t | да |

Состав текущего матча читается из `match_players` этого матча. Это
СОЗНАТЕЛЬНОЕ соглашение, унаследованное от Phase 7 и зафиксированное в
`docs/features.md` (категория C, `stand_in_flag`): для обучения факт
состава известен, для продового инференса состав обязан приходить из
анонса лайнапа, а не из результата матча. Никакая ИГРОВАЯ статистика
текущего матча (kills/gpm/xpm/damage/lane_role) здесь не используется.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from typing import Dict, FrozenSet, Iterable, List, Optional, Protocol, Tuple

from sqlalchemy import Engine, text

DEFAULT_K = 16.0          # тот же, что team-Elo (Phase 6.5), НЕ подбирался заново
DEFAULT_BASE_RATING = 1000.0
SYNERGY_PRIOR_GAMES = 20.0  # сила shrinkage к 0.5 для pair synergy (раздел 20/25 задания)


class MatchWithRoster(Protocol):
    match_id: int
    start_time: datetime
    radiant_team_id: int
    dire_team_id: int
    radiant_roster: FrozenSet[int]
    dire_roster: FrozenSet[int]
    radiant_win: bool


@dataclass(frozen=True)
class PlayerFeatureRow:
    match_id: int
    as_of_timestamp: datetime
    radiant_team_id: int
    dire_team_id: int

    # R1/R3 — сила текущей пятёрки
    radiant_player_elo_mean: float
    dire_player_elo_mean: float
    radiant_player_elo_min: float   # "слабое звено"
    dire_player_elo_min: float
    radiant_player_elo_max: float
    dire_player_elo_max: float

    # R4 — насколько текущая пятёрка сильнее/слабее исторической силы team_id
    radiant_roster_vs_team_delta: Optional[float]
    dire_roster_vs_team_delta: Optional[float]

    # R6 — сыгранность пары (shrunk)
    radiant_pair_synergy: Optional[float]
    dire_pair_synergy: Optional[float]

    # R7 — объём доказательств (cold-start uncertainty)
    radiant_player_matches_min: int
    dire_player_matches_min: int
    radiant_player_matches_mean: float
    dire_player_matches_mean: float

    radiant_win: bool


class _PlayerRatingEngine:
    """Walk-forward player Elo. Состояние читается ДО матча, обновляется
    строго ПОСЛЕ (тот же контракт, что RatingEngine из Phase 4)."""

    def __init__(self, k_factor: float = DEFAULT_K, base_rating: float = DEFAULT_BASE_RATING):
        self.k_factor = k_factor
        self.base_rating = base_rating
        self._ratings: Dict[int, float] = {}
        self._matches_played: Dict[int, int] = defaultdict(int)
        # team-Elo, посчитанный ЭТИМ ЖЕ движком по team_id — нужен для R4,
        # чтобы roster_vs_team_delta сравнивал величины одной природы/шкалы
        # (сравнивать player-Elo с Elo из другого движка/другим K было бы
        # некорректно — разные динамики).
        self._team_ratings: Dict[int, float] = {}

    def rating(self, account_id: int) -> float:
        return self._ratings.get(account_id, self.base_rating)

    def team_rating(self, team_id: int) -> float:
        return self._team_ratings.get(team_id, self.base_rating)

    def matches_played(self, account_id: int) -> int:
        return self._matches_played.get(account_id, 0)

    def roster_rating(self, roster: FrozenSet[int]) -> float:
        if not roster:
            return self.base_rating
        return sum(self.rating(p) for p in roster) / len(roster)

    def process(
        self,
        radiant_roster: FrozenSet[int],
        dire_roster: FrozenSet[int],
        radiant_team_id: int,
        dire_team_id: int,
        radiant_win: bool,
    ) -> None:
        r_rating = self.roster_rating(radiant_roster)
        d_rating = self.roster_rating(dire_roster)
        expected_r = 1.0 / (1.0 + 10 ** ((d_rating - r_rating) / 400.0))
        score_r = 1.0 if radiant_win else 0.0
        delta = self.k_factor * (score_r - expected_r)

        for p in radiant_roster:
            self._ratings[p] = self.rating(p) + delta
            self._matches_played[p] += 1
        for p in dire_roster:
            self._ratings[p] = self.rating(p) - delta
            self._matches_played[p] += 1

        # Параллельный team-Elo на тех же K/базе (для R4)
        tr, td = self.team_rating(radiant_team_id), self.team_rating(dire_team_id)
        exp_t = 1.0 / (1.0 + 10 ** ((td - tr) / 400.0))
        dt = self.k_factor * (score_r - exp_t)
        self._team_ratings[radiant_team_id] = tr + dt
        self._team_ratings[dire_team_id] = td - dt


class _PairSynergyTracker:
    """Совместные результаты пар игроков со shrinkage к 0.5.

    Сырой win rate пары при N=2 бессмысленен (раздел 20 задания: "не
    доверять synergy estimate при N=2"), поэтому используется
    Beta-Binomial сглаживание:

        shrunk_wr = (wins + m/2) / (games + m),  m = SYNERGY_PRIOR_GAMES

    При games=0 это ровно 0.5 (нет информации), при games >> m стремится к
    наблюдаемому win rate. Возвращается ОТКЛОНЕНИЕ от 0.5, чтобы 0
    означало "нет сигнала".
    """

    def __init__(self, prior_games: float = SYNERGY_PRIOR_GAMES):
        self.prior_games = prior_games
        self._games: Dict[Tuple[int, int], int] = defaultdict(int)
        self._wins: Dict[Tuple[int, int], int] = defaultdict(int)

    @staticmethod
    def _key(a: int, b: int) -> Tuple[int, int]:
        return (a, b) if a <= b else (b, a)

    def mean_pair_synergy(self, roster: FrozenSet[int]) -> Optional[float]:
        if len(roster) < 2:
            return None
        vals = []
        for a, b in combinations(sorted(roster), 2):
            k = self._key(a, b)
            g = self._games.get(k, 0)
            w = self._wins.get(k, 0)
            vals.append((w + self.prior_games / 2.0) / (g + self.prior_games) - 0.5)
        return sum(vals) / len(vals) if vals else None

    def observe(self, roster: FrozenSet[int], won: bool) -> None:
        for a, b in combinations(sorted(roster), 2):
            k = self._key(a, b)
            self._games[k] += 1
            self._wins[k] += int(won)


def build_player_features(
    matches: Iterable[MatchWithRoster],
    k_factor: float = DEFAULT_K,
) -> List[PlayerFeatureRow]:
    """matches ОБЯЗАН быть отсортирован по (start_time, match_id) — тот же
    tie-breaker, что и все остальные Feature Set модули (Phase 6.5, р.10)."""
    engine = _PlayerRatingEngine(k_factor=k_factor)
    synergy = _PairSynergyTracker()
    rows: List[PlayerFeatureRow] = []

    for m in matches:
        r_roster, d_roster = m.radiant_roster, m.dire_roster

        r_ratings = [engine.rating(p) for p in r_roster] or [engine.base_rating]
        d_ratings = [engine.rating(p) for p in d_roster] or [engine.base_rating]

        r_mean = sum(r_ratings) / len(r_ratings)
        d_mean = sum(d_ratings) / len(d_ratings)

        r_team = engine.team_rating(m.radiant_team_id)
        d_team = engine.team_rating(m.dire_team_id)

        r_played = [engine.matches_played(p) for p in r_roster] or [0]
        d_played = [engine.matches_played(p) for p in d_roster] or [0]

        rows.append(
            PlayerFeatureRow(
                match_id=m.match_id,
                as_of_timestamp=m.start_time,
                radiant_team_id=m.radiant_team_id,
                dire_team_id=m.dire_team_id,
                radiant_player_elo_mean=r_mean,
                dire_player_elo_mean=d_mean,
                radiant_player_elo_min=min(r_ratings),
                dire_player_elo_min=min(d_ratings),
                radiant_player_elo_max=max(r_ratings),
                dire_player_elo_max=max(d_ratings),
                radiant_roster_vs_team_delta=r_mean - r_team,
                dire_roster_vs_team_delta=d_mean - d_team,
                radiant_pair_synergy=synergy.mean_pair_synergy(r_roster),
                dire_pair_synergy=synergy.mean_pair_synergy(d_roster),
                radiant_player_matches_min=min(r_played),
                dire_player_matches_min=min(d_played),
                radiant_player_matches_mean=sum(r_played) / len(r_played),
                dire_player_matches_mean=sum(d_played) / len(d_played),
                radiant_win=m.radiant_win,
            )
        )

        # Обновление состояния — строго ПОСЛЕ фиксации признаков.
        engine.process(r_roster, d_roster, m.radiant_team_id, m.dire_team_id, m.radiant_win)
        synergy.observe(r_roster, m.radiant_win)
        synergy.observe(d_roster, not m.radiant_win)

    return rows
