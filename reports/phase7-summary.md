# PHASE 7 SUMMARY — Roster + Draft Signal Research

Дата: 2026-08-25. Продолжение `reports/phase6_5-summary.md`. Все цифры —
из `scripts/phase7_pipeline.py` (+ дополнительная roster temporal-stability
проверка, см. раздел 14). Реестр: `reports/experiments/phase7_roster_draft.json`.
Графики: `reports/figures/phase7_*.png`. `git commit 9d6e90d` (код Phase 7
поверх него в этой сессии). `random_seed=42` везде.

## 1. Objective

Проверить, добавляют ли roster (состав команды) и draft (picks/bans)
независимый predictive signal поверх **frozen baseline** Phase 6.5
(`Elo(K=16) + Form(window=3)`), НЕ изменяя сам baseline, с раздельными
режимами PRE_DRAFT (roster) и POST_DRAFT (draft) и честными,
subset-специфичными сравнениями (раздел 28 задания).

## 2. Baseline

Frozen, не менялся: `elo_difference` (K=16) + `form_3_difference`.
Переобучен в этой фазе ТОЛЬКО на новых, subset-специфичных train/test
границах (ROSTER_SET, DRAFT_SET) — сами признаки и гиперпараметры Elo
идентичны Phase 6.5.

## 3. Data coverage

`picks_bans`/`match_players` для непрерывного датасета 2021-2026 были
собраны ЭТОЙ фазой (`--with-draft-players`, новая опция
`OpenDotaExplorerSource`/`scripts/backfill_continuous.py`) — в начале фазы
покрытие было ~1% (наследие Phase 5, per-match `/matches/{id}`). Bulk-метод
через `/explorer` (picks_bans — колонка на `matches`, player-level — отдельная
explorer-таблица `player_matches`) поднял покрытие до:

| Год | Матчей | С picks_bans | С match_players |
|---|---:|---:|---:|
| 2021 | 16 071 | 15 914 (99.0%) | 16 071 (100.0%) |
| 2022 | 23 629 | 23 461 (99.3%) | 23 629 (100.0%) |
| 2023 | 29 625 | 29 462 (99.4%) | 29 625 (100.0%) |
| 2024 | 24 977 | 24 746 (99.1%) | 24 977 (100.0%) |
| 2025 | 18 709 | 18 323 (97.9%) | 18 709 (100.0%) |
| 2026 (частично) | 4 364 | 4 338 (99.4%) | 4 364 (100.0%) |

Точечная оптимизация (не архитектурное изменение): `upsert_match_players`/
`upsert_picks_bans` (`src/repositories/match_repository.py`) переписаны с
N отдельных `execute()` на один multi-row `execute()` на матч — throughput
вырос с ~24 до ~56 матчей/сек, полный backfill занял ~35 минут вместо
оценочных ~1.5 часов.

**`picks_bans.team` semantics проверена ЭМПИРИЧЕСКИ** (раздел 10 задания:
"не предполагай semantics из названий полей"), не по комментарию в коде:
сопоставлено 2000 пар `(picks_bans.team, picks_bans.hero_id)` с
`(match_players.is_radiant, match_players.hero_id)` того же героя того же
матча — **`team=0 ↔ is_radiant=True`, 2000/2000 совпадений, 0 расхождений.**

## 4. Roster reconstruction

`src/datasets/roster_features.py` — walk-forward, НЕ использует
`/teams/{id}/players` (all-time агрегат, Phase 5). "Активный состав"
команды = account_id, сыгравшие её ПОСЛЕДНИЙ матч; "состав текущего
матча" читается из `match_players` этого же матча (pre-draft информация в
реальности — составы объявляются до игры, не in-match статистика).

Проверено на реальных данных: **234 741 / 234 742 (100.0%) team-match
записей имеют полный состав из 5 игроков** — почти идеальное покрытие.

Leakage-safety: 4 adversarial-теста
(`tests/leakage/test_roster_features_leakage.py`) — смена состава
детектируется корректно, прошлые строки не меняются при мутации
будущих/текущих исходов, pre-match state НЕ включает состав текущего
матча. Все PASS.

## 5. Roster features

R1 (`roster_size`), R2 (`roster_matches_together` — число подряд матчей
ТЕКУЩЕГО состава, пре-матчевое состояние), R3 (`roster_age_days`), R4
(`player_continuity` — доля игроков, совпадающих с пре-матчевым составом),
R5 (`roster_changes` за 7/30/90 дней). Player-level статистика (KDA/GPM)
намеренно НЕ используется (раздел 8 задания).

## 6. Draft data

`src/datasets/draft_features.py` — walk-forward, все статистики (hero
winrate, team-hero history, pick popularity, hero matchup) считаются
ТОЛЬКО по матчам со `start_time` строго раньше текущего. **DRAFT_COMPLETE_SET
= 116 237 / 117 371 (99.0%) матчей** с полным drafts (5 пиков на сторону) —
основной continuous dataset НЕ тронут, это отдельная выборка (раздел 2
задания).

Leakage-safety: 4 adversarial-теста
(`tests/leakage/test_draft_features_leakage.py`), включая явную проверку
"hero_winrate матча N не учитывает исход самого матча N, даже если герой
там же встречается". Все PASS.

## 7. Draft features

- **D1 — hero strength**: mean win rate 5 героев команды, as-of prediction
  time (overall + patch-scoped вариант, D13).
- **D2 — team-hero history**: mean опыт/winrate команды именно с этими
  героями.
- **D3 — pick popularity**: mean историческая частота пика героев команды
  (прокси pick statistics; ban-специфичные признаки не реализованы в этой
  фазе — см. Limitations).
- **D4/D14 — matchup interaction**: mean историческая win rate герой-vs-герой
  по всем 5×5 парам пиков radiant/dire (НЕ вручную заданные "контры" —
  раздел 14: "если строишь hero A vs hero B, оцени его исторически").

## 8. Pre-draft experiments (MODE A = Elo+Form3+Roster)

ROSTER_COMPLETE_SET (n=111 230, TRAIN 77 861 / VAL 16 684 / TEST 16 685):

| Модель | Accuracy | Log Loss | Brier | ROC-AUC |
|---|---:|---:|---:|---:|
| Baseline (Elo+Form3) | 0.6028 | 0.6606 | 0.2343 | 0.6418 |
| + Roster Stability | 0.6028 | 0.6606 | 0.2343 | 0.6418 |
| + Roster Age | 0.6030 | 0.6606 | 0.2343 | 0.6419 |
| + Player Continuity | 0.6013 | 0.6606 | 0.2343 | 0.6416 |
| + All Roster | 0.6014 | 0.6606 | 0.2343 | 0.6418 |

Ни одна комбинация не улучшила baseline измеримо — различия (0.60-0.60%
accuracy) в пределах шума одного split.

## 9. Draft-aware experiments (MODE B = Elo+Form3+Draft)

DRAFT_COMPLETE_SET (n=110 170, TRAIN 77 119 / VAL 16 525 / TEST 16 526):

| Модель | Accuracy | Log Loss | Brier | ROC-AUC |
|---|---:|---:|---:|---:|
| DRAFT-0: Baseline | 0.6038 | 0.6600 | 0.2340 | 0.6434 |
| **DRAFT-1: + Hero Strength** | **0.6085** | 0.6573 | 0.2327 | 0.6498 |
| DRAFT-2: + Team-Hero History | 0.6066 | 0.6568 | 0.2325 | 0.6502 |
| DRAFT-3: + Pick Popularity | 0.6073 | 0.6567 | 0.2325 | 0.6500 |
| DRAFT-4: + Matchup Interaction | 0.6075 | **0.6565** | **0.2324** | **0.6504** |
| DRAFT-1b: + Hero Strength (patch-scoped) | 0.6060 | 0.6584 | 0.2332 | 0.6473 |

**Hero strength (D1) даёт наибольший прирост по accuracy**; полный стек
(DRAFT-4) — лучший по log_loss/Brier/ROC-AUC, но не по accuracy. Это
реальная, не противоречивая находка: разные признаки оптимизируют разные
аспекты качества прогноза. Patch-scoped hero strength (D1b) оказался ХУЖЕ
overall-версии — вероятно, per-patch выборка (тысячи, не десятки тысяч
матчей на патч) слишком шумная для надёжной оценки силы героя.

## 10. Ablation

См. разделы 8-9. Основной вывод: **roster ablation — плоская линия (нет
сигнала); draft ablation — измеримый, но не монотонный прирост**, основной
вклад от hero strength, дальнейшие уровни (team-hero history, pick
popularity, matchup) добавляют преимущественно к калибровке (log_loss),
не к accuracy.

## 11. Walk-forward

**Draft (MODE B) vs Baseline, DRAFT_SET, по годам** — expanding window,
из полного прогона:

| Год | n | Baseline acc | MODE B acc | Δ |
|---|---:|---:|---:|---:|
| 2022 | 22 502 | 0.579 | 0.583 | +0.004 |
| 2023 | 28 539 | 0.582 | 0.588 | +0.007 |
| 2024 | 23 538 | 0.579 | 0.586 | +0.007 |
| 2025 | 17 100 | 0.601 | 0.603 | +0.002 |
| 2026 (частично) | 3 948 | 0.620 | 0.624 | +0.004 |

**MODE B ≥ Baseline в КАЖДОМ из 5 лет** — консистентно положительный,
пусть и небольшой эффект.

**Roster (MODE A) vs Baseline, ROSTER_SET, по годам** (дополнительная
проверка, не в основном прогоне, воспроизводима отдельным вызовом):

| Год | n | Baseline acc | MODE A acc | Δ |
|---|---:|---:|---:|---:|
| 2022 | 22 668 | 0.579 | 0.577 | −0.002 |
| 2023 | 28 697 | 0.582 | 0.581 | −0.001 |
| 2024 | 23 742 | 0.579 | 0.580 | +0.001 |
| 2025 | 17 452 | 0.601 | 0.600 | −0.000 |
| 2026 (частично) | 3 972 | 0.619 | 0.616 | −0.003 |

**Знак Δ меняется случайно по годам (3 отрицательных, 1 около нуля, 1
слабо положительный)** — НЕТ консистентности, в отличие от draft. Это
согласуется с выводом раздела 8 (roster ablation) и статистикой ниже.

## 12. Statistical validation

Block bootstrap (`block_size=20`, `n_bootstrap=2000`, `seed=42`, тот же
метод, что Phase 6.5), McNemar exact test:

| Сравнение | Δaccuracy | 95% CI | Δlog_loss | 95% CI | McNemar p |
|---|---:|---|---:|---|---:|
| MODE A (roster) vs Baseline | −0.0016 | [−0.0037, +0.0005] | +0.0000 | [−0.0003, +0.0004] | 0.1364 |
| MODE B (draft) vs Baseline | **+0.0039** | [−0.0006, **+0.0087**] | **−0.0034** | **[−0.0045, −0.0023]** | 0.1046 |

**Roster**: оба CI пересекают 0 → эффект статистически неотличим от нуля.
**Draft**: log_loss CI **полностью отрицателен** (MODE B статистически
значимо лучше по log_loss) — но accuracy CI **едва пересекает 0** снизу
(нижняя граница −0.0006, почти значимо, но формально не исключает 0), и
McNemar p=0.1046 (> 0.05, формально не значим). **Смешанная картина**, не
однозначное "да" — раздел 21 задания прямо предупреждает не судить по
одному p-value; см. раздел 21 ниже для комплексной оценки.

## 13. Calibration

Не строился отдельный calibration curve для MODE A/B в этой фазе (roster
не показал сигнала, calibration для него неинформативна; для draft —
log_loss/Brier уже улучшились у DRAFT-4 сильнее, чем у DRAFT-1, что
косвенно говорит об улучшении калибровки, не только дискриминации) —
кандидат для отдельной проверки, если draft будет развиваться дальше
(раздел 20 ниже).

## 14. Temporal stability

См. раздел 11 (walk-forward) — та же таблица отвечает на вопрос temporal
stability для обоих режимов. Draft: 5/5 лет положительный Δ. Roster: знак
нестабилен.

## 15. Patch stability

DRAFT_SET TEST, MODE B vs Baseline, патчи с n≥100:

| patch_id | n | Baseline acc | MODE B acc | Δ |
|---:|---:|---:|---:|---:|
| 57 | 2 629 | 0.6132 | 0.6109 | −0.0023 |
| 58 | 9 735 | 0.5957 | 0.6008 | +0.0051 |
| 59 | 1 934 | 0.6179 | 0.6246 | +0.0067 |
| 60 | 2 228 | 0.6158 | 0.6194 | +0.0036 |

3 из 4 патчей — положительный Δ, один (57, самый маленький n в тестовом
срезе) — слабо отрицательный. Не противоречит выводу о наличии слабого,
но не идеально стабильного эффекта.

## 16. Error analysis

MODE B vs Baseline, DRAFT_SET TEST (n=16 526): **810 случаев**, где
baseline ошибался, а MODE B предсказал верно; **745 случаев** — обратный
эффект (draft "испортил" верный прогноз baseline). Net effect: **+65** в
пользу MODE B — совпадает по знаку с Δaccuracy (+0.0039 × 16 526 ≈ +64,
согласовано). И "выигрышей", и "проигрышей" от добавления draft —
сотни в обе стороны, что типично для признака с небольшим, но не
доминирующим эффектом — не единичные показательные случаи, которые стоило
бы разбирать вручную по одному.

## 17. Leakage audit

8 adversarial-тестов (4 roster + 4 draft), все PASS: смена исхода
прошлого/будущего матча не влияет на более ранние строки, признаки первого
матча (без истории) — `None`/нейтральны, а не тайно unlocked, hero
winrate/roster continuity не используют результат текущего матча. Плюс
проверка `available_at <= prediction_timestamp` реализована структурно
(walk-forward: состояние читается ДО обновления, раздел 29 задания) и
подтверждена этими тестами, не только "по построению". `pytest tests/` —
**61/61 PASS** (53 из Phase 5-6.5 + 8 новых).

## 18. Limitations

1. **Ban-specific frequency features не реализованы** — раздел 10 задания
   предлагает pick+ban statistics как один уровень (DRAFT-3); реализована
   только pick-часть. Кандидат для отдельного эксперимента.
2. **Roster features — простые агрегаты** (разница count/age/continuity) —
   не исключено, что более сложные формулировки (например,
   roster-weighted Elo, «сыгранность» через синергию конкретных пар
   игроков) показали бы иной результат; текущий null-результат относится
   К ЭТИМ КОНКРЕТНЫМ признакам, не к "roster в принципе бесполезен".
3. **Draft accuracy CI формально пересекает 0** (нижняя граница −0.0006) —
   вывод "signal exists" опирается на СОВОКУПНОСТЬ доказательств
   (log_loss CI, walk-forward 5/5, error analysis net-positive), не на
   одном тесте — это осознанное применение раздела 21 задания, но означает,
   что вывод менее железобетонный, чем находки Phase 6.5 про Elo/Form.
4. **DRAFT-2 (team-hero history) снизил accuracy** относительно DRAFT-1 —
   не исследовано глубоко, почему (возможна коллинеарность с hero
   strength — команда, часто играющая на герое, коррелирует с тем, что
   герой у неё "сильный").
5. **Patch-scoped hero strength (D1b) хуже overall** — вероятно, sample
   size per patch недостаточен; не пробовалась promежуточная схема
   (recency-weighted, не жёсткое разбиение по патчу, раздел 13 задания
   предлагал это как альтернативу).
6. **Calibration/confidence buckets для MODE A/B не построены** (раздел 13
   формата отчёта) — не приоритизировано в рамках эффорт-бюджета этой
   фазы, roster не показал сигнала (неинформативно), draft — кандидат для
   Phase 8, если продолжится.

## 19. Decision

| Источник | Решение | Обоснование |
|---|---|---|
| **Roster (stability/age/continuity, текущая формулировка)** | **REMOVE / RESEARCH FURTHER** (не KEEP) | Оба CI (accuracy, log_loss) пересекают 0; McNemar p=0.14; walk-forward знак нестабилен по годам (3 отрицательных, 1 нейтральный, 1 положительный) — НЕ удовлетворяет критерию раздела 34 ("большинство из": out-of-sample improvement, positive effect size, CI supports, walk-forward consistency) |
| **Draft — Hero Strength (D1)** | **KEEP (weak-to-moderate signal), RESEARCH FURTHER для уточнения** | log_loss CI полностью в пользу draft; walk-forward 5/5 лет положительный; error analysis net +65; НО accuracy CI и McNemar формально не достигают порога значимости — удовлетворяет БОЛЬШИНСТВУ (не всем) критериев раздела 34 |
| **Draft — Team-Hero History, Pick Popularity, Matchup** | **RESEARCH FURTHER** | Улучшают log_loss/Brier/ROC-AUC поверх DRAFT-1, но не accuracy; вклад скромнее hero strength; не изучена причина немонотонности (Limitations, п.4) |

## 20. Recommended next experiment

1. **Draft — DRAFT-1 (hero strength) как следующий кандидат в production
   feature set**, но ТОЛЬКО после калибровочной проверки (reliability
   diagram для MODE B, п.13 Limitations) и, желательно, подтверждения на
   более длинном walk-forward окне (2027, когда появятся новые данные) —
   текущий результат реален, но не "железобетонный" по одному тесту
   (accuracy CI).
2. **Разобраться, почему team-hero history снижает accuracy** относительно
   чистого hero strength — возможна коллинеарность, кандидат для
   feature-selection эксперимента (L1-регуляризация LogReg или
   permutation importance).
3. **Roster — не отбрасывать концепцию полностью**, но текущие простые
   агрегатные признаки не работают; следующая итерация — roster-weighted
   Elo (Elo, взвешенный по тому, сколько игроков состава реально
   постоянны) вместо отдельных стабильность/возраст/continuity фич.
4. **Ban-statistics** (раздел 10 задания, не реализовано в Phase 7) —
   дешёвое расширение существующего `_DraftStatsTracker`.

---

# PHASE 7 SUMMARY

## Q1 — Насколько хорошо можно предсказывать матч ДО драфта?

Baseline (Elo K=16 + Form3) на новых, более крупных continuous
evaluation subset'ах (Phase 7 передел): **accuracy 0.60-0.60, ROC-AUC
0.64-0.64, log_loss 0.66** — консистентно со всей историей Phase 6.5.
Roster поверх этого НЕ добавляет измеримого улучшения — на сегодня
pre-draft прогноз практически полностью объясняется Elo+Form, без вклада
от простых roster-агрегатов.

## Q2 — Добавляет ли знание состава независимый signal?

**Нет, не в текущей формулировке признаков.** Оба статистических теста
(block bootstrap, McNemar) не отличают эффект от нуля, walk-forward не
показывает консистентного направления по годам. Это НЕ доказывает, что
состав команды принципиально неважен для результата матча — только что
ПРОСТЫЕ агрегаты (stability/age/continuity), проверенные в этой фазе, не
несут сигнала поверх уже сильного Elo+Form baseline.

## Q3 — Насколько меняется prediction после полного draft?

**Скромно, но измеримо и в основном консистентно.** Accuracy +0.39pp,
log_loss −0.0034 (статистически значимо по CI), эффект положителен в 5/5
walk-forward лет и в 3/4 патчах. Основной вклад — от исторической силы
выбранных героев (hero strength), не от team-специфичной истории или
взаимодействий героев (те добавляют к калибровке, не к точности).

## Следующее наиболее ценное направление

**Draft (hero strength), не roster** — в текущем виде. Roster требует
переосмысления признаков (не просто "убрать", а "попробовать другую
формулировку"), прежде чем возвращаться к нему.

Останавливаюсь здесь. Жду решения по дальнейшему шагу.
