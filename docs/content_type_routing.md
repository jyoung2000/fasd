# Content-Type Routing Flowchart

This doc is the **single source of truth** for how a user's content-type
selection flows through the reframe pipeline, end-to-end. It was written
after Phase 1 of the v2 parity work fixed the plumbing gap where the UI
dropdown sent strings (`"gameplay"`, `"movie"`, `"podcast"`) that didn't
match the `ContentType` enum values (`"gaming"`, `"narrative"`,
`"podcast"`) — so only `"podcast"` worked by coincidence.

Future contributors: if you add a new UI dropdown value, walk this
flowchart and add the new row to every layer.

---

## The flowchart

```
┌─────────────────────────────────────────────────────────────────┐
│ 1.  USER picks a value in the Upload.jsx dropdown               │
│     (frontend/src/pages/Upload.jsx ~line 1126)                  │
│                                                                 │
│     "podcast", "debate", "vlog", "narrative", "anime",          │
│     "music_video", "gameplay", "gameplay_moba",                 │
│     "gameplay_tps", "gameplay_racing", "stream", "sports",      │
│     "" (Auto-detect)                                            │
│                                                                 │
│     + optional sub-dropdown values:                             │
│       - anime_subtype: "action" | "dialogue" | "slice_of_life"  │
│       - music_subtype: "performance" | "narrative" | "lyric"    │
│       - game_type: "valorant" | "league_of_legends" | ...       │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼ POST /api/upload/init
                        │       (chunked) or POST /api/upload
                        │
┌─────────────────────────────────────────────────────────────────┐
│ 2.  UPLOAD ROUTER stores the fields on JobResult                │
│     (backend/models.py line 163 — content_type_override,        │
│      game_type, anime_subtype, music_subtype)                   │
│     (backend/routers/chunked_upload.py + upload.py)             │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼ run_analysis(job_id)
                        │
┌─────────────────────────────────────────────────────────────────┐
│ 3.  PIPELINE reads JobResult.content_type_override              │
│     (backend/services/pipeline.py ~line 1223)                   │
│                                                                 │
│     _content_override = job.content_type_override               │
│     _game_type        = job.game_type                           │
│     _anime_subtype    = job.anime_subtype                       │
│     _music_subtype    = job.music_subtype                       │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼ normalize_ui_content_type(_content_override)
                        │
┌─────────────────────────────────────────────────────────────────┐
│ 4.  NORMALIZER maps UI token → ContentType + flags              │
│     (backend/services/content_type_strings.py)                  │
│                                                                 │
│     _UI_TO_ENUM dict:                                           │
│       "gameplay"       → GAMING,   gameplay_fastpath=True       │
│       "gameplay_fps"   → GAMING,   gameplay_subtype="fps"       │
│       "gameplay_moba"  → GAMING,   gameplay_subtype="moba"      │
│       "gameplay_tps"   → GAMING,   gameplay_subtype="tps"       │
│       "gameplay_racing"→ GAMING,   gameplay_subtype="racing"    │
│       "stream"         → GAMING,   gameplay_subtype="stream"    │
│                                                                 │
│       "podcast"        → PODCAST                                │
│       "debate"         → PODCAST,  panel=True                   │
│       "panel"          → PODCAST,  panel=True                   │
│       "interview"      → PODCAST                                │
│                                                                 │
│       "vlog"           → VLOG                                   │
│       "narrative"      → NARRATIVE                              │
│       "cinematic"      → NARRATIVE                              │
│       "movie"          → NARRATIVE (legacy)                     │
│                                                                 │
│       "anime"          → ANIME,    animated=True                │
│       "cartoon"        → ANIME,    animated=True                │
│       "music_video"    → MUSIC_VIDEO                            │
│       "sports"         → SPORTS                                 │
│                                                                 │
│       "", "auto", "unknown", <invalid> → None                   │
│                                                                 │
│     Returns NormalizedContentType with:                         │
│       - content_type: ContentType enum                          │
│       - is_multi_speaker_panel: bool                            │
│       - is_animated: bool                                       │
│       - gameplay_subtype: str | None                            │
│       - is_gameplay_fastpath: bool                              │
│       - raw: original token                                     │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼ classify_content(metadata=_classifier_metadata)
                        │       where _classifier_metadata includes
                        │       content_type_override / anime_subtype /
                        │       music_subtype / game_type
                        │
┌─────────────────────────────────────────────────────────────────┐
│ 5.  CLASSIFIER short-circuits the heuristic path when a valid   │
│     override is present                                         │
│     (backend/services/content_classifier.py::classify_content)  │
│                                                                 │
│     profile = ContentProfile(                                   │
│         content_type    = normalized.content_type.value,        │
│         confidence      = 1.0,                                  │
│         is_multi_speaker_panel = normalized.is_multi_speaker_panel,│
│         is_animated     = normalized.is_animated,               │
│         anime_subtype   = (from metadata) if animated,          │
│         music_subtype   = (from metadata) if MUSIC_VIDEO,       │
│         gameplay_subtype = normalized.gameplay_subtype,         │
│         game_type       = (from metadata, always),              │
│     )                                                           │
│                                                                 │
│     When the override is missing / invalid, classify_content    │
│     falls through to the heuristic path that picks a type       │
│     based on face geometry / cut rate / scene keywords.         │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼ ContentProfile passed into the segmenter
                        │
┌─────────────────────────────────────────────────────────────────┐
│ 6.  REFRAME SEGMENTER reads the profile + sub-type flags and    │
│     activates per-content-type tuning                           │
│     (backend/services/reframe_segmenter.py)                     │
│                                                                 │
│     Stage 1-7:   intent tracking uses                           │
│                    CONTENT_TYPE_CONFIG[profile.content_type]    │
│     Stage 7b:    anime anchor override (Phase 6) —              │
│                    profile.is_animated AND CLIPAI_ANIME_ANCHOR  │
│     Stage 7c:    gameplay tracker override (Phase 7) —          │
│                    profile.gameplay_subtype AND                 │
│                    CLIPAI_GAMEPLAY_TRACKER                      │
│     Stage 7d:    STREAM layout routing (Phase 7) —              │
│                    profile.gameplay_subtype == "stream"         │
│                    (NO flag — editorial decision)               │
│     Stage 8:     lead room (Phase 4) —                          │
│                    CONTENT_TYPE_CONFIG.apply_lead_room          │
│                    + anime_action_multiplier via                │
│                    profile.anime_subtype                        │
│     Stage 9b:    editorial prior (Phase 8) —                    │
│                    applies_to_profile(profile) AND              │
│                    CLIPAI_EDITORIAL_PRIOR                       │
│     Stage 10a:   multi-region LP (Phase 3) —                    │
│                    CLIPAI_MULTI_REGION_LP                       │
│     Stage 10:    L1 camera path solve                           │
│     Stage 10c:   Phase 4 post-process — lead room + thirds bias │
│                    + anime action multiplier on motion paths    │
│     Stage 11:    music-video beat snap (Phase 5) —              │
│                    profile.content_type == "music_video"        │
│                    AND CLIPAI_MUSIC_BEAT_SNAP                   │
│                    AND music_beat_grid is populated             │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼ classify_clip(content_profile=profile)
                        │       (called from plan_layout)
                        │
┌─────────────────────────────────────────────────────────────────┐
│ 7.  CLIP-LEVEL CLASSIFIER maps ContentType → ClipContentType    │
│     (backend/services/content_classifier.py::classify_clip)     │
│                                                                 │
│     Base mapping (_CONTENT_TYPE_MAP):                           │
│       PODCAST      → TALKING_HEAD                               │
│       VLOG         → TALKING_HEAD                               │
│       ANIME        → ANIMATION                                  │
│       MUSIC_VIDEO  → MUSIC_VIDEO                                │
│       GAMING       → GAMEPLAY (overridden by subtype below)     │
│       NARRATIVE    → GENERIC (promoted to CINEMATIC_DIALOGUE    │
│                                when dialogue signal fires)      │
│       SPORTS       → GENERIC                                    │
│                                                                 │
│     Sub-type overrides (applied AFTER base mapping):            │
│       gameplay_subtype="fps"     → GAMEPLAY                     │
│       gameplay_subtype="moba"    → GAMEPLAY_MOBA                │
│       gameplay_subtype="tps"     → GAMEPLAY_TPS                 │
│       gameplay_subtype="racing"  → GAMEPLAY_RACING              │
│       gameplay_subtype="stream"  → STREAM                       │
│                                                                 │
│       is_multi_speaker_panel=True → MULTI_SPEAKER_PANEL         │
│                                      (highest priority)        │
│                                                                 │
│       is_cinematic_dialogue=True → CINEMATIC_DIALOGUE           │
│                                                                 │
│       is_animated=True:                                         │
│         anime_subtype="action"         → ANIMATION              │
│         anime_subtype="dialogue"       → ANIMATION_DIALOGUE     │
│         anime_subtype="slice_of_life"  → ANIMATION_DIALOGUE     │
│         no subtype + base was GENERIC / TALKING_HEAD            │
│                                        → ANIMATION_DIALOGUE     │
│         no subtype + base was ANIMATION                         │
│                                        → ANIMATION              │
└───────────────────────┬─────────────────────────────────────────┘
                        │
                        ▼ ClipContentType feeds into:
                        │
┌─────────────────────────────────────────────────────────────────┐
│ 8.  TUNING LAYERS consume ClipContentType                       │
│                                                                 │
│     camera_solver.py               — per-clip solver tuning     │
│     required_regions.py            — weight caps                │
│     layout_engine.py               — STACKED_GAMEPLAY pick      │
│     multi_region_layout.py         — fallback choice per type   │
│     thirds_bias.py::applies_to_    — gate on eligible types     │
│        profile                                                  │
│     editorial_prior.py::applies_   — gate on dialogue types     │
│        to_profile                                               │
│                                                                 │
│     The per-ClipContentType logic is the single-source lookup   │
│     for "what does this clip want the camera to do":            │
│       TALKING_HEAD         — speaker tracker, no thirds bias on │
│                              debate panels                      │
│       MULTI_SPEAKER_PANEL  — symmetric split/wide fallback      │
│       CINEMATIC_DIALOGUE   — thirds bias + lead room            │
│       ANIMATION            — anime shot cadence + action center │
│       ANIMATION_DIALOGUE   — like CINEMATIC_DIALOGUE + anime    │
│                              face detector                     │
│       MUSIC_VIDEO          — beat snap + pulse cuts             │
│       GAMEPLAY             — FPS center crosshair               │
│       GAMEPLAY_MOBA        — wider safe-zone center             │
│       GAMEPLAY_TPS         — character offset (50, 45)          │
│       GAMEPLAY_RACING      — lower third (50, 65)               │
│       STREAM               — STACKED_GAMEPLAY layout            │
│       GENERIC              — default tuning                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Summary table — UI → ContentType → ClipContentType

Fully populated table of every UI token that Upload.jsx can emit,
matching `backend/services/content_type_strings._UI_TO_ENUM` and
the Phase 2 dropdown.

| UI token | `ContentType` | flags | `ClipContentType` (typical) |
|---|---|---|---|
| `""` / `"auto"` | heuristic | — | (heuristic) |
| `podcast` | `PODCAST` | — | `TALKING_HEAD` |
| `debate` / `panel` | `PODCAST` | `is_multi_speaker_panel` | `MULTI_SPEAKER_PANEL` |
| `interview` | `PODCAST` | — | `TALKING_HEAD` |
| `vlog` | `VLOG` | — | `TALKING_HEAD` |
| `narrative` / `cinematic` | `NARRATIVE` | — | `GENERIC` (or `CINEMATIC_DIALOGUE`) |
| `movie` (legacy) | `NARRATIVE` | — | same as `narrative` |
| `anime` / `cartoon` | `ANIME` | `is_animated` | `ANIMATION` or `ANIMATION_DIALOGUE` (via subtype) |
| `music_video` | `MUSIC_VIDEO` | — | `MUSIC_VIDEO` |
| `gameplay` (legacy) | `GAMING` | `gameplay_subtype="fps"`, fastpath | `GAMEPLAY` |
| `gameplay_fps` | `GAMING` | `gameplay_subtype="fps"`, fastpath | `GAMEPLAY` |
| `gameplay_moba` | `GAMING` | `gameplay_subtype="moba"`, fastpath | `GAMEPLAY_MOBA` |
| `gameplay_tps` | `GAMING` | `gameplay_subtype="tps"`, fastpath | `GAMEPLAY_TPS` |
| `gameplay_racing` | `GAMING` | `gameplay_subtype="racing"`, fastpath | `GAMEPLAY_RACING` |
| `stream` | `GAMING` | `gameplay_subtype="stream"`, **not** fastpath | `STREAM` |
| `sports` | `SPORTS` | — | `GENERIC` |

Notes:
- `is_gameplay_fastpath=True` triggers the `pipeline._is_gameplay`
  branch that currently skips face tracking entirely. Stream is
  deliberately NOT a fast-path candidate — the facecam needs face
  tracking.
- `is_multi_speaker_panel=True` short-circuits the dialogue /
  animation branches in `classify_clip` because a seated panel
  is a panel regardless of whether the raw classifier guessed
  `podcast` / `narrative` / `vlog` / `generic`.
- `is_animated=True` + `anime_subtype="action"` explicitly OPTS
  OUT of the Phase 4 thirds bias and Phase 8 editorial prior.
  Action anime compositions are choreographed in source.

---

## Feature flag inventory

Every v2 feature ships behind a flag. The ``Code default`` column is
the value used when the env var is unset; ``Effective?`` tells you
whether flipping the flag actually changes runtime behavior. A flag
is "Yes (live)" when its module has ≥1 production call site outside
its defining file; it is "No (dormant)" when the module is defined
but no other pipeline code invokes it.

Last reconciled: Week 1 flag audit (post-v4 universal reframing).
See ``docs/reframing_autoflip_parity.md`` "Week 1 flag audit" section
for the validation record.

| Flag | Phase | Code default | Effective? | Gates |
|---|---|---|---|---|
| `USE_CONTENT_AWARE_REFRAME` | 0+ | ON | Yes (live) | per-content-type tuning in the segmenter (Stage 8 lead-room, etc) |
| `CLIPAI_MULTI_REGION_LP` | 3 | ON (Week 1) | Yes (live) | Stage 10a multi-region LP fit-check |
| `CLIPAI_GAZE_LEAD_ROOM_V2` | 4 | ON | Yes (live) | Stage 8 + Stage 10c continuous-yaw lead room |
| `CLIPAI_THIRDS_BIAS` | 4 | ON | Yes (live) | Stage 8 + Stage 10c thirds offset |
| `CLIPAI_MUSIC_BEAT_SNAP` | 5 | ON | Yes (live) | Stage 11 music-video beat snap + pulse cuts |
| `CLIPAI_ANIME_SHOT_DETECTOR` | 6 | OFF | **No (dormant)** | module defined but 0 production call sites; Week 2 wires it into `shot_detector.py` |
| `CLIPAI_ANIME_FACE_DETECTOR` | 6 | OFF | **No (dormant)** | module defined but 0 production call sites; Week 2 wires it into `face_detector.py` |
| `CLIPAI_ANIME_ANCHOR` | 6 | ON | Yes (live) | Stage 7b anime saliency anchor override (pipeline.py + reframe_segmenter.py) |
| `CLIPAI_ANIME_CHARACTER_CLUSTERING` | 6 follow-up | OFF | **No (dormant)** | module defined but 0 production call sites; Week 2 wires it into `face_registry.py` |
| `CLIPAI_GAMEPLAY_TRACKER` | 7 | ON | Yes (live) | Stage 7c gameplay subject tracker override |
| `CLIPAI_EDITORIAL_PRIOR` | 8 | ON | Yes (live) | Stage 9b editorial J/L cuts + listener holds + reaction beats |

### Dormant code warning

The three flags marked **No (dormant)** are gates on modules that
exist in ``backend/services/`` but are not called from anywhere in
the pipeline (grep for their public symbols returns only the defining
module, the ``validate_v2_phases.py`` env-var setup, and per-module
unit tests). Flipping them has no runtime effect today. The Week 2
plan (``week2_anime_wiring_and_sports_subtype.md``) wires them. Until
then, anime content relies on ``CLIPAI_ANIME_ANCHOR`` (saliency-based
fallback) for subject tracking.

The regression guard ``backend/tests/test_dormant_flags_labeled.py``
asserts zero call sites for each dormant module, so any future commit
that wires one will fail the test and force a coordinated update
(flip the default + update this table + run ``validate_v2_phases``).

---

## How to add a new UI token

1. **Add the dropdown option in `Upload.jsx`** under the right
   optgroup.

2. **Add the token to `_UI_TO_ENUM`** in
   `backend/services/content_type_strings.py` with the correct
   `ContentType` + flags bundle. If it introduces a new flag
   semantic (beyond panel / animated / gameplay_subtype), add
   the field to `NormalizedContentType`.

3. **Add a row to the routing matrix test** in
   `backend/tests/test_content_routing_matrix.py`. The
   parametrized test asserts the full chain:
   normalize → classify_content → classify_clip.

4. **If the token needs a new `ClipContentType`**, add it to
   `content_classifier.ClipContentType` and to the sub-type
   override block in `classify_clip`. Mirror the Phase 7
   gameplay sub-type pattern.

5. **If the token introduces per-content-type tuning**, add a
   `CONTENT_TYPE_CONFIG` entry in `content_type_config.py`.

6. **Walk the per-phase gate functions** to decide whether the
   new type should opt in / out:
   - `thirds_bias.applies_to_profile`
   - `editorial_prior.applies_to_profile`
   - any phase-specific flag gating in the segmenter

7. **Update this doc** — add a row to the summary table and
   note any new flag semantics.

---

## Plumbing regression guard

`backend/tests/test_content_type_override_plumbing.py` is the
canary. It AST-walks `pipeline.py` and asserts:

- Every UI token round-trips through the normalizer to the
  right `ContentType` enum value.
- The pipeline's gameplay fast-path uses `is_gameplay_override()`
  instead of a bare `== "gameplay"` literal.
- The classifier receives `content_type_override` via the
  `_classifier_metadata` dict (the Phase 1 plumbing fix).

Run it before merging any content-type-routing change:

```bash
pytest backend/tests/test_content_type_override_plumbing.py \
       backend/tests/test_content_routing_matrix.py
```

Both files together lock the contract all the way from the UI
dropdown down to the segmenter's per-content-type tuning.
