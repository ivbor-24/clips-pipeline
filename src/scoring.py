"""
TASK-04: Semantic Scoring Engine Module

Implements heuristic scoring of transcript segments based on:
1. Term density (KeyBERT, or statistical extraction without a model)
2. Definition markers (e.g., "это", "называется", "представляет собой")
3. Rhetorical markers (questions, emphasis)
4. Self-containment (low pronoun ratio, complete thoughts)

Inputs:
- artifacts/transcript.json
- config/config.yaml (scoring section)

Outputs:
- artifacts/scored_segments.json (list of scored clips with metadata)
"""

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import structlog

from src.config import Config, LLMConfig, ScoringConfig
from src.notices import add_notice, clear_notices

LLM_FALLBACK_MESSAGE = (
    "The LLM failed, so clips were picked by simple heuristics and are likely worse. "
    "Check the model and the GPU (just check-gpu), then run the job again."
)

logger = structlog.get_logger("scoring")

# Number of transcript segments sent to the LLM per analysis batch.
# Keeps the JSON response well under max_tokens (large transcripts get
# truncated mid-JSON and fail to parse when sent all at once).
LLM_SEGMENT_BATCH_SIZE = 15


# Russian definition markers
DEFINITION_MARKERS_RU = [
    r"\bэто\b",  # "this is"
    r"\bназывается\b",  # "is called"
    r"\bпредставляет собой\b",  # "represents"
    r"\bявляется\b",  # "is"
    r"\bозначает\b",  # "means"
    r"\bопределяется как\b",  # "is defined as"
    r"\bможно определить как\b",  # "can be defined as"
    r"\bпо сути\b",  # "essentially"
    r"\bдругими словами\b",  # "in other words"
    r"\bто есть\b",  # "that is"
    r"\bа именно\b",  # "namely"
]

# English definition markers
DEFINITION_MARKERS_EN = [
    r"\bis\s+(a|an|the)?\s*\w+",  # "is a/an/the..."
    r"\bare\s+(a|an|the)?\s*\w+",  # "are a/an/the..."
    r"\brefers\s+to\b",
    r"\bmeans\b",
    r"\bdefined\s+as\b",
    r"\bdefinition\s+of\b",
    r"\bin\s+other\s+words\b",
    r"\bthat\s+is\b",
    r"\bnamely\b",
    r"\bessentially\b",
]

# Rhetorical/question markers
RHETORICAL_MARKERS_RU = [
    r"\?",  # Question mark
    r"\bпочему\b",  # "why"
    r"\bкак\b",  # "how"
    r"\bчто\b",  # "what"
    r"\bгде\b",  # "where"
    r"\bкогда\b",  # "when"
    r"\bкакой\b",  # "which/what kind"
    r"\bзачем\b",  # "for what purpose"
    r"\bобратите внимание\b",  # "note that"
    r"\bважно\b",  # "important"
    r"\bключевой\b",  # "key"
    r"\bглавный\b",  # "main"
]

RHETORICAL_MARKERS_EN = [
    r"\?",  # Question mark
    r"\bwhy\b",
    r"\bhow\b",
    r"\bwhat\b",
    r"\bwhere\b",
    r"\bwhen\b",
    r"\bwhich\b",
    r"\bwho\b",
    r"\bnote\s+that\b",
    r"\bimportant\b",
    r"\bkey\b",
    r"\bcrucial\b",
    r"\bmain\b",
]

# Pronouns that reduce self-containment score
PRONOUNS_RU = [
    r"\bон\b",
    r"\bона\b",
    r"\boно\b",
    r"\bони\b",  # he/she/it/they
    r"\bего\b",
    r"\bеё\b",
    r"\bих\b",  # his/her/their (genitive)
    r"\bему\b",
    r"\bей\b",
    r"\bим\b",  # him/her/them (dative)
    r"\bним\b",
    r"\bней\b",  # him/her (prepositional)
    r"\bтот\b",
    r"\bэта\b",
    r"\bэто\b",
    r"\bэти\b",  # that/this/these
    r"\bтакой\b",
    r"\bтакая\b",
    r"\bтакое\b",
    r"\bтакие\b",  # such
]

PRONOUNS_EN = [
    r"\bit\b",
    r"\bthey\b",
    r"\bthem\b",
    r"\bhe\b",
    r"\bshe\b",
    r"\bhis\b",
    r"\bher\b",
    r"\btheir\b",
    r"\bits\b",
    r"\bthat\b",
    r"\bthis\b",
    r"\bthese\b",
    r"\bthose\b",
    r"\bsuch\b",
]

# Common scientific terms patterns (for fallback when KeyBERT not available)
TERM_PATTERNS = [
    r"\b[A-Z][A-Z]+\b",  # Acronyms like DNA, RNA, AI
    r"\b\w+(-\w+)+\b",  # Hyphenated compounds
    r"\bне\w+\b",  # Russian negations
]


@dataclass
class ScoredSegment:
    """Represents a scored segment with metadata."""

    id: str
    start: float
    end: float
    text: str
    score: float
    tags: List[str] = field(default_factory=list)
    self_contained: bool = True
    term_density: float = 0.0
    definition_score: float = 0.0
    rhetorical_score: float = 0.0
    self_containment_score: float = 1.0


class ScoringError(Exception):
    """Custom exception for scoring errors."""

    pass


def load_transcript(transcript_path: Path) -> List[Dict[str, Any]]:
    """Load transcript from JSON file."""
    if not transcript_path.exists():
        raise ScoringError(f"Transcript file not found: {transcript_path}")

    with open(transcript_path, "r", encoding="utf-8") as f:
        transcript = json.load(f)

    logger.info("Transcript loaded", path=str(transcript_path), segments=len(transcript))
    return transcript


def detect_language(text: str) -> str:
    """Detect language of text (simple heuristic based on Cyrillic characters)."""
    cyrillic_chars = sum(1 for c in text if "а" <= c <= "я" or "А" <= c <= "Я" or c == "ё")
    total_alpha = sum(1 for c in text if c.isalpha())

    if total_alpha == 0:
        return "unknown"

    cyrillic_ratio = cyrillic_chars / total_alpha
    if cyrillic_ratio > 0.5:
        return "ru"
    else:
        return "en"


def extract_terms_statistical(text: str, top_n: int = 10) -> List[str]:
    """
    Extract terms using simple statistical methods (fallback when KeyBERT unavailable).

    Uses:
    - Noun extraction (capitalized words, long words)
    - TF-like scoring based on frequency and length
    """
    # Simple word frequency analysis
    words = re.findall(r"\b\w+\b", text.lower())

    # Filter out common stop words (Russian + English)
    stop_words = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "и",
        "в",
        "во",
        "на",
        "с",
        "со",
        "за",
        "под",
        "над",
        "из",
        "от",
        "до",
        "по",
        "для",
        "при",
        "через",
        "между",
        "к",
        "у",
        "о",
        "об",
        "без",
        "это",
        "что",
        "как",
        "так",
        "же",
        "ли",
        "бы",
        "мне",
        "тебе",
        "себе",
    }

    # Count word frequencies
    word_freq = {}
    for word in words:
        if word not in stop_words and len(word) > 3:
            word_freq[word] = word_freq.get(word, 0) + 1

    # Score by frequency * length (longer words tend to be more specific)
    scored_words = [(word, freq * len(word)) for word, freq in word_freq.items()]
    scored_words.sort(key=lambda x: x[1], reverse=True)

    return [word for word, _ in scored_words[:top_n]]


def try_keybert_extraction(text: str, top_n: int = 10) -> Optional[List[str]]:
    """Try to extract terms using KeyBERT, return None if unavailable."""
    try:
        from keybert import KeyBERT

        from src.model_registry import KEYBERT_MODEL

        kw_model = KeyBERT(model=KEYBERT_MODEL)
        keywords = kw_model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 2),
            stop_words=None,  # KeyBERT handles this internally
            top_n=top_n,
        )
        # KeyBERT returns list of (keyword, score) tuples
        return [kw[0] for kw in keywords]
    except ImportError:
        return None
    except Exception as e:
        logger.warning("KeyBERT extraction failed", error=str(e))
        return None


def extract_terms(text: str, method: str = "keybert", top_n: int = 10) -> List[str]:
    """
    Extract key terms from text.

    Args:
        text: Input text
        method: "keybert" or "statistical"
        top_n: Number of top terms to return

    Returns:
        List of extracted terms
    """
    # Try KeyBERT first if requested
    if method == "keybert":
        result = try_keybert_extraction(text, top_n)
        if result is not None:
            return result

    # Fallback to statistical extraction
    logger.debug("Using statistical term extraction as fallback")
    return extract_terms_statistical(text, top_n)


def compute_term_density_score(
    text: str, terms: List[str], method: str = "keybert"
) -> Tuple[float, List[str]]:
    """
    Compute term density score for a segment.

    Higher density of meaningful terms = higher score.

    Returns:
        Tuple of (score 0-1, list of found terms)
    """
    if not text.strip():
        return 0.0, []

    text_lower = text.lower()
    found_terms = []

    for term in terms:
        term_lower = term.lower()
        if term_lower in text_lower:
            found_terms.append(term)

    # Normalize by text length
    word_count = len(text.split())
    if word_count == 0:
        return 0.0, []

    # Score: ratio of unique terms to words, capped at 1.0
    # Typical good segments have ~5-15% term density
    term_density = len(found_terms) / word_count
    score = min(1.0, term_density * 10)  # Scale so 10% density = 1.0

    return score, found_terms


def compute_definition_score(text: str) -> float:
    """
    Compute definition marker score.

    Detects phrases that indicate definitions or explanations.

    Returns:
        Score 0-1 based on presence of definition markers
    """
    if not text.strip():
        return 0.0

    lang = detect_language(text)

    if lang == "ru":
        markers = DEFINITION_MARKERS_RU
    else:
        markers = DEFINITION_MARKERS_EN

    matches = 0
    for pattern in markers:
        if re.search(pattern, text, re.IGNORECASE):
            matches += 1

    # Score based on number of matches (capped at 1.0)
    # 1 match = 0.5, 2+ matches = 1.0
    if matches == 0:
        return 0.0
    elif matches == 1:
        return 0.5
    else:
        return min(1.0, 0.5 + (matches - 1) * 0.25)


def compute_rhetorical_score(text: str) -> float:
    """
    Compute rhetorical/engagement score.

    Detects questions, emphasis markers, and engaging language.

    Returns:
        Score 0-1 based on rhetorical devices
    """
    if not text.strip():
        return 0.0

    lang = detect_language(text)

    if lang == "ru":
        markers = RHETORICAL_MARKERS_RU
    else:
        markers = RHETORICAL_MARKERS_EN

    matches = 0
    for pattern in markers:
        if re.search(pattern, text, re.IGNORECASE):
            matches += 1

    # Questions are particularly engaging
    has_question = "?" in text

    # Score: base on matches, bonus for questions
    base_score = min(0.7, matches * 0.15)
    question_bonus = 0.3 if has_question else 0.0

    return min(1.0, base_score + question_bonus)


def compute_self_containment_score(text: str) -> Tuple[float, bool]:
    """
    Compute self-containment score.

    Segments with fewer pronouns and more complete thoughts are more self-contained.

    Returns:
        Tuple of (score 0-1, boolean self_contained flag)
    """
    if not text.strip():
        return 0.0, False

    lang = detect_language(text)

    if lang == "ru":
        pronoun_patterns = PRONOUNS_RU
    else:
        pronoun_patterns = PRONOUNS_EN

    # Count pronouns
    pronoun_count = 0
    for pattern in pronoun_patterns:
        pronoun_count += len(re.findall(pattern, text, re.IGNORECASE))

    word_count = len(text.split())
    if word_count == 0:
        return 0.0, False

    # Pronoun ratio
    pronoun_ratio = pronoun_count / word_count

    # Lower pronoun ratio = higher self-containment
    # < 5% pronouns = very self-contained
    # > 20% pronouns = not self-contained
    if pronoun_ratio < 0.05:
        score = 1.0
    elif pronoun_ratio > 0.20:
        score = 0.0
    else:
        # Linear interpolation between 0.05 and 0.20
        score = 1.0 - (pronoun_ratio - 0.05) / 0.15

    # Threshold for self_contained flag
    self_contained = pronoun_ratio < 0.15

    return score, self_contained


def merge_segments_for_clips(
    segments: List["ScoredSegment"],
    max_duration: float,
) -> List["ScoredSegment"]:
    """
    Merge consecutive short segments into clips up to max_duration.

    faster-whisper produces sentence-level segments (typically a few seconds
    each), but clips need to be ~45-90 seconds. Greedily coalesce consecutive
    segments (ordered by start time) into clips, never exceeding max_duration.
    Merged clips carry a union of tags.
    The clip score is the MAX of its member segments: a clip is as strong as
    its best moment (averaging a strong 5-second segment across a 60-second
    clip would dilute it below the selection threshold and yield no clips).

    Args:
        segments: Scored segments in any order.
        max_duration: Maximum clip length in seconds.

    Returns:
        Clips ordered by start time, ids renumbered clip_001, clip_002, ...
    """
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: s.start)
    merged: List["ScoredSegment"] = [ordered[0]]

    for seg in ordered[1:]:
        last = merged[-1]
        merged_end = max(last.end, seg.end)
        if merged_end - last.start <= max_duration:
            merged[-1] = _merge_scored_pair(last, seg)
        else:
            merged.append(seg)

    for i, clip in enumerate(merged, start=1):
        clip.id = f"clip_{i:03d}"

    return merged


def _merge_scored_pair(a: "ScoredSegment", b: "ScoredSegment") -> "ScoredSegment":
    """Merge two scored segments into one (score = max of the pair)."""
    start = min(a.start, b.start)
    end = max(a.end, b.end)

    score = round(max(a.score, b.score), 3)

    tags: List[str] = []
    for tag in a.tags + b.tags:
        if tag not in tags:
            tags.append(tag)

    return ScoredSegment(
        id=a.id,
        start=start,
        end=end,
        text=(a.text + " " + b.text).strip(),
        score=score,
        tags=tags,
        self_contained=a.self_contained and b.self_contained,
        term_density=round((a.term_density + b.term_density) / 2, 3),
        definition_score=round((a.definition_score + b.definition_score) / 2, 3),
        rhetorical_score=round((a.rhetorical_score + b.rhetorical_score) / 2, 3),
        self_containment_score=round((a.self_containment_score + b.self_containment_score) / 2, 3),
    )


def filter_by_duration(
    segments: List[ScoredSegment], min_duration: int, max_duration: int
) -> List[ScoredSegment]:
    """Filter segments by duration constraints."""
    filtered = []
    for seg in segments:
        duration = seg.end - seg.start
        if min_duration <= duration <= max_duration:
            filtered.append(seg)
        else:
            logger.debug(
                "Segment filtered by duration",
                segment_id=seg.id,
                duration=duration,
                min=min_duration,
                max=max_duration,
            )
    return filtered


def filter_by_score(segments: List[ScoredSegment], min_threshold: float) -> List[ScoredSegment]:
    """Filter segments by minimum score threshold."""
    filtered = [seg for seg in segments if seg.score >= min_threshold]
    logger.info(
        "Score filtering applied",
        before=len(segments),
        after=len(filtered),
        threshold=min_threshold,
    )
    return filtered


def select_top_segments(segments: List[ScoredSegment], max_count: int) -> List[ScoredSegment]:
    """Select top N segments by score."""
    sorted_segs = sorted(segments, key=lambda s: s.score, reverse=True)
    return sorted_segs[:max_count]


def score_segment(
    segment: Dict[str, Any], global_terms: List[str], scoring_config: ScoringConfig
) -> ScoredSegment:
    """
    Score a single segment using all heuristics.

    Weights:
    - Term density: 0.30
    - Definition markers: 0.35
    - Rhetorical devices: 0.15
    - Self-containment: 0.20
    """
    text = segment.get("text", "")
    start = segment.get("start", 0.0)
    end = segment.get("end", 0.0)
    seg_id = segment.get("id", f"seg_{start}_{end}")

    term_density_score, found_terms = compute_term_density_score(
        text, global_terms, scoring_config.term_extraction_method
    )
    definition_score = compute_definition_score(text)
    rhetorical_score = compute_rhetorical_score(text)
    self_containment_score, is_self_contained = compute_self_containment_score(text)

    weights = {
        "term_density": 0.30,
        "definition": 0.35,
        "rhetorical": 0.15,
        "self_containment": 0.20,
    }

    final_score = (
        term_density_score * weights["term_density"]
        + definition_score * weights["definition"]
        + rhetorical_score * weights["rhetorical"]
        + self_containment_score * weights["self_containment"]
    )

    tags = []
    if definition_score >= 0.5:
        tags.append("definition")
    if rhetorical_score >= 0.5:
        tags.append("engaging")
    if term_density_score >= 0.5:
        tags.append("high_term_density")
    if is_self_contained:
        tags.append("self_contained")
    if found_terms:
        tags.append(f"terms:{','.join(found_terms[:3])}")

    return ScoredSegment(
        id=seg_id,
        start=start,
        end=end,
        text=text,
        score=round(final_score, 3),
        tags=tags,
        self_contained=is_self_contained,
        term_density=round(term_density_score, 3),
        definition_score=round(definition_score, 3),
        rhetorical_score=round(rhetorical_score, 3),
        self_containment_score=round(self_containment_score, 3),
    )


def apply_llm_scores(
    scored_segments: List[ScoredSegment],
    llm_results: List[Dict[str, Any]],
    llm_weight: float,
) -> List[ScoredSegment]:
    """
    Combine heuristic scores with LLM scores using weighted average.

    Args:
        scored_segments: Segments with heuristic scores
        llm_results: LLM analysis results (list of dicts with 'id' and 'llm_score')
        llm_weight: Weight for LLM score (0-1), heuristic weight = 1 - llm_weight

    Returns:
        Updated scored segments with combined scores
    """
    llm_map = {r["id"]: r for r in llm_results if "id" in r}
    heuristic_weight = 1.0 - llm_weight

    for seg in scored_segments:
        llm_data = llm_map.get(seg.id)
        if llm_data is None:
            continue

        llm_score = llm_data.get("llm_score")
        if llm_score is None or not isinstance(llm_score, (int, float)):
            continue

        combined = seg.score * heuristic_weight + float(llm_score) * llm_weight
        seg.score = round(combined, 3)

        content_type = llm_data.get("content_type")
        if content_type and content_type not in seg.tags:
            seg.tags.append(f"llm:{content_type}")

        virality = llm_data.get("virality_score")
        if virality is not None and isinstance(virality, (int, float)) and virality >= 0.7:
            if "viral_potential" not in seg.tags:
                seg.tags.append("viral_potential")

    logger.info(
        "llm_scores_applied",
        segments_updated=len(llm_map),
        llm_weight=llm_weight,
    )
    return scored_segments


def save_llm_analysis(
    path: Path,
    llm_cfg: LLMConfig,
    llm_results: List[Dict[str, Any]],
    segments: List[ScoredSegment],
    batches_total: int,
    batches_failed: int,
) -> None:
    """
    Persist raw LLM results next to the heuristic scores they are mixed with.

    Debug artifact for comparing scoring iterations; nothing downstream
    reads it.

    Args:
        path: Output JSON path (artifacts/llm_analysis.json).
        llm_cfg: LLM configuration used for the run.
        llm_results: Raw per-segment results from the adapter.
        segments: Scored segments, still holding heuristic scores.
        batches_total: Number of LLM batches.
        batches_failed: Number of failed (skipped) batches.
    """
    by_id = {s.id: s for s in segments}
    records = []
    for result in llm_results:
        seg = by_id.get(result.get("id"))
        records.append(
            {
                **result,
                "start": seg.start if seg else None,
                "end": seg.end if seg else None,
                "heuristic_score": seg.score if seg else None,
            }
        )
    payload = {
        "provider": llm_cfg.provider,
        "model": llm_cfg.model,
        "prompt_file": llm_cfg.prompt_file,
        "llm_weight": llm_cfg.llm_weight,
        "batches": {"total": batches_total, "failed": batches_failed},
        "results": records,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    logger.info("llm_analysis_saved", path=str(path), results=len(records))


def extract_global_terms(
    transcript: List[Dict[str, Any]], method: str = "keybert", top_n: int = 50
) -> List[str]:
    """
    Extract global terms from entire transcript.

    These terms are used to identify term-dense segments.
    """
    # Concatenate all text
    full_text = " ".join(seg.get("text", "") for seg in transcript)

    logger.info("Extracting global terms", method=method, top_n=top_n)
    terms = extract_terms(full_text, method, top_n)

    logger.info("Global terms extracted", count=len(terms))
    return terms


def score_transcript(config: Config, dry_run: bool = False) -> Optional[List[Dict[str, Any]]]:
    """
    Main scoring function.

    Args:
        config: Pipeline configuration
        dry_run: If True, validate without executing

    Returns:
        List of scored segments on success, None on failure

    Raises:
        ScoringError: On scoring failure
    """
    logger.info("Starting semantic scoring", dry_run=dry_run)

    artifacts_dir = config.work_dir / "artifacts"
    transcript_path = artifacts_dir / "transcript.json"
    scored_path = artifacts_dir / "scored_segments.json"

    if not transcript_path.exists():
        raise ScoringError(f"Transcript file not found: {transcript_path}")

    transcript = load_transcript(transcript_path)

    if dry_run:
        logger.info("Dry run - skipping actual scoring")
        return [{"status": "dry_run_passed"}]

    clear_notices(config.work_dir, "scoring")
    llm_cfg = config.scoring.llm
    use_llm = llm_cfg.enabled
    if config.scoring.strategy == "llm_windows":
        if llm_cfg.enabled:
            output = _score_with_windows(config, transcript, artifacts_dir)
            if output is not None:
                _save_scored_segments(output, scored_path)
                return output
        else:
            logger.warning("llm_windows_requires_llm", fallback="segment_merge")
        # The LLM already failed (or is off): heuristics only.
        use_llm = False

    global_terms = extract_global_terms(
        transcript, method=config.scoring.term_extraction_method, top_n=50
    )

    scored_segments = []
    for i, segment in enumerate(transcript):
        try:
            scored = score_segment(segment, global_terms, config.scoring)
            scored_segments.append(scored)
        except Exception as e:
            logger.warning("Failed to score segment", segment_index=i, error=str(e))
            continue

    logger.info("All segments scored", count=len(scored_segments))

    if use_llm:
        logger.info("llm_analysis_enabled", provider=llm_cfg.provider, model=llm_cfg.model)
        try:
            from src.llm_adapter import create_llm_adapter

            adapter = create_llm_adapter(llm_cfg, config.work_dir, "scoring")
            try:
                # Select segments for LLM analysis. Long videos produce
                # thousands of short whisper segments; analyzing all of them
                # is impractical, so cap to the top heuristic candidates.
                llm_candidates: List[Dict[str, Any]] = [
                    {
                        "id": s.id,
                        "start": s.start,
                        "end": s.end,
                        "text": s.text,
                    }
                    for s in scored_segments
                ]
                max_segments = llm_cfg.max_segments
                if max_segments and len(llm_candidates) > max_segments:
                    ranked = sorted(scored_segments, key=lambda s: s.score, reverse=True)[
                        :max_segments
                    ]
                    llm_candidates = [
                        {"id": s.id, "start": s.start, "end": s.end, "text": s.text} for s in ranked
                    ]
                    logger.info(
                        "llm_segment_cap_applied",
                        analyzed=len(llm_candidates),
                        total=len(scored_segments),
                    )

                # Process in batches so the LLM response fits into max_tokens
                # (single-shot analysis of a full transcript gets truncated
                # mid-JSON and fails to parse). A failed batch is skipped;
                # heuristics take over only if most batches fail.
                batches = [
                    llm_candidates[i : i + LLM_SEGMENT_BATCH_SIZE]
                    for i in range(0, len(llm_candidates), LLM_SEGMENT_BATCH_SIZE)
                ]
                llm_results, _, failed = run_llm_batches(batches, adapter.analyze_segments)
                # Before apply_llm_scores overwrites the heuristic scores.
                save_llm_analysis(
                    artifacts_dir / "llm_analysis.json",
                    llm_cfg,
                    llm_results,
                    scored_segments,
                    batches_total=len(batches),
                    batches_failed=failed,
                )
                scored_segments = apply_llm_scores(scored_segments, llm_results, llm_cfg.llm_weight)
                logger.info("llm_analysis_complete", results_count=len(llm_results))
            finally:
                adapter.unload()
        except Exception as e:
            add_notice(
                config.work_dir,
                "scoring",
                "llm_failed_heuristics_used",
                LLM_FALLBACK_MESSAGE,
                error=str(e),
                strategy="segment_merge",
            )
    else:
        logger.info("llm_analysis_disabled")

    # Whisper produces short sentence-level segments; merge them into clips of
    # target length so the duration filter does not drop everything.
    merged = merge_segments_for_clips(scored_segments, config.scoring.max_duration)

    filtered = filter_by_duration(merged, config.scoring.min_duration, config.scoring.max_duration)

    if not filtered and merged:
        # Graceful degradation: a short source video (< min_duration) would
        # otherwise yield zero clips — keep the merged clips anyway.
        logger.warning(
            "duration_filter_removed_all_clips",
            reason="source shorter than min_duration",
            merged_count=len(merged),
        )
        filtered = merged

    filtered = filter_by_score(filtered, config.scoring.min_score_threshold)

    top_segments = select_top_segments(filtered, config.scoring.max_clips_per_video)

    logger.info(
        "Scoring complete",
        total_scored=len(scored_segments),
        after_merge=len(merged),
        after_duration_filter=len(filtered),
        final_selected=len(top_segments),
    )

    output = []
    for seg in top_segments:
        output.append(
            {
                "id": seg.id,
                "start": seg.start,
                "end": seg.end,
                "score": seg.score,
                "tags": seg.tags,
                "self_contained": seg.self_contained,
                "_debug": {
                    "term_density": seg.term_density,
                    "definition_score": seg.definition_score,
                    "rhetorical_score": seg.rhetorical_score,
                    "self_containment_score": seg.self_containment_score,
                },
            }
        )

    _save_scored_segments(output, scored_path)
    return output


def _save_scored_segments(output: List[Dict[str, Any]], scored_path: Path) -> None:
    with open(scored_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info("Scored segments saved", path=str(scored_path), count=len(output))


def run_llm_batches(
    batches: List[Any],
    call: Callable[[Any], List[Dict[str, Any]]],
) -> Tuple[List[Dict[str, Any]], int, int]:
    """
    Run LLM calls batch by batch; a failed batch is skipped.

    Args:
        batches: Batch payloads.
        call: Performs one LLM call for a batch and returns parsed results.

    Returns:
        (results of all successful batches, ok count, failed count)

    Raises:
        LLMUnavailableError: The backend cannot serve the run (propagated).
        LLMAdapterError: More than half of the batches failed (checked early,
            so the remaining batches are not attempted).
    """
    from src.llm_adapter import LLMAdapterError, LLMUnavailableError

    results: List[Dict[str, Any]] = []
    failed = 0
    for batch_index, batch in enumerate(batches, start=1):
        logger.info(
            "llm_batch_analysis_start",
            batch_index=batch_index,
            total_batches=len(batches),
            items_in_batch=len(batch),
        )
        try:
            batch_results = call(batch)
        except LLMUnavailableError:
            raise
        except Exception as e:
            failed += 1
            logger.warning(
                "llm_batch_failed",
                batch_index=batch_index,
                total_batches=len(batches),
                error=str(e),
            )
            if failed * 2 > len(batches):
                raise LLMAdapterError(f"{failed} of {len(batches)} LLM batches failed") from e
            continue
        if batch_results:
            results.extend(batch_results)
    ok = len(batches) - failed
    logger.info("llm_batches_summary", ok=ok, failed=failed, total=len(batches))
    return results, ok, failed


def _score_with_windows(
    config: Config,
    transcript: List[Dict[str, Any]],
    artifacts_dir: Path,
) -> Optional[List[Dict[str, Any]]]:
    """
    Scoring strategy "llm_windows" (see src/window_scoring.py).

    Returns:
        scored_segments.json payload, or None if the LLM path failed and the
        caller should fall back to segment_merge.
    """
    from src.llm_adapter import create_llm_adapter
    from src.window_scoring import (
        load_prompt_template,
        score_windows,
        to_scored_segments,
    )

    llm_cfg = config.scoring.llm
    logger.info("window_scoring_enabled", provider=llm_cfg.provider, model=llm_cfg.model)
    try:
        template = load_prompt_template(llm_cfg.window_prompt_file)
        adapter = create_llm_adapter(llm_cfg, config.work_dir, "scoring")
        try:
            selected, candidates, debug = score_windows(
                transcript,
                config.scoring,
                lambda payload, prompt: adapter.analyze_segments(payload, prompt=prompt),
                run_llm_batches,
                template,
            )
        finally:
            adapter.unload()
    except Exception as e:
        add_notice(
            config.work_dir,
            "scoring",
            "llm_failed_heuristics_used",
            LLM_FALLBACK_MESSAGE,
            error=str(e),
            strategy="llm_windows",
        )
        return None

    debug.update(
        provider=llm_cfg.provider,
        model=llm_cfg.model,
        prompt_file=llm_cfg.window_prompt_file,
    )
    with open(artifacts_dir / "llm_analysis.json", "w", encoding="utf-8") as f:
        json.dump(debug, f, indent=2, ensure_ascii=False)

    output = to_scored_segments(selected)
    logger.info(
        "Scoring complete",
        strategy="llm_windows",
        candidates=len(candidates),
        final_selected=len(output),
    )
    return output


def main():
    """CLI entry point for standalone scoring."""
    import sys

    try:
        config = Config.load("config/config.yaml")
    except FileNotFoundError:
        print("Error: config/config.yaml not found")
        sys.exit(1)

    try:
        result = score_transcript(config, dry_run=False)
        if result:
            print(f"Successfully scored {len(result)} segments")
            for seg in result:
                print(f"  {seg['id']}: score={seg['score']}, tags={seg['tags']}")
    except ScoringError as e:
        print(f"Scoring error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
