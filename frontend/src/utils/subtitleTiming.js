// Shared subtitle / transcript timing helpers.
//
// Both ``SubtitleOverlay`` (burned-in preview subtitles) and
// ``TranscriptViewer`` (Analysis tab active-line highlight) need to answer
// the same question: "at time ``t``, which transcript segment is actually
// being spoken?" The honest answer isn't ``seg.start <= t < seg.end`` —
// Whisper segment-level timestamps are boundary estimates that routinely
// drift ±100–300 ms from real speech onset/offset. Whisper's *per-word*
// timestamps (captured upstream in ``backend/services/transcription.py``
// and already round-tripped through the frontend as ``seg.words``) are
// much tighter.
//
// ``spokenWindow(seg)`` returns a ``{ start, end }`` tuple that prefers
// ``words[0].start`` / ``words[-1].end`` when a segment has per-word
// timestamps, and falls back to the segment-level values when it doesn't
// (or when the word list is broken/inverted). The small head/tail paddings
// pull the window slightly earlier than the first word's phoneme onset
// and slightly past the last word's release, so the visible highlight
// lands *with* the first syllable and doesn't cut off the final phoneme —
// tuned to fall inside one 30 fps video frame of the real audio.

// Tail padding (s) after the last word's Whisper end — covers the decay /
// release of the final phoneme without lingering. Tuned empirically to
// land inside a single 30 fps video frame of the real audio.
export const WORD_TAIL_S = 0.06;

// Head padding (s) before the first word's Whisper start — Whisper word
// ``start`` is usually placed at the phoneme onset, which is ~30–50 ms
// late relative to the perceptual attack. Pulling the visible window
// slightly earlier makes the subtitle appear *with* the first syllable,
// not after it.
export const WORD_HEAD_S = 0.04;

/**
 * Compute the ``[start, end]`` window during which a transcript segment
 * is actually being spoken. Prefers per-word timestamps when available
 * and plausible; falls back to segment-level timestamps otherwise.
 *
 * The returned shape matches ``{ start, end }`` so callers can treat the
 * output as a drop-in replacement for ``seg.start`` / ``seg.end``.
 */
export function spokenWindow(seg) {
  if (!seg) return { start: 0, end: 0 };
  const segStart = seg.start ?? 0;
  const segEnd = seg.end ?? segStart;
  const words = Array.isArray(seg.words) ? seg.words : null;
  if (!words || words.length === 0) {
    return { start: segStart, end: segEnd };
  }
  // Guard: the word list must bracket the segment plausibly. If word
  // timestamps are broken (all zero, inverted, or missing numeric fields)
  // fall back to segment-level.
  const first = words[0];
  const last = words[words.length - 1];
  const ws = first?.start ?? first?.startTime;
  const we = last?.end ?? last?.endTime;
  if (typeof ws !== 'number' || typeof we !== 'number' || we <= ws) {
    return { start: segStart, end: segEnd };
  }
  // Clamp to segment bounds with a small bleed so we can't overlap
  // neighbors. ``segStart - 0.02`` / ``segEnd + 0.15`` matches the
  // tolerances the export pipeline uses for boundary nudging.
  const start = Math.max(segStart - 0.02, ws - WORD_HEAD_S);
  const end = Math.min(segEnd + 0.15, we + WORD_TAIL_S);
  if (end <= start) return { start: segStart, end: segEnd };
  return { start, end };
}

/**
 * True iff ``t`` is inside the spoken window of ``seg``. Shared predicate
 * used by ``SubtitleOverlay`` and ``TranscriptViewer`` so their notion of
 * "active" stays consistent.
 */
export function isSpokenAt(seg, t) {
  const { start, end } = spokenWindow(seg);
  return start <= t && t < end;
}
