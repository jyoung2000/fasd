// Ad-hoc smoke test for frontend/src/utils/subtitleTiming.js.
//
// No frontend test harness exists, so this file is intentionally a
// framework-free Node script. Run it with:
//
//     node scripts/test_subtitleTiming.mjs
//
// It exits non-zero on any failed assertion so it's CI-friendly if you
// decide to wire it up later.

import {
  spokenWindow,
  isSpokenAt,
  WORD_HEAD_S,
  WORD_TAIL_S,
} from '../frontend/src/utils/subtitleTiming.js';

let failures = 0;
function check(name, cond, details) {
  if (cond) {
    console.log(`  ok  ${name}`);
  } else {
    failures += 1;
    console.log(`  FAIL ${name}`);
    if (details !== undefined) console.log('       ', details);
  }
}
function approx(a, b, eps = 1e-9) {
  return Math.abs(a - b) < eps;
}

console.log('spokenWindow — segment with no words');
{
  const seg = { start: 1.0, end: 2.5 };
  const w = spokenWindow(seg);
  check('returns segment-level start', approx(w.start, 1.0), w);
  check('returns segment-level end', approx(w.end, 2.5), w);
}

console.log('spokenWindow — segment with valid words');
{
  const seg = {
    start: 1.0,
    end: 2.5,
    words: [
      { start: 1.10, end: 1.40 },
      { start: 1.45, end: 1.80 },
      { start: 1.85, end: 2.30 },
    ],
  };
  const w = spokenWindow(seg);
  // Expected: clamp(max(segStart - 0.02, ws - WORD_HEAD_S), ws - WORD_HEAD_S)
  //   ws = 1.10, ws - WORD_HEAD_S = 1.06, segStart - 0.02 = 0.98
  //   -> max(0.98, 1.06) = 1.06
  check('start uses words[0].start - WORD_HEAD_S', approx(w.start, 1.10 - WORD_HEAD_S), w);
  // we = 2.30, we + WORD_TAIL_S = 2.36, segEnd + 0.15 = 2.65
  //   -> min(2.65, 2.36) = 2.36
  check('end uses words[-1].end + WORD_TAIL_S', approx(w.end, 2.30 + WORD_TAIL_S), w);
}

console.log('spokenWindow — segment with inverted/broken words');
{
  const seg1 = { start: 1.0, end: 2.0, words: [{ start: 0, end: 0 }] };
  const w1 = spokenWindow(seg1);
  check('zero-duration word falls back to seg', approx(w1.start, 1.0) && approx(w1.end, 2.0), w1);

  const seg2 = { start: 1.0, end: 2.0, words: [{ start: 1.5, end: 1.2 }] };
  const w2 = spokenWindow(seg2);
  check('inverted word (end <= start) falls back to seg', approx(w2.start, 1.0) && approx(w2.end, 2.0), w2);

  const seg3 = { start: 1.0, end: 2.0, words: [{ start: 'bad', end: 1.5 }] };
  const w3 = spokenWindow(seg3);
  check('non-numeric word start falls back to seg', approx(w3.start, 1.0) && approx(w3.end, 2.0), w3);

  const seg4 = { start: 1.0, end: 2.0, words: [] };
  const w4 = spokenWindow(seg4);
  check('empty word array falls back to seg', approx(w4.start, 1.0) && approx(w4.end, 2.0), w4);
}

console.log('spokenWindow — clamp to segment bounds');
{
  // Words that start way before or end way after segment bounds get
  // clamped to segStart - 0.02 and segEnd + 0.15.
  const seg = {
    start: 5.0,
    end: 6.0,
    words: [
      { start: 1.00, end: 1.20 }, // pathological early
      { start: 9.00, end: 9.30 }, // pathological late
    ],
  };
  const w = spokenWindow(seg);
  // ws - WORD_HEAD_S = 0.96; segStart - 0.02 = 4.98 → max = 4.98
  check('start clamped to segStart - 0.02', approx(w.start, 4.98), w);
  // we + WORD_TAIL_S = 9.36; segEnd + 0.15 = 6.15 → min = 6.15
  check('end clamped to segEnd + 0.15', approx(w.end, 6.15), w);
}

console.log('isSpokenAt — boundary conditions');
{
  const seg = {
    start: 1.0,
    end: 2.0,
    words: [
      { start: 1.02, end: 1.50 },
      { start: 1.55, end: 1.95 },
    ],
  };
  const w = spokenWindow(seg);
  // w.start = max(0.98, 0.98) = 0.98; w.end = min(2.15, 2.01) = 2.01
  check('inclusive at start', isSpokenAt(seg, w.start) === true, w);
  check('exclusive at end', isSpokenAt(seg, w.end) === false, w);
  check('1 ms inside is active', isSpokenAt(seg, w.start + 0.001) === true, w);
  check('1 ms after end is not active', isSpokenAt(seg, w.end + 0.001) === false, w);
  check('1 ms before start is not active', isSpokenAt(seg, w.start - 0.001) === false, w);
}

console.log('Integration — TranscriptViewer gap-hold math');
{
  // Minimal re-implementation of the TranscriptViewer memo body so we
  // can unit-test the gap-hold behavior without a React harness.
  const activeIdx = (transcript, currentTime, GAP_HOLD_SEC = 0.75) => {
    if (currentTime == null || !transcript.length) return -1;
    let lastEndedIdx = -1;
    let lastEndedTime = -Infinity;
    for (let i = 0; i < transcript.length; i++) {
      const w = spokenWindow(transcript[i]);
      if (w.start <= currentTime && currentTime < w.end) return i;
      if (w.end <= currentTime && w.end > lastEndedTime) {
        lastEndedTime = w.end;
        lastEndedIdx = i;
      }
    }
    if (lastEndedIdx >= 0 && currentTime - lastEndedTime <= GAP_HOLD_SEC) {
      return lastEndedIdx;
    }
    return -1;
  };

  // Segments with no word timestamps, so spokenWindow returns seg-level.
  //   seg0: 0.0–1.0
  //   (short gap: 0.3 s)
  //   seg1: 1.3–2.0
  //   (long gap: 2.0 s)
  //   seg2: 4.0–5.0
  const tr = [
    { start: 0.0, end: 1.0 },
    { start: 1.3, end: 2.0 },
    { start: 4.0, end: 5.0 },
  ];
  check('t=0.5 → seg0', activeIdx(tr, 0.5) === 0);
  check('t=1.1 (in 0.3s gap, hold seg0)', activeIdx(tr, 1.1) === 0);
  check('t=1.5 → seg1', activeIdx(tr, 1.5) === 1);
  // Right after seg1 ends, inside 0.75 s hold → still seg1
  check('t=2.3 (in long gap but <=0.75s hold) → seg1', activeIdx(tr, 2.3) === 1);
  // Past hold window → clear
  check('t=2.9 (past 0.75s hold) → -1', activeIdx(tr, 2.9) === -1);
  check('t=3.5 (far past hold) → -1', activeIdx(tr, 3.5) === -1);
  check('t=4.1 → seg2', activeIdx(tr, 4.1) === 2);
  check('t=99 (past everything, past hold) → -1', activeIdx(tr, 99) === -1);
}

console.log('');
if (failures > 0) {
  console.log(`FAILED: ${failures} assertion(s)`);
  process.exit(1);
} else {
  console.log('All subtitleTiming smoke tests passed.');
}
