import React, { useEffect, useMemo, useState } from 'react';

/**
 * TranscriptionCoverageBadge — Phase 1 of the "Close the Transcription
 * Coverage Gap" project.
 *
 * Fetches GET /api/jobs/{jobId}/coverage and renders:
 *   1. A coloured badge ("Coverage: 97%")
 *      - green   when coverage >= 95
 *      - amber   when coverage 85–94
 *      - red     when coverage < 85
 *   2. An expandable list of uncovered-speech "gaps" — clicking a gap
 *      fires onSeek(startSec) so the preview jumps to that timestamp
 *      and the user can verify by ear.
 *
 * The backend audit is non-destructive: if silero-vad is not installed or
 * the audit failed, the report arrives with status "skipped" or "error".
 * In that case we render a neutral, dismissable info strip instead of
 * surfacing a scary red badge.
 */
export default function TranscriptionCoverageBadge({ jobId, onSeek }) {
  const [report, setReport] = useState(null);
  const [loading, setLoading] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!jobId) return;
    let cancelled = false;
    setLoading(true);
    setError(null);
    fetch(`/api/jobs/${jobId}/coverage`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (cancelled) return;
        setReport(data?.coverage_report || null);
      })
      .catch((e) => {
        if (cancelled) return;
        setError(String(e?.message || e));
      })
      .finally(() => {
        if (cancelled) return;
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [jobId]);

  const palette = useMemo(() => {
    const pct = report?.coverage_pct;
    if (typeof pct !== 'number') return { bg: 'var(--bg-elevated)', fg: 'var(--text-secondary)', border: 'var(--border)' };
    if (pct >= 95) return { bg: 'rgba(16,185,129,0.12)', fg: '#10B981', border: 'rgba(16,185,129,0.4)' };
    if (pct >= 85) return { bg: 'rgba(245,158,11,0.14)', fg: '#F59E0B', border: 'rgba(245,158,11,0.4)' };
    return { bg: 'rgba(239,68,68,0.14)', fg: '#EF4444', border: 'rgba(239,68,68,0.45)' };
  }, [report]);

  if (!jobId) return null;
  if (loading) return null; // Stay quiet until we know the result
  if (error) return null;   // Silent on error — this is a nice-to-have

  // Audit not available or not run (job too old / silero-vad not installed)
  if (!report || report.status !== 'ok') {
    if (!report) return null;
    return (
      <div
        data-testid="coverage-badge-unavailable"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 8,
          padding: '6px 10px',
          marginBottom: 10,
          background: 'var(--bg-elevated)',
          border: '1px solid var(--border)',
          borderRadius: 'var(--radius-sm)',
          fontSize: 11,
          color: 'var(--text-secondary)',
        }}
        title={report.error || `Audit status: ${report.status}`}
      >
        <span>Coverage audit unavailable</span>
        <span style={{ opacity: 0.6 }}>({report.status})</span>
      </div>
    );
  }

  const pct = report.coverage_pct;
  const gaps = report.gaps || [];
  // Gaps with real energy are far more interesting than low-RMS VAD misfires
  const ENERGY_THRESHOLD = 0.005;
  const highEnergyCount = gaps.filter((g) => (g.energy_rms || 0) >= ENERGY_THRESHOLD).length;

  return (
    <div
      data-testid="coverage-badge"
      style={{
        marginBottom: 12,
      }}
    >
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '8px 12px',
          background: palette.bg,
          border: `1px solid ${palette.border}`,
          borderRadius: 'var(--radius-sm)',
          fontSize: 12,
          cursor: gaps.length ? 'pointer' : 'default',
        }}
        onClick={() => gaps.length && setExpanded((v) => !v)}
        title={
          gaps.length
            ? `Click to ${expanded ? 'hide' : 'show'} ${gaps.length} uncovered speech region${gaps.length === 1 ? '' : 's'}`
            : 'No missed speech detected'
        }
      >
        <strong style={{ color: palette.fg, fontSize: 13 }}>
          Coverage: {Number.isFinite(pct) ? `${pct.toFixed(1)}%` : '—'}
        </strong>
        <span style={{ color: 'var(--text-secondary)', fontSize: 11 }}>
          {report.covered_seconds?.toFixed?.(1) ?? '0.0'}s /{' '}
          {report.total_speech_seconds?.toFixed?.(1) ?? '0.0'}s speech
        </span>
        {gaps.length > 0 && (
          <span
            style={{
              marginLeft: 'auto',
              padding: '2px 8px',
              fontSize: 11,
              background: 'rgba(0,0,0,0.25)',
              border: `1px solid ${palette.border}`,
              borderRadius: 999,
              color: palette.fg,
              fontWeight: 600,
            }}
          >
            {gaps.length} gap{gaps.length === 1 ? '' : 's'}
            {highEnergyCount > 0 && highEnergyCount !== gaps.length && (
              <span style={{ opacity: 0.75, marginLeft: 4 }}>
                ({highEnergyCount} likely speech)
              </span>
            )}
          </span>
        )}
      </div>

      {expanded && gaps.length > 0 && (
        <div
          data-testid="coverage-gap-list"
          style={{
            marginTop: 6,
            padding: '6px 10px',
            background: 'var(--bg-elevated)',
            border: '1px solid var(--border)',
            borderRadius: 'var(--radius-sm)',
            maxHeight: 180,
            overflowY: 'auto',
          }}
        >
          <div style={{ fontSize: 10, color: 'var(--text-secondary)', marginBottom: 6, textTransform: 'uppercase', letterSpacing: 0.5 }}>
            Possible missed dialogue — click to jump
          </div>
          {gaps.map((g, i) => {
            const isLowEnergy = (g.energy_rms || 0) < ENERGY_THRESHOLD;
            return (
              <div
                key={`${g.start}-${g.end}-${i}`}
                role="button"
                tabIndex={0}
                onClick={(e) => {
                  e.stopPropagation();
                  onSeek?.(g.start);
                }}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    e.preventDefault();
                    onSeek?.(g.start);
                  }
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '4px 6px',
                  fontSize: 11,
                  borderRadius: 'var(--radius-sm)',
                  cursor: 'pointer',
                  opacity: isLowEnergy ? 0.6 : 1,
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.background = 'rgba(255,255,255,0.05)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.background = 'transparent';
                }}
                title={
                  isLowEnergy
                    ? `Low-energy gap (${g.energy_rms?.toFixed?.(4) ?? '0'}) — likely VAD false positive`
                    : `Click to play from ${formatTime(g.start)}`
                }
              >
                <span style={{ fontFamily: 'var(--font-mono, monospace)', color: 'var(--text-primary)' }}>
                  {formatTime(g.start)}
                  {' → '}
                  {formatTime(g.end)}
                </span>
                <span style={{ color: 'var(--text-secondary)' }}>
                  {g.duration?.toFixed?.(1) ?? '0.0'}s
                </span>
                <span
                  style={{
                    fontSize: 10,
                    padding: '1px 6px',
                    borderRadius: 999,
                    background: isLowEnergy ? 'rgba(255,255,255,0.05)' : 'rgba(239,68,68,0.2)',
                    color: isLowEnergy ? 'var(--text-secondary)' : '#EF4444',
                  }}
                >
                  {isLowEnergy ? 'quiet' : 'speech?'}
                </span>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function formatTime(sec) {
  if (!Number.isFinite(sec)) return '0:00';
  const h = Math.floor(sec / 3600);
  const m = Math.floor((sec % 3600) / 60);
  const s = Math.floor(sec % 60);
  const pad = (n) => String(n).padStart(2, '0');
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}
