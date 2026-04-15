/**
 * Shared default clip settings — SINGLE SOURCE OF TRUTH.
 * Used by ClipSettingsPanel, Analysis, ViralClips, and anywhere
 * that needs default subtitle/clip settings.
 */
export const DEFAULT_CLIP_SETTINGS = {
  clipCount: 12,
  minDuration: 15,
  maxDuration: 600,
  aspectRatio: null,
  subtitlesEnabled: true,  // Subtitles ON by default when transcript exists
  subtitleFont: 'DM Sans',
  subtitleSize: 30,
  subtitleFontWeight: 700,
  subtitleFontColor: '#FFFFFF',
  subtitlePosition: 'bottom',
  speakerColors: {},
  subtitleBgEnabled: false,
  subtitleBgColor: '#000000',
  subtitleBgOpacity: 75,
  subtitleBgRadius: 0,
  subtitleOutlineColor: '#000000',
  subtitleOutlineOpacity: 100,
  subtitleOutlineWidth: 2,
  showSpeakerLabels: false,
  subtitleMaxWidth: 90,
  subtitleOffsetV: 4,
  subtitleMaxWords: 0,
  activeWordEnabled: false,
  activeWordColor: '#FFD700',
  activeWordOutlineColor: '#000000',
  activeWordBgColor: '#000000',
  activeWordBgOpacity: 0,
  activeWordBgRadius: 4,
  useSpeakerColors: true,
  exportQuality: '1080p',
  playbackVolume: 100,
  playbackSpeed: 1.0,
};

/**
 * Clip-generation panel defaults (Analysis page "Generate Clips" controls).
 * Persisted in localStorage under GEN_STORAGE_KEY by Analysis.jsx. Bug 6
 * in the clip-focus audit: keep minDuration/maxDuration matching
 * DEFAULT_CLIP_SETTINGS above so first-run users see one canonical value.
 */
export const DEFAULT_GEN_SETTINGS = {
  clipCount: 12,
  minDuration: 15,
  maxDuration: 600,
  viralScoreMin: 0,
  viralScoreMax: 100,
  // Focus mode — persisted so a refresh doesn't wipe the user's query
  clipFocusEnabled: false,
  clipFocusText: '',
  minRelevance: 50,
  // Focus query history — last 10 unique queries, MRU order
  focusHistory: [],
};
