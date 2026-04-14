/**
 * Settings page section: Cloud Storage.
 *
 * Shows one card per provider (Google Drive + Box). For each provider:
 *
 *   - If the server has no client id/secret configured, we show a
 *     "Not configured" state with a link to the setup doc.
 *   - If configured but not connected, we show a Connect button that
 *     pops open the OAuth URL in a new window. When that window
 *     redirects back through /api/cloud/{provider}/callback it lands
 *     on /settings?connected=... — we detect that query string on mount
 *     and refresh the provider list.
 *   - If connected, we show the display name (email), scopes, and a
 *     Disconnect button.
 *
 * This component is intentionally self-contained so Settings.jsx only
 * needs a single `<CloudStorageSection />` import.
 */
import React, { useCallback, useEffect, useState } from 'react';
import useCloudProviders from '../../hooks/useCloudProviders';

const PROVIDER_LABELS = {
  google_drive: {
    title: 'Google Drive',
    description:
      'Browse videos from your Google Drive and analyze them without a local download.',
    icon: 'GD',
  },
  box: {
    title: 'Box',
    description:
      'Browse videos from your Box account and analyze them without a local download.',
    icon: 'BX',
  },
};

function cardStyle() {
  return {
    background: 'var(--bg-panel)',
    border: '1px solid var(--border)',
    borderRadius: 'var(--radius-md)',
    padding: 16,
    marginBottom: 12,
  };
}

function buttonStyle(variant = 'primary') {
  if (variant === 'danger') {
    return {
      padding: '8px 16px',
      fontSize: 13,
      background: 'transparent',
      color: 'var(--text-secondary)',
      border: '1px solid var(--border)',
      borderRadius: 'var(--radius-sm)',
      cursor: 'pointer',
    };
  }
  return {
    padding: '8px 16px',
    fontSize: 13,
    background: 'var(--accent-cyan)',
    color: '#000',
    border: 'none',
    borderRadius: 'var(--radius-sm)',
    cursor: 'pointer',
    fontWeight: 600,
  };
}

export default function CloudStorageSection() {
  const { loading, enabled, providers, error, refresh } = useCloudProviders();
  const [busy, setBusy] = useState('');

  // If a callback-driven redirect lands us here with ?connected=<provider>,
  // refresh the provider list and clean up the URL.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const connected = params.get('connected');
    if (connected) {
      refresh();
      params.delete('connected');
      const newQs = params.toString();
      const newPath =
        window.location.pathname + (newQs ? `?${newQs}` : '') + window.location.hash;
      window.history.replaceState({}, '', newPath);
    }
  }, [refresh]);

  // Listen for postMessage from the OAuth popup (if a child window flow
  // is used instead of a full top-level redirect).
  useEffect(() => {
    const onMessage = (event) => {
      if (!event?.data || typeof event.data !== 'object') return;
      if (event.data.type === 'clipai:cloud-connected') {
        refresh();
      }
    };
    window.addEventListener('message', onMessage);
    return () => window.removeEventListener('message', onMessage);
  }, [refresh]);

  const handleConnect = useCallback(
    async (providerName) => {
      setBusy(providerName);
      try {
        const resp = await fetch(`/api/cloud/${providerName}/authorize`);
        if (!resp.ok) {
          const body = await resp.json().catch(() => ({}));
          throw new Error(body.detail || `HTTP ${resp.status}`);
        }
        const { authorize_url } = await resp.json();
        // Open in a new tab — the callback redirect handler will bounce
        // users back to /settings?connected=<provider>. This is the most
        // compatible approach across browsers and iframe sandboxes.
        window.open(authorize_url, '_blank', 'noopener');
      } catch (err) {
        alert(`Failed to start OAuth for ${providerName}: ${err.message}`);
      } finally {
        setBusy('');
      }
    },
    []
  );

  const handleDisconnect = useCallback(
    async (accountId, providerName) => {
      if (!window.confirm(`Disconnect ${PROVIDER_LABELS[providerName]?.title || providerName}?`)) {
        return;
      }
      setBusy(providerName);
      try {
        const resp = await fetch(`/api/cloud/accounts/${accountId}`, { method: 'DELETE' });
        if (!resp.ok) {
          const body = await resp.json().catch(() => ({}));
          throw new Error(body.detail || `HTTP ${resp.status}`);
        }
        await refresh();
      } catch (err) {
        alert(`Failed to disconnect: ${err.message}`);
      } finally {
        setBusy('');
      }
    },
    [refresh]
  );

  if (loading) {
    return (
      <div style={{ padding: 16, color: 'var(--text-secondary)' }}>
        Loading cloud storage providers...
      </div>
    );
  }

  if (!enabled) {
    return (
      <div style={{ padding: 16, color: 'var(--text-secondary)' }}>
        Cloud storage integration is disabled. Set
        <code style={{ margin: '0 6px' }}>CLIPAI_CLOUD_STORAGE_ENABLED=true</code>
        to enable it.
      </div>
    );
  }

  if (error) {
    return (
      <div style={{ padding: 16, color: 'var(--accent-red, #f66)' }}>
        Failed to load cloud providers: {error}
      </div>
    );
  }

  return (
    <div>
      <p
        style={{
          fontSize: 13,
          color: 'var(--text-secondary)',
          marginTop: 0,
          marginBottom: 12,
        }}
      >
        Connect a cloud account to browse and import videos directly from Google Drive or
        Box. Files are streamed server-side &mdash; no tokens touch your browser and nothing
        is copied to your local machine.
      </p>

      {providers.map((p) => {
        const label = PROVIDER_LABELS[p.name] || { title: p.name, description: '', icon: '?' };
        return (
          <div key={p.name} style={cardStyle()}>
            <div style={{ display: 'flex', alignItems: 'flex-start', gap: 12 }}>
              <div
                style={{
                  width: 40,
                  height: 40,
                  borderRadius: 'var(--radius-sm)',
                  background: 'var(--bg-hover, #1a1a1a)',
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'center',
                  fontWeight: 700,
                  color: 'var(--accent-cyan)',
                  fontFamily: 'var(--font-mono)',
                }}
              >
                {label.icon}
              </div>
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{ fontWeight: 600, marginBottom: 4 }}>{label.title}</div>
                <div
                  style={{
                    fontSize: 13,
                    color: 'var(--text-secondary)',
                    marginBottom: 8,
                  }}
                >
                  {label.description}
                </div>

                {!p.configured && (
                  <div
                    style={{
                      fontSize: 12,
                      color: 'var(--text-muted)',
                      fontStyle: 'italic',
                    }}
                  >
                    Not configured on the server.{' '}
                    <a
                      href="/docs/cloud-storage/SETUP.md"
                      target="_blank"
                      rel="noopener noreferrer"
                      style={{ color: 'var(--accent-cyan)' }}
                    >
                      Setup guide &rarr;
                    </a>
                  </div>
                )}

                {p.configured && !p.connected && (
                  <button
                    type="button"
                    onClick={() => handleConnect(p.name)}
                    disabled={busy === p.name}
                    style={buttonStyle('primary')}
                  >
                    {busy === p.name ? 'Opening...' : 'Connect'}
                  </button>
                )}

                {p.configured && p.connected && p.account && (
                  <div>
                    <div
                      style={{
                        fontSize: 13,
                        color: 'var(--text-primary)',
                        marginBottom: 6,
                      }}
                    >
                      Connected as{' '}
                      <strong>{p.account.display_name}</strong>
                    </div>
                    {p.account.token_expires_at && (
                      <div
                        style={{
                          fontSize: 11,
                          color: 'var(--text-muted)',
                          marginBottom: 8,
                        }}
                      >
                        Token expires: {new Date(p.account.token_expires_at).toLocaleString()}
                      </div>
                    )}
                    <button
                      type="button"
                      onClick={() => handleDisconnect(p.account.id, p.name)}
                      disabled={busy === p.name}
                      style={buttonStyle('danger')}
                    >
                      {busy === p.name ? 'Working...' : 'Disconnect'}
                    </button>
                  </div>
                )}
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}
