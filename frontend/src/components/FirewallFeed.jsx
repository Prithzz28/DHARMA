import { useEffect, useState } from 'react';
import { Radio, ShieldCheck, ShieldAlert, ShieldX, ShieldQuestion, ExternalLink } from 'lucide-react';
import { fetchRecentInterceptions, fetchRpcInfo } from '../services/api';

const POLL_MS = 5000;

const VERDICTS = {
  allowed: { label: 'Allowed', icon: ShieldCheck, color: 'text-emerald-400', chip: 'bg-emerald-500/10 border-emerald-500/20' },
  warned: { label: 'Warned', icon: ShieldAlert, color: 'text-amber-400', chip: 'bg-amber-500/10 border-amber-500/20' },
  blocked: { label: 'Blocked', icon: ShieldX, color: 'text-red-400', chip: 'bg-red-500/10 border-red-500/20' },
  overridden: { label: 'Overridden', icon: ShieldQuestion, color: 'text-violet-400', chip: 'bg-violet-500/10 border-violet-500/20' },
};

const formatTime = (iso) => new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });

function Row({ item, explorer }) {
  const v = VERDICTS[item.verdict];
  const Icon = v.icon;
  return (
    <li className="py-3 border-b border-white/5 last:border-0">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
        <span className={`inline-flex items-center gap-1.5 font-bold w-28 ${v.color}`}>
          <Icon className="w-4 h-4" /> {v.label}
        </span>
        <span className="font-mono text-gray-300">
          {item.from_address} → {item.to_address}
        </span>
        <span className="font-mono text-gray-400">{item.value_eth.toFixed(4)} ETH</span>
        <span className="font-mono text-gray-400" title="DeFiGuard sender-pattern score">
          model {Math.round(item.fraud_probability * 100)}%
        </span>
        <span className="ml-auto text-xs text-gray-500 font-mono">{formatTime(item.time)}</span>
      </div>
      {(item.counterparty.length > 0 || item.model_signals.length > 0 || item.tx_hash || item.upstream_error) && (
        <div className="mt-1 pl-0 sm:pl-32 text-xs text-gray-500 space-y-0.5">
          {item.counterparty.map((finding) => (
            <p key={finding.code + finding.message}>
              <span className={finding.severity === 'block' ? 'text-red-400' : 'text-amber-400'}>Counterparty:</span>{' '}
              {finding.message}
            </p>
          ))}
          {item.model_signals.length > 0 && (
            <p>
              <span className="text-gray-400">Sender pattern:</span> {item.model_signals[0]}
            </p>
          )}
          {item.tx_hash && explorer && (
            <a
              href={`${explorer.replace(/\/+$/, '')}/tx/${item.tx_hash}`}
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 font-mono text-cyan-400 hover:text-cyan-300"
            >
              {item.tx_hash.slice(0, 10)}…{item.tx_hash.slice(-6)} <ExternalLink className="w-3 h-3" />
            </a>
          )}
          {item.upstream_error && <p className="text-amber-400/80">Node rejected it: {item.upstream_error}</p>}
        </div>
      )}
    </li>
  );
}

export default function FirewallFeed() {
  const [feed, setFeed] = useState(null);
  const [explorer, setExplorer] = useState('');
  const [error, setError] = useState(null);

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    fetchRpcInfo({ signal }).then((info) => setExplorer(info.explorer_url)).catch(() => {});

    const poll = async () => {
      if (document.visibilityState === 'hidden') return;
      try {
        setFeed(await fetchRecentInterceptions({ limit: 15, signal }));
        setError(null);
      } catch (err) {
        if (!signal.aborted) setError(err.message);
      }
    };
    poll();
    const interval = setInterval(poll, POLL_MS);
    return () => {
      controller.abort();
      clearInterval(interval);
    };
  }, []);

  return (
    <section id="live-feed" className="pb-28 relative">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="glass-card rounded-2xl p-6 border border-white/10">
          <div className="flex flex-wrap items-center gap-3 mb-4">
            <Radio className="w-4 h-4 text-cyan-400" />
            <h3 className="text-lg font-bold text-white">Live firewall feed</h3>
            <span className="text-xs text-gray-500">Transactions sent through the RPC, newest first · refreshes every 5s</span>
            <div className="flex flex-wrap gap-2 sm:ml-auto">
              {Object.entries(VERDICTS).map(([key, v]) => (
                <span key={key} className={`px-2.5 py-1 rounded-lg border text-xs font-mono ${v.chip} ${v.color}`}>
                  {v.label} {feed?.totals?.[key] ?? 0}
                </span>
              ))}
            </div>
          </div>

          {error && <p className="text-sm text-red-400 py-4">Feed unavailable: {error}</p>}
          {!error && feed && feed.items.length === 0 && (
            <p className="text-sm text-gray-500 py-6 text-center">
              No transactions yet. Add the RPC to your wallet and send a testnet transaction.
            </p>
          )}
          {!error && feed && feed.items.length > 0 && (
            <ul>
              {feed.items.map((item) => (
                <Row key={item.id} item={item} explorer={explorer} />
              ))}
            </ul>
          )}
          {!error && !feed && <p className="text-sm text-gray-500 py-6 text-center">Loading…</p>}
        </div>
      </div>
    </section>
  );
}
