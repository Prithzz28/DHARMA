import { useEffect, useRef, useState } from 'react';
import { motion, useInView } from 'framer-motion';
import {
  Wallet, Copy, Check, ShieldCheck, ShieldAlert, ShieldX, AlertTriangle, Terminal, Plug,
} from 'lucide-react';
import { addChainParams, fetchRpcInfo, rpcUrl } from '../services/api';
import { modelMetrics, pct } from '../data/modelMetrics';

// Shown until /rpc/info answers (or if the API is unreachable).
const DEFAULT_INFO = {
  chain_id: 11155111,
  chain_id_hex: '0xaa36a7',
  network_name: 'Sepolia',
  currency_symbol: 'SepoliaETH',
  explorer_url: 'https://sepolia.etherscan.io',
  model_loaded: false,
  warn_threshold: null,
  block_threshold: 1,
  model_blocks: false,
  flagged_addresses: 1763,
  override_window_seconds: 120,
};

const percent = (value) => (value == null ? '—' : `${Math.round(value * 100)}%`);

/** Held-out pending-mode figures at a threshold, when that threshold was evaluated. */
function evaluationAt(threshold) {
  return modelMetrics.thresholdCurve.find((row) => Math.abs(row.threshold - threshold) < 0.005);
}

function CopyRow({ label, value, mono = true }) {
  const [copied, setCopied] = useState(false);
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(String(value));
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    } catch {
      // Clipboard blocked (e.g. insecure origin) — the value stays selectable.
    }
  };
  return (
    <div className="flex items-center justify-between gap-3 py-2.5 border-b border-white/5 last:border-0">
      <span className="text-xs text-gray-500 shrink-0">{label}</span>
      <div className="flex items-center gap-2 min-w-0">
        <span className={`text-sm text-white truncate select-all ${mono ? 'font-mono' : ''}`}>{value}</span>
        <button
          type="button"
          onClick={copy}
          aria-label={`Copy ${label}`}
          className="p-1.5 rounded-lg text-gray-500 hover:text-cyan-300 hover:bg-white/5 transition-colors shrink-0"
        >
          {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
        </button>
      </div>
    </div>
  );
}

function snippets(url, info) {
  return {
    curl: `curl -s ${url} \\
  -H "Content-Type: application/json" \\
  -d '{"jsonrpc":"2.0","id":1,"method":"eth_chainId","params":[]}'`,
    'ethers.js': `import { JsonRpcProvider, Wallet } from 'ethers';

const provider = new JsonRpcProvider('${url}');
const wallet = new Wallet(process.env.PRIVATE_KEY, provider);

try {
  const tx = await wallet.sendTransaction({ to, value });
  console.log('Allowed:', tx.hash);
} catch (err) {
  // Blocked by DeFiGuard: JSON-RPC error -32003 with the reason
  console.error(err.error?.message ?? err.message);
}`,
    'web3.py': `from web3 import Web3

w3 = Web3(Web3.HTTPProvider("${url}"))
assert w3.eth.chain_id == ${info.chain_id}

signed = w3.eth.account.sign_transaction(tx, private_key)
try:
    tx_hash = w3.eth.send_raw_transaction(signed.raw_transaction)
except ValueError as err:   # blocked: {'code': -32003, 'message': ...}
    print(err)`,
  };
}

export default function ConnectWallet() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-100px' });
  const [info, setInfo] = useState(DEFAULT_INFO);
  const [online, setOnline] = useState(null);
  const [addStatus, setAddStatus] = useState(null);
  const [tab, setTab] = useState('curl');
  const url = rpcUrl();

  useEffect(() => {
    const controller = new AbortController();
    fetchRpcInfo({ signal: controller.signal })
      .then((data) => {
        // Merge so fields an older API version lacks keep their defaults.
        setInfo({ ...DEFAULT_INFO, ...data });
        setOnline(true);
      })
      .catch(() => {
        if (!controller.signal.aborted) setOnline(false);
      });
    return () => controller.abort();
  }, []);

  const addToWallet = async () => {
    if (!window.ethereum) {
      setAddStatus({ ok: false, text: 'No browser wallet detected. Use the manual steps below.' });
      return;
    }
    try {
      await window.ethereum.request({ method: 'wallet_addEthereumChain', params: [addChainParams(info, url)] });
      setAddStatus({ ok: true, text: 'Network added. Select it in your wallet to send through the firewall.' });
    } catch (err) {
      setAddStatus({
        ok: false,
        text: `${err?.message || 'The wallet refused the request.'} Use the manual steps below.`,
      });
    }
  };

  const warnThreshold = info.warn_threshold ?? modelMetrics.threshold;
  const warnEval = evaluationAt(warnThreshold);
  const window_s = info.override_window_seconds;
  const listed = (info.flagged_addresses ?? 0).toLocaleString();
  const bands = [
    {
      icon: ShieldX, color: 'text-red-400', border: 'border-red-500/20', bg: 'bg-red-500/5',
      title: 'Blocked', range: info.model_blocks
        ? `counterparty red flag, or model risk ≥ ${percent(info.block_threshold)}`
        : 'counterparty red flag',
      text: `Paying or approving one of ${listed} fraud-labelled wallets, or granting an unlimited or approve-all
        permission to a plain wallet. Never broadcast; your wallet shows the reason.${
        window_s ? ` Sending the same transaction again within ${window_s}s overrides the block.` : ''
      }`,
    },
    {
      icon: ShieldAlert, color: 'text-amber-400', border: 'border-amber-500/20', bg: 'bg-amber-500/5',
      title: 'Warned', range: `model risk ≥ ${percent(warnThreshold)}, or a softer counterparty finding`,
      text: 'Forwarded, and listed with its signals in the live feed — for example an unlimited approval to a contract.',
    },
    {
      icon: ShieldCheck, color: 'text-emerald-400', border: 'border-emerald-500/20', bg: 'bg-emerald-500/5',
      title: 'Allowed', range: `no finding, model risk below ${percent(warnThreshold)}`,
      text: 'Forwarded to the network straight away.',
    },
  ];

  return (
    <section id="connect" className="py-28 relative overflow-hidden">
      <div
        className="absolute inset-0"
        style={{ background: 'radial-gradient(ellipse 70% 50% at 70% 40%, rgba(6,182,212,0.06) 0%, transparent 60%)' }}
      />
      <div ref={ref} className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="grid lg:grid-cols-2 gap-12 items-start">
          {/* ── Left: network details ─────────────────────────────────────── */}
          <div>
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass border border-cyan-500/20 text-sm text-cyan-400 font-medium mb-6"
            >
              <Plug className="w-4 h-4" />
              Wallet Firewall
            </motion.div>
            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.1 }}
              className="text-4xl lg:text-5xl font-black text-white mb-4 font-[Manrope]"
            >
              Connect Your <span className="text-gradient">Wallet</span>
            </motion.h2>
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.2 }}
              className="text-gray-400 text-lg mb-8 leading-relaxed"
            >
              Add the DeFiGuard RPC to your wallet as a network. Nothing to install: every transaction you
              send is checked before it reaches the chain, and everything else passes straight through.
            </motion.p>

            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.3 }}
              className="glass-card rounded-2xl p-6 border border-white/10"
            >
              <div className="flex items-center gap-2 mb-3">
                <Wallet className="w-4 h-4 text-cyan-400" />
                <span className="text-sm font-bold text-white">Network details</span>
                <span
                  className={`ml-auto text-[10px] font-mono px-2 py-0.5 rounded-full ${
                    online === null ? 'bg-white/5 text-gray-500'
                      : online && info.model_loaded ? 'bg-emerald-500/15 text-emerald-300'
                        : 'bg-red-500/15 text-red-300'
                  }`}
                >
                  {online === null ? 'checking…' : online && info.model_loaded ? 'firewall online' : 'firewall offline'}
                </span>
              </div>
              <CopyRow label="Network name" value={`${info.network_name} (DeFiGuard)`} mono={false} />
              <CopyRow label="RPC URL" value={url} />
              <CopyRow label="Chain ID" value={info.chain_id} />
              <CopyRow label="Currency symbol" value={info.currency_symbol} />
              {info.explorer_url && <CopyRow label="Block explorer" value={info.explorer_url} />}

              <button
                type="button"
                onClick={addToWallet}
                className="mt-5 w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-bold text-white bg-gradient-to-r from-cyan-500 to-blue-600 hover:shadow-lg hover:shadow-cyan-500/30 transition-all"
              >
                <Wallet className="w-4 h-4" />
                Add to MetaMask
              </button>
              {addStatus && (
                <p className={`mt-3 text-xs ${addStatus.ok ? 'text-emerald-400' : 'text-amber-400'}`}>{addStatus.text}</p>
              )}
            </motion.div>

            <div className="mt-6 glass-card rounded-2xl p-6 border border-white/10">
              <div className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3">Manual setup (MetaMask)</div>
              <ol className="space-y-2 text-sm text-gray-400 list-decimal list-inside">
                <li>Open the network menu and choose <span className="text-white">Add a custom network</span>.</li>
                <li>Paste the RPC URL and chain ID above, then fill in the name, symbol and explorer.</li>
                <li>If {info.network_name} is already listed, edit it, add the RPC URL as a new endpoint and select it.</li>
                <li>Save, switch to the network, and send a transaction as usual.</li>
              </ol>
            </div>
          </div>

          {/* ── Right: policy + developer snippets ────────────────────────── */}
          <div className="space-y-6">
            <div className="glass-card rounded-2xl p-6 border border-white/10">
              <div className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-4">How each transaction is handled</div>
              <div className="space-y-3">
                {bands.map(({ icon: Icon, color, border, bg, title, range, text }) => (
                  <div key={title} className={`flex gap-3 p-4 rounded-xl border ${border} ${bg}`}>
                    <Icon className={`w-5 h-5 mt-0.5 shrink-0 ${color}`} />
                    <div>
                      <div className="flex flex-wrap items-baseline gap-x-2">
                        <span className={`text-sm font-bold ${color}`}>{title}</span>
                        <span className="text-xs font-mono text-gray-500">{range}</span>
                      </div>
                      <p className="text-sm text-gray-400 mt-0.5">{text}</p>
                    </div>
                  </div>
                ))}
              </div>
              <p className="mt-4 text-xs text-gray-500 leading-relaxed">
                The model scores how closely a transaction resembles ones <em>sent by</em> fraud-labelled wallets
                (the dataset labels whole wallets), so it warns but does not block on its own.
                {warnEval && (
                  <>
                    {' '}At its {percent(warnThreshold)} threshold, held-out evaluation flagged{' '}
                    <span className="text-gray-300">{pct(warnEval.recall)}</span> of fraud-labelled and{' '}
                    <span className="text-gray-300">{pct(warnEval.fpr)}</span> of legitimate pending transactions.
                  </>
                )}
              </p>
            </div>

            <div className="flex gap-3 p-4 rounded-xl border border-amber-500/20 bg-amber-500/5">
              <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0 mt-0.5" />
              <p className="text-sm text-gray-400 leading-relaxed">
                <span className="text-amber-300 font-semibold">Testnet demo.</span> Fraud-labelled wallets tend to
                be young, so accounts with few past transactions get a high model score: expect warnings on
                ordinary transfers from a fresh wallet. The flagged wallets are Ethereum mainnet addresses from the
                research dataset. Your transactions pass through this server, which keeps the last 50 verdicts with
                masked addresses.
              </p>
            </div>

            <div className="glass-card rounded-2xl border border-white/10 overflow-hidden">
              <div className="flex items-center gap-1 px-4 pt-4 border-b border-white/5">
                <Terminal className="w-4 h-4 text-gray-500 mr-2" />
                {Object.keys(snippets(url, info)).map((name) => (
                  <button
                    key={name}
                    type="button"
                    onClick={() => setTab(name)}
                    className={`px-3 py-2 text-xs font-mono border-b-2 -mb-px transition-colors ${
                      tab === name ? 'border-cyan-400 text-white' : 'border-transparent text-gray-500 hover:text-gray-300'
                    }`}
                  >
                    {name}
                  </button>
                ))}
              </div>
              <pre className="p-4 text-xs text-gray-300 font-mono overflow-x-auto leading-relaxed">
                <code>{snippets(url, info)[tab]}</code>
              </pre>
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}
