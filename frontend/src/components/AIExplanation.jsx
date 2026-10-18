import { useRef, useState, useEffect, useCallback } from 'react';
import { motion, useInView, AnimatePresence } from 'framer-motion';
import {
  Lightbulb, Info, Sparkles, RefreshCw, ShieldAlert, ShieldCheck,
  Send, Zap, Cpu, Layers, AlertTriangle, Activity,
  ChevronDown, ChevronUp,
} from 'lucide-react';
import { predictTransaction, fetchRiskExplanation, buildFeatureVector } from '../services/api';

// ─── Feature metadata for the 7 RF model features ────────────────────────────
const MODEL_FEATURES = [
  {
    key: 'length_transaction_hash',
    label: 'Tx Hash Length',
    desc: 'Length of the transaction hash string. Standard hashes are 66 characters — anomalous lengths flag potential spoofing.',
  },
  {
    key: 'length_to',
    label: 'Recipient Addr Length',
    desc: 'Length of the recipient address string. Standard Ethereum addresses are 42 characters.',
  },
  {
    key: 'block_number',
    label: 'Block Number',
    desc: 'Block height at which the transaction is included. Unusual block numbers can indicate test / spam activity.',
  },
  {
    key: 'gas_efficiency',
    label: 'Gas Efficiency',
    desc: 'Ratio of gas consumed to gas limit. A ratio close to 1.0 suggests tightly-estimated gas (common in exploit bots).',
  },
  {
    key: 'chain_id',
    label: 'Chain ID',
    desc: 'Ethereum network chain ID. 1 = Mainnet, 11155111 = Sepolia testnet.',
  },
  {
    key: 'effective_gas_price',
    label: 'Effective Gas Price',
    desc: 'Effective gas price paid in wei. Extremely high values indicate front-running or gas manipulation.',
  },
  {
    key: 'is_same_address',
    label: 'Self-Transfer Flag',
    desc: '1.0 when sender and recipient are the same address — a common wallet-drainer pattern.',
  },
];

// ─── Normalisation reference max values (for display bars only) ───────────────
const FEATURE_MAX = {
  length_transaction_hash: 66,
  length_to: 42,
  block_number: 20_000_000,
  gas_efficiency: 1.0,
  chain_id: 11155111,
  effective_gas_price: 500e9,
  is_same_address: 1.0,
};

// ─── Preset attack scenarios ──────────────────────────────────────────────────
const PRESETS = [
  {
    label: 'Phishing Drainer',
    color: 'bg-red-500/15 border-red-500/30 text-red-400 hover:bg-red-500/25',
    data: {
      from: '0x742d35Cc6634C0532925a3b844Bc9e7595f44e2a',
      to: '0x742d35Cc6634C0532925a3b844Bc9e7595f44e2a',
      amount: '0.00',
      gasPrice: '204',
      gasLimit: '848000',
      blockNumber: '1632637',
      hash: '0x' + 'a'.repeat(64),
      logCount: '0',
    },
  },
  {
    label: 'Honeypot Scam',
    color: 'bg-orange-500/15 border-orange-500/30 text-orange-400 hover:bg-orange-500/25',
    data: {
      from: '0x1c9b4f2a3d5e6f7890abcdef1234567890abcdef',
      to: '0xDeaD000000000000000000000000000000000000',
      amount: '10.0',
      gasPrice: '50',
      gasLimit: '1200000',
      blockNumber: '1680319',
      hash: '0x' + 'b'.repeat(64),
      logCount: '3',
    },
  },
  {
    label: 'Reentrancy Exploit',
    color: 'bg-rose-500/15 border-rose-500/30 text-rose-400 hover:bg-rose-500/25',
    data: {
      from: '0x1234567890abcdef1234567890abcdef12345678',
      to: '0xfedcba0987654321fedcba0987654321fedcba09',
      amount: '2.0',
      gasPrice: '100',
      gasLimit: '2490000',
      blockNumber: '1727488',
      hash: '0x' + 'd'.repeat(64),
      logCount: '5',
    },
  },
  {
    label: 'MEV Bot Sandwich',
    color: 'bg-purple-500/15 border-purple-500/30 text-purple-400 hover:bg-purple-500/25',
    data: {
      from: '0x3333333333333333333333333333333333333333',
      to: '0x4444444444444444444444444444444444444444',
      amount: '25.0',
      gasPrice: '250',
      gasLimit: '350000',
      blockNumber: '1650000',
      hash: '0x' + 'e'.repeat(64),
      logCount: '2',
    },
  },
  {
    label: 'Legit Transfer',
    color: 'bg-green-500/15 border-green-500/30 text-green-400 hover:bg-green-500/25',
    data: {
      from: '0x3b2a1f88e4c5d6a7890bcdef2345678901bcde22',
      to: '0x8c5d7e12f3a4b5c6d7890abcdef3456789012cd33',
      amount: '0.45',
      gasPrice: '22',
      gasLimit: '21000',
      blockNumber: '19000000',
      hash: '0x' + 'c'.repeat(64),
      logCount: '0',
    },
  },
  {
    label: 'Legit DEX Swap',
    color: 'bg-cyan-500/15 border-cyan-500/30 text-cyan-400 hover:bg-cyan-500/25',
    data: {
      from: '0x5555555555555555555555555555555555555555',
      to: '0xE592427A0AEce92De3Edee1F18E0157C05861564',
      amount: '1.5',
      gasPrice: '28',
      gasLimit: '145000',
      blockNumber: '19000015',
      hash: '0x' + 'f'.repeat(64),
      logCount: '4',
    },
  },
];

// ─── Feature bar component ────────────────────────────────────────────────────
function FeatureBar({ feature, value, maxValue, index, inView }) {
  const normalizedPct = maxValue > 0 ? Math.min((Math.abs(value) / maxValue) * 100, 100) : 0;
  const colorClass =
    normalizedPct > 70
      ? 'from-red-500 to-orange-500'
      : normalizedPct > 40
      ? 'from-yellow-500 to-amber-400'
      : 'from-green-500 to-teal-500';

  const displayValue =
    value >= 1e9
      ? `${(value / 1e9).toFixed(1)} Gwei`
      : value >= 1e6
      ? `${(value / 1e6).toFixed(1)}M`
      : typeof value === 'number' && !Number.isInteger(value)
      ? value.toFixed(3)
      : value.toLocaleString();

  return (
    <motion.div
      initial={{ opacity: 0, x: -20 }}
      animate={inView ? { opacity: 1, x: 0 } : {}}
      transition={{ delay: index * 0.08, duration: 0.4 }}
      className="group"
    >
      <div className="flex justify-between items-center mb-2">
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-white">{feature.label}</span>
          <div className="relative group/tooltip">
            <Info className="w-3.5 h-3.5 text-gray-600 cursor-help" />
            <div className="absolute left-0 bottom-full mb-2 w-56 bg-[#1a2332] border border-white/10 rounded-xl p-3 text-xs text-gray-400 opacity-0 group-hover/tooltip:opacity-100 transition-opacity pointer-events-none z-10 shadow-xl">
              {feature.desc}
            </div>
          </div>
        </div>
        <span className="text-sm font-black text-white font-mono">{displayValue}</span>
      </div>
      <div className="h-2.5 bg-white/5 rounded-full overflow-hidden">
        <motion.div
          initial={{ width: 0 }}
          animate={inView ? { width: `${Math.max(normalizedPct, 2)}%` } : {}}
          transition={{ delay: 0.2 + index * 0.08, duration: 0.7, ease: 'easeOut' }}
          className={`h-full bg-gradient-to-r ${colorClass} rounded-full relative`}
        >
          <div className="absolute right-0 top-0 h-full w-3 bg-white/30 rounded-full blur-sm" />
        </motion.div>
      </div>
    </motion.div>
  );
}

// ─── Input Field ──────────────────────────────────────────────────────────────
function InputField({ label, placeholder, value, onChange, mono = true, hint }) {
  return (
    <div>
      <div className="flex justify-between items-center mb-1">
        <label className="text-xs text-gray-500">{label}</label>
        {hint && <span className="text-[10px] text-gray-600 font-mono">{hint}</span>}
      </div>
      <input
        type="text"
        placeholder={placeholder}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className={`w-full px-3 py-2 rounded-xl bg-white/5 border border-white/10 text-sm text-white placeholder-gray-600 focus:border-purple-500/50 focus:outline-none transition-colors ${
          mono ? 'font-mono' : ''
        }`}
      />
    </div>
  );
}

// ─── Main component ───────────────────────────────────────────────────────────
export default function AIExplanation() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-80px' });

  const [formData, setFormData] = useState({
    from: '',
    to: '',
    amount: '',
    gasPrice: '',
    gasLimit: '21000',
    blockNumber: '',
    hash: '',
    logCount: '0',
  });

  const [loading, setLoading] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [showAdvanced, setShowAdvanced] = useState(false);

  // ── Core analysis function — always uses real inference API ─────────────────
  const analyzeData = useCallback(async (data, isAuto = false) => {
    if (!data.from && !data.to) {
      if (!isAuto) {
        setError('Enter at least a From or To address, or pick a preset scenario.');
      }
      return;
    }

    if (isAuto) {
      setIsScanning(true);
    } else {
      setLoading(true);
      setError(null);
    }

    try {
      // Build feature vector from form data (no hardcoded fraud signals)
      const featureVector = buildFeatureVector(data);

      // 1. POST /predict — get fraud score from ensemble model
      const prediction = await predictTransaction(featureVector);

      // 2. POST /predict/explain — get AI or rule-based explanation
      let explanation;
      try {
        explanation = await fetchRiskExplanation(featureVector, data);
      } catch {
        // Fallback if explain endpoint fails
        explanation = {
          risk_level: prediction.is_fraud ? 'high' : 'low',
          reasons: prediction.is_fraud
            ? ['Model detected anomalous feature patterns consistent with fraudulent activity.']
            : ['Transaction parameters align with legitimate baseline activity.'],
          recommendation: prediction.is_fraud
            ? 'CRITICAL: Do NOT sign this transaction. High probability of financial loss.'
            : 'Transaction appears safe. Normal baseline parameters detected.',
          is_fallback: true,
          exec_time_ms: 0,
        };
      }

      setResult({ prediction, explanation, featureVector });
      setError(null);
    } catch (err) {
      if (!isAuto) {
        setError(
          `Backend error: ${err.message}. Make sure the inference engine is running on port 8001.`
        );
      }
    } finally {
      setLoading(false);
      setIsScanning(false);
    }
  }, []);

  // ── Field change handler ────────────────────────────────────────────────────
  const handleFieldChange = (field, value) => {
    if (error) setError(null);
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  // ── Preset handler ──────────────────────────────────────────────────────────
  const handlePreset = (preset) => {
    setError(null);
    setResult(null);
    const updated = { ...formData, ...preset.data };
    setFormData(updated);
    analyzeData(updated, false);
  };

  // ── Manual scan ─────────────────────────────────────────────────────────────
  const handleAnalyze = () => {
    setError(null);
    analyzeData(formData, false);
  };

  // ── Debounced auto-analysis on field changes ─────────────────────────────────
  useEffect(() => {
    if (!formData.from && !formData.to) return;
    const timer = setTimeout(() => {
      analyzeData(formData, true);
    }, 600);
    return () => clearTimeout(timer);
  }, [formData, analyzeData]);

  return (
    <section id="ai-explanation" className="py-28 relative overflow-hidden">
      <div
        className="absolute inset-0"
        style={{
          background:
            'radial-gradient(ellipse 70% 50% at 30% 50%, rgba(139,92,246,0.05) 0%, transparent 60%)',
        }}
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div className="grid lg:grid-cols-2 gap-16 items-start">
          {/* ── Left: Form ────────────────────────────────────────────────────── */}
          <div ref={ref}>
            {/* Section badge */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass border border-purple-500/20 text-sm text-purple-400 font-medium mb-6"
            >
              <Lightbulb className="w-4 h-4" />
              Explainable Security (XAI)
            </motion.div>

            <motion.h2
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.1 }}
              className="text-4xl lg:text-5xl font-black text-white mb-4 font-[Manrope]"
            >
              Live Transaction{' '}
              <span className="text-gradient">Scanner</span>
            </motion.h2>

            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.2 }}
              className="text-gray-400 text-lg mb-8 leading-relaxed"
            >
              Enter transaction details or pick a preset scenario. Every scan routes through the live
              Random Forest + DeFiGuard ensemble inference API — no hardcoded rules.
            </motion.p>

            {/* Feature highlights */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.3 }}
              className="space-y-3 mb-8"
            >
              {[
                'Routes through live /predict + /predict/explain API',
                'Dual-model ensemble: Random Forest + DeFiGuard neural net',
                'AI-generated risk explanations via Groq (Llama 3)',
                'Feature-level breakdown from real model output',
              ].map((point) => (
                <div key={point} className="flex items-center gap-3">
                  <div className="w-5 h-5 rounded-full bg-gradient-to-br from-purple-500 to-violet-600 flex items-center justify-center flex-shrink-0">
                    <span className="text-white text-xs font-bold">✓</span>
                  </div>
                  <span className="text-gray-300 text-sm">{point}</span>
                </div>
              ))}
            </motion.div>

            {/* Preset buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.35 }}
              className="mb-6"
            >
              <div className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3">
                Quick Scenarios
              </div>
              <div className="flex flex-wrap gap-2">
                {PRESETS.map((preset) => (
                  <button
                    key={preset.label}
                    onClick={() => handlePreset(preset)}
                    disabled={loading}
                    className={`flex items-center gap-1.5 px-3.5 py-2 rounded-xl text-xs font-bold border transition-all disabled:opacity-50 ${preset.color}`}
                  >
                    {preset.label}
                  </button>
                ))}
              </div>
            </motion.div>

            {/* Transaction Input Form */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={inView ? { opacity: 1, y: 0 } : {}}
              transition={{ delay: 0.4 }}
              className="glass-card rounded-2xl p-6 border border-white/10 space-y-4"
            >
              <div className="flex items-center gap-2 mb-1">
                <Activity className="w-4 h-4 text-purple-400" />
                <span className="text-sm font-bold text-white">Custom Transaction Analysis</span>
                {isScanning && (
                  <span className="ml-auto inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] bg-purple-500/20 text-purple-300 font-mono animate-pulse">
                    <RefreshCw className="w-2.5 h-2.5 animate-spin" />
                    Scanning…
                  </span>
                )}
              </div>

              {/* Core fields */}
              <div className="grid grid-cols-2 gap-3">
                <div className="col-span-2 sm:col-span-1">
                  <InputField
                    label="From Address"
                    placeholder="0x742d...f44e"
                    value={formData.from}
                    onChange={(v) => handleFieldChange('from', v)}
                  />
                </div>
                <div className="col-span-2 sm:col-span-1">
                  <InputField
                    label="To Address"
                    placeholder="0x1c9b...8d44"
                    value={formData.to}
                    onChange={(v) => handleFieldChange('to', v)}
                  />
                </div>
                <div>
                  <InputField
                    label="Amount (ETH)"
                    placeholder="0.45"
                    value={formData.amount}
                    onChange={(v) => handleFieldChange('amount', v)}
                  />
                </div>
                <div>
                  <InputField
                    label="Gas Price (Gwei)"
                    placeholder="22"
                    value={formData.gasPrice}
                    onChange={(v) => handleFieldChange('gasPrice', v)}
                  />
                </div>
                <div className="col-span-2">
                  <InputField
                    label="Gas Limit (Units)"
                    placeholder="21000"
                    hint="Transfer: 21,000 · Contract: ~150k–2.5M"
                    value={formData.gasLimit}
                    onChange={(v) => handleFieldChange('gasLimit', v)}
                  />
                </div>
              </div>

              {/* Advanced toggle */}
              <button
                type="button"
                onClick={() => setShowAdvanced((s) => !s)}
                className="flex items-center gap-1.5 text-xs text-gray-500 hover:text-gray-300 transition-colors"
              >
                {showAdvanced ? (
                  <ChevronUp className="w-3.5 h-3.5" />
                ) : (
                  <ChevronDown className="w-3.5 h-3.5" />
                )}
                {showAdvanced ? 'Hide' : 'Show'} advanced fields
              </button>

              <AnimatePresence>
                {showAdvanced && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    className="overflow-hidden"
                  >
                    <div className="grid grid-cols-2 gap-3 pt-2 border-t border-white/5">
                      <div className="col-span-2">
                        <InputField
                          label="Transaction Hash"
                          placeholder="0xabc123… (66 chars)"
                          value={formData.hash}
                          onChange={(v) => handleFieldChange('hash', v)}
                        />
                      </div>
                      <div>
                        <InputField
                          label="Block Number"
                          placeholder="19000000"
                          value={formData.blockNumber}
                          onChange={(v) => handleFieldChange('blockNumber', v)}
                        />
                      </div>
                      <div>
                        <InputField
                          label="Log Count"
                          placeholder="0"
                          value={formData.logCount}
                          onChange={(v) => handleFieldChange('logCount', v)}
                        />
                      </div>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Analyse button */}
              <button
                onClick={handleAnalyze}
                disabled={loading}
                className="w-full flex items-center justify-center gap-2 px-6 py-3 rounded-xl font-bold text-white bg-gradient-to-r from-purple-500 to-violet-600 hover:shadow-lg hover:shadow-purple-500/30 transition-all disabled:opacity-50"
              >
                {loading ? (
                  <>
                    <RefreshCw className="w-4 h-4 animate-spin" />
                    Running Ensemble Inference…
                  </>
                ) : (
                  <>
                    <Send className="w-4 h-4" />
                    Scan Transaction
                  </>
                )}
              </button>

              {/* Error message */}
              {error && (
                <div className="flex items-start justify-between gap-3 px-4 py-3 rounded-xl bg-red-500/10 border border-red-500/20 text-xs text-red-400">
                  <div className="flex items-center gap-2">
                    <AlertTriangle className="w-4 h-4 flex-shrink-0 text-red-400" />
                    <span>{error}</span>
                  </div>
                  <button
                    onClick={() => setError(null)}
                    className="text-red-400 hover:text-white transition-colors ml-auto font-bold text-sm leading-none"
                    title="Dismiss"
                  >
                    ×
                  </button>
                </div>
              )}
            </motion.div>
          </div>

          {/* ── Right: Results Panel ──────────────────────────────────────────── */}
          <motion.div
            initial={{ opacity: 0, x: 40 }}
            animate={inView ? { opacity: 1, x: 0 } : {}}
            transition={{ delay: 0.2, duration: 0.7 }}
            className="glass-card rounded-3xl p-8 border border-purple-500/15 lg:sticky lg:top-24"
            style={{ boxShadow: '0 0 40px rgba(139,92,246,0.1)' }}
          >
            <AnimatePresence mode="wait">
              {result ? (
                <motion.div
                  key="result"
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                >
                  {/* Card header */}
                  <div className="flex items-center justify-between mb-6">
                    <div>
                      <div className="flex items-center gap-2">
                        <h3 className="text-lg font-bold text-white font-[Manrope]">
                          Risk Analysis
                        </h3>
                        {isScanning && (
                          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] bg-purple-500/20 text-purple-300 font-mono animate-pulse">
                            <RefreshCw className="w-2.5 h-2.5 animate-spin" />
                            Updating…
                          </span>
                        )}
                      </div>
                      <p className="text-xs text-gray-500">
                        {result.explanation.is_fallback
                          ? 'Rule-based explainer'
                          : 'AI-generated explanation (Groq)'}
                        {' · '}
                        {result.prediction.exec_time_ms.toFixed(1)}ms inference
                      </p>
                    </div>
                    <div
                      className={`flex items-center gap-2 text-xs font-bold ${
                        result.explanation.risk_level === 'high'
                          ? 'text-red-400'
                          : result.explanation.risk_level === 'medium'
                          ? 'text-yellow-400'
                          : 'text-green-400'
                      }`}
                    >
                      <div
                        className={`w-2 h-2 rounded-full animate-pulse ${
                          result.explanation.risk_level === 'high'
                            ? 'bg-red-400'
                            : result.explanation.risk_level === 'medium'
                            ? 'bg-yellow-400'
                            : 'bg-green-400'
                        }`}
                      />
                      {result.explanation.risk_level.toUpperCase()} RISK
                    </div>
                  </div>

                  {/* Feature importance bars */}
                  <div className="space-y-4 mb-6">
                    {MODEL_FEATURES.map((feature, idx) => (
                      <FeatureBar
                        key={feature.key}
                        feature={feature}
                        value={result.featureVector[feature.key] || 0}
                        maxValue={FEATURE_MAX[feature.key] || 1}
                        index={idx}
                        inView={true}
                      />
                    ))}
                  </div>

                  {/* AI Risk Explanation */}
                  <div className="p-4 rounded-2xl bg-white/5 border border-white/10 space-y-3 mb-6">
                    <div className="flex items-center gap-2 text-xs font-bold text-purple-400">
                      <Sparkles className="w-4 h-4" />
                      {result.explanation.is_fallback
                        ? 'Rule-Based Explanation'
                        : 'AI Explanation (Llama 3 via Groq)'}
                    </div>
                    <div className="text-sm text-gray-300 leading-relaxed font-semibold">
                      {result.explanation.recommendation}
                    </div>
                    <ul className="space-y-1.5 pt-2 border-t border-white/5">
                      {result.explanation.reasons.map((reason, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-xs text-gray-400">
                          <span className="text-purple-400 font-bold mt-0.5">•</span>
                          {reason}
                        </li>
                      ))}
                    </ul>
                  </div>

                  {/* Ensemble fraud probability */}
                  <div className="pt-4 border-t border-white/5 mb-4">
                    <div className="flex justify-between items-center">
                      <div>
                        <div className="text-xs text-gray-500 mb-1 flex items-center gap-1.5">
                          Ensemble Fraud Probability
                          <span className="text-[10px] px-1.5 py-0.5 rounded bg-cyan-500/10 text-cyan-400 font-mono">
                            {result.prediction.ensemble_mode}
                          </span>
                        </div>
                        <div
                          className={`text-3xl font-black font-[Manrope] ${
                            result.prediction.is_fraud ? 'text-red-400' : 'text-green-400'
                          }`}
                        >
                          {(result.prediction.fraud_probability * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div
                        className={`rounded-xl px-4 py-2 text-center border ${
                          result.prediction.is_fraud
                            ? 'bg-red-500/10 border-red-500/20'
                            : 'bg-green-500/10 border-green-500/20'
                        }`}
                      >
                        <div className="text-xs text-gray-500 mb-1">Ensemble Verdict</div>
                        <div
                          className={`text-sm font-black flex items-center gap-1.5 ${
                            result.prediction.is_fraud ? 'text-red-400' : 'text-green-400'
                          }`}
                        >
                          {result.prediction.is_fraud ? (
                            <>
                              <ShieldAlert className="w-4 h-4" /> BLOCKED
                            </>
                          ) : (
                            <>
                              <ShieldCheck className="w-4 h-4" /> SAFE
                            </>
                          )}
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Dual-model breakdown */}
                  <div className="pt-3 border-t border-white/5 space-y-2.5">
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-bold text-gray-300 flex items-center gap-1.5 font-[Manrope]">
                        <Layers className="w-3.5 h-3.5 text-cyan-400" />
                        Dual-Model Classification
                      </span>
                      <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/20">
                        Max-Recall Ensemble
                      </span>
                    </div>

                    <div className="grid grid-cols-2 gap-2.5">
                      {/* Random Forest */}
                      <div
                        className={`p-3 rounded-xl border transition-all ${
                          result.prediction.primary_model === 'random_forest'
                            ? 'bg-cyan-500/10 border-cyan-500/30 shadow-[0_0_12px_rgba(6,182,212,0.12)]'
                            : 'bg-white/5 border-white/10 opacity-75'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1.5">
                          <span className="text-[11px] font-semibold text-gray-300 flex items-center gap-1">
                            <Cpu className="w-3 h-3 text-cyan-400" />
                            Random Forest
                          </span>
                          {result.prediction.primary_model === 'random_forest' && (
                            <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-cyan-400/20 text-cyan-300 uppercase">
                              Primary
                            </span>
                          )}
                        </div>
                        <div className="text-lg font-bold font-[Manrope] text-white">
                          {(result.prediction.rf_score * 100).toFixed(1)}%
                        </div>
                        <div className="text-[10px] text-gray-400 mt-0.5">
                          7-Feature Tabular Model
                        </div>
                      </div>

                      {/* DeFiGuard */}
                      <div
                        className={`p-3 rounded-xl border transition-all ${
                          result.prediction.primary_model === 'defiguard'
                            ? 'bg-purple-500/10 border-purple-500/30 shadow-[0_0_12px_rgba(168,85,247,0.12)]'
                            : 'bg-white/5 border-white/10 opacity-75'
                        }`}
                      >
                        <div className="flex items-center justify-between mb-1.5">
                          <span className="text-[11px] font-semibold text-gray-300 flex items-center gap-1">
                            <Zap className="w-3 h-3 text-purple-400" />
                            DeFiGuard Net
                          </span>
                          {result.prediction.primary_model === 'defiguard' && (
                            <span className="text-[9px] font-bold px-1.5 py-0.5 rounded bg-purple-400/20 text-purple-300 uppercase">
                              Primary
                            </span>
                          )}
                        </div>
                        <div className="text-lg font-bold font-[Manrope] text-white">
                          {result.prediction.defiguard_score != null
                            ? `${(result.prediction.defiguard_score * 100).toFixed(1)}%`
                            : 'Standby'}
                        </div>
                        <div className="text-[10px] text-gray-400 mt-0.5">
                          GNN + Tabular Neural Net
                        </div>
                      </div>
                    </div>
                  </div>
                </motion.div>
              ) : (
                <motion.div
                  key="placeholder"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  className="flex flex-col items-center justify-center py-16 text-center"
                >
                  {loading || isScanning ? (
                    <>
                      <div className="w-16 h-16 rounded-2xl bg-purple-500/10 border border-purple-500/30 flex items-center justify-center mb-6 animate-pulse">
                        <RefreshCw className="w-8 h-8 text-purple-400 animate-spin" />
                      </div>
                      <h3 className="text-lg font-bold text-white font-[Manrope] mb-2">
                        Running Ensemble Inference…
                      </h3>
                      <p className="text-sm text-gray-500 max-w-xs">
                        Querying Random Forest + DeFiGuard ensemble. Usually under 100ms.
                      </p>
                    </>
                  ) : (
                    <>
                      <div className="w-16 h-16 rounded-2xl bg-purple-500/10 border border-purple-500/20 flex items-center justify-center mb-6">
                        <Zap className="w-8 h-8 text-purple-400" />
                      </div>
                      <h3 className="text-lg font-bold text-white font-[Manrope] mb-2">
                        Ready to Scan
                      </h3>
                      <p className="text-sm text-gray-500 max-w-xs">
                        Pick a preset scenario above or enter custom transaction parameters.
                        Every scan routes through the live inference API.
                      </p>
                    </>
                  )}
                </motion.div>
              )}
            </AnimatePresence>
          </motion.div>
        </div>
      </div>
    </section>
  );
}
