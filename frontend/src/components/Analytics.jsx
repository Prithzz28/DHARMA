import { useRef, useState, useEffect } from 'react';
import { motion, useInView } from 'framer-motion';
import { BarChart2, TrendingUp, ShieldAlert, Activity, Timer } from 'lucide-react';
import {
  LineChart, Line, BarChart, Bar, PieChart, Pie, Cell, ReferenceLine, Legend,
  ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid,
} from 'recharts';
import { fetchStats } from '../services/api';
import { modelMetrics, evaluationSource, pct } from '../data/modelMetrics';

// ─── Chart data, all derived from the measured evaluation ─────────────────────
const thresholdData = modelMetrics.thresholdCurve.map((m) => ({
  threshold: m.threshold.toFixed(2),
  Precision: +(m.precision * 100).toFixed(1),
  Recall: +(m.recall * 100).toFixed(1),
  'False-positive rate': +(m.fpr * 100).toFixed(1),
}));

const histogramData = modelMetrics.scoreHistogram.map((b) => ({
  range: b.range,
  Fraud: b.fraud,
  Legitimate: b.legit,
}));

const outcomeData = [
  { name: 'Fraud blocked', value: modelMetrics.outcomes.fraud_caught, color: '#22C55E' },
  { name: 'Fraud missed', value: modelMetrics.outcomes.fraud_missed, color: '#EF4444' },
  { name: 'Legit blocked', value: modelMetrics.outcomes.legit_flagged, color: '#F59E0B' },
  { name: 'Legit passed', value: modelMetrics.outcomes.legit_passed, color: '#06B6D4' },
];

const datasetData = [
  { name: 'Legitimate', rows: modelMetrics.datasetRows.legit },
  { name: 'Fraud', rows: modelMetrics.datasetRows.fraud },
];

const STATS_POLL_MS = 15_000;

const CustomTooltip = ({ active, payload, label, unit = '' }) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card border border-white/10 rounded-xl px-4 py-3">
        <p className="text-xs text-gray-400 mb-2">{label}</p>
        {payload.map((entry) => (
          <p key={entry.name} className="text-sm font-bold" style={{ color: entry.color || entry.payload?.color }}>
            {entry.name}: {Number(entry.value).toLocaleString()}{unit}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

function ChartCard({ title, subtitle, delay, inView, children }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 30 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ delay }}
      className="glass-card rounded-2xl p-6 border border-white/5"
    >
      <h3 className="font-bold text-white mb-1 font-[Manrope]">{title}</h3>
      <p className="text-xs text-gray-500 mb-6">{subtitle}</p>
      {children}
    </motion.div>
  );
}

/** Live counters from GET /stats; null while the engine is unreachable. */
function useLiveStats() {
  const [stats, setStats] = useState(null);
  useEffect(() => {
    let controller;
    const load = async () => {
      controller?.abort();
      controller = new AbortController();
      try {
        setStats(await fetchStats({ signal: controller.signal }));
      } catch (err) {
        if (err?.name !== 'AbortError') setStats(null);
      }
    };
    load();
    const timer = setInterval(load, STATS_POLL_MS);
    return () => {
      clearInterval(timer);
      controller?.abort();
    };
  }, []);
  return stats;
}

export default function Analytics() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });
  const stats = useLiveStats();

  const live = (value) => (stats ? value : '—');
  const metrics = [
    { label: 'Transactions Scanned', value: live(stats?.total_predictions.toLocaleString()), icon: Activity, color: 'text-cyan-400', bg: 'bg-cyan-500/10' },
    { label: 'Blocked as Fraud', value: live(stats?.total_fraud.toLocaleString()), icon: ShieldAlert, color: 'text-red-400', bg: 'bg-red-500/10' },
    { label: 'Block Rate', value: live(`${stats?.fraud_rate_pct.toFixed(1)}%`), icon: TrendingUp, color: 'text-green-400', bg: 'bg-green-500/10' },
    { label: 'Avg Inference Time', value: live(`${stats?.avg_latency_ms.toFixed(1)} ms`), icon: Timer, color: 'text-orange-400', bg: 'bg-orange-500/10' },
  ];

  return (
    <section id="analytics" className="py-28 relative">
      <div className="absolute inset-0 grid-bg opacity-15" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Header */}
        <div ref={ref} className="text-center mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass border border-cyan-500/20 text-sm text-cyan-400 font-medium mb-6"
          >
            <BarChart2 className="w-4 h-4" />
            Analytics & Insights
          </motion.div>

          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 0.1 }}
            className="text-4xl lg:text-5xl font-black text-white mb-4 font-[Manrope]"
          >
            Data-Driven <span className="text-gradient">Security Intelligence</span>
          </motion.h2>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 0.2 }}
            className="text-gray-400 text-lg max-w-2xl mx-auto mb-3"
          >
            Live activity from this inference engine, and how the model performs on real labelled transactions.
          </motion.p>
        </div>

        {/* Live counters (GET /stats) */}
        <div className="flex items-center justify-between mb-3 text-xs text-gray-500">
          <span className="flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full ${stats ? 'bg-green-400 animate-pulse' : 'bg-gray-600'}`} />
            {stats
              ? `Live from this engine · up ${Math.floor(stats.uptime_seconds / 60)} min`
              : 'Inference engine unreachable — live counters unavailable'}
          </span>
        </div>
        <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 mb-8">
          {metrics.map((metric, idx) => {
            const Icon = metric.icon;
            return (
              <motion.div
                key={metric.label}
                initial={{ opacity: 0, y: 20 }}
                animate={inView ? { opacity: 1, y: 0 } : {}}
                transition={{ delay: 0.3 + idx * 0.1 }}
                whileHover={{ y: -4 }}
                className="glass-card rounded-2xl p-6 border border-white/5 hover:border-white/10 transition-all duration-300"
              >
                <div className={`w-10 h-10 rounded-xl ${metric.bg} flex items-center justify-center mb-4`}>
                  <Icon className={`w-5 h-5 ${metric.color}`} />
                </div>
                <div className={`text-3xl font-black ${metric.color} font-[Manrope] mb-1`}>{metric.value}</div>
                <div className="text-sm text-gray-500">{metric.label}</div>
              </motion.div>
            );
          })}
        </div>

        {/* Charts Grid — measured evaluation */}
        <div className="grid lg:grid-cols-2 gap-6">
          <ChartCard
            title="Precision, Recall & False Positives"
            subtitle="By block threshold — dashed line marks the configured threshold"
            delay={0.5}
            inView={inView}
          >
            <ResponsiveContainer width="100%" height={220}>
              <LineChart data={thresholdData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                <XAxis dataKey="threshold" tick={{ fontSize: 11, fill: '#6B7280' }} axisLine={false} tickLine={false} />
                <YAxis domain={[0, 100]} unit="%" tick={{ fontSize: 11, fill: '#6B7280' }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip unit="%" />} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <ReferenceLine x={modelMetrics.threshold.toFixed(2)} stroke="#9CA3AF" strokeDasharray="4 4" />
                <Line type="monotone" dataKey="Precision" stroke="#22C55E" strokeWidth={2} dot />
                <Line type="monotone" dataKey="Recall" stroke="#8B5CF6" strokeWidth={2} dot />
                <Line type="monotone" dataKey="False-positive rate" stroke="#EF4444" strokeWidth={2} dot />
              </LineChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard
            title="Fraud Score Distribution"
            subtitle="Evaluation transactions per DeFiGuard score range, by true label"
            delay={0.6}
            inView={inView}
          >
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={histogramData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                <XAxis dataKey="range" tick={{ fontSize: 9, fill: '#6B7280' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: '#6B7280' }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="Legitimate" stackId="label" fill="#06B6D4" fillOpacity={0.8} />
                <Bar dataKey="Fraud" stackId="label" fill="#EF4444" fillOpacity={0.8} radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>

          <ChartCard
            title="Outcomes at the Block Threshold"
            subtitle={`${modelMetrics.sampleSize.toLocaleString()} evaluation transactions at threshold ${modelMetrics.threshold.toFixed(2)}`}
            delay={0.7}
            inView={inView}
          >
            <div className="flex items-center justify-between">
              <ResponsiveContainer width="60%" height={200}>
                <PieChart>
                  <Pie data={outcomeData} cx="50%" cy="50%" innerRadius={55} outerRadius={85} strokeWidth={0} dataKey="value">
                    {outcomeData.map((entry) => (
                      <Cell key={entry.name} fill={entry.color} fillOpacity={0.85} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex flex-col gap-3">
                {outcomeData.map((item) => (
                  <div key={item.name} className="flex items-center gap-2">
                    <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }} />
                    <div>
                      <div className="text-xs font-bold text-white">{item.name}</div>
                      <div className="text-xs text-gray-500">{item.value.toLocaleString()}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </ChartCard>

          <ChartCard
            title="Training Dataset Composition"
            subtitle="Labelled transactions in the leakage-controlled DeFiGuard dataset"
            delay={0.8}
            inView={inView}
          >
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={datasetData} layout="vertical" margin={{ left: 20 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                <XAxis type="number" tick={{ fontSize: 11, fill: '#6B7280' }} axisLine={false} tickLine={false}
                  tickFormatter={(v) => `${(v / 1000).toLocaleString()}k`} />
                <YAxis type="category" dataKey="name" tick={{ fontSize: 11, fill: '#6B7280' }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="rows" name="Transactions" radius={[0, 6, 6, 0]}>
                  {datasetData.map((entry) => (
                    <Cell key={entry.name} fill={entry.name === 'Fraud' ? '#EF4444' : '#22C55E'} fillOpacity={0.8} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </ChartCard>
        </div>

        {/* Measured benchmark */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 0.85 }}
          className="mt-8 glass-card rounded-2xl p-6 lg:p-8 border border-cyan-500/20"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="w-2.5 h-2.5 rounded-full bg-cyan-400" />
                <h3 className="text-lg font-bold text-white font-[Manrope]">DeFiGuard Engine Benchmark Metrics</h3>
              </div>
              <p className="text-xs text-gray-400">{evaluationSource}</p>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                Threshold: {modelMetrics.threshold.toFixed(2)}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl bg-white/5 border border-white/5">
              <div className="text-xs text-gray-400 mb-1">Accuracy</div>
              <div className="text-2xl font-black text-cyan-400 font-mono">{pct(modelMetrics.accuracy)}</div>
              <div className="text-[10px] text-gray-500 mt-1">Balanced evaluation sample</div>
            </div>
            <div className="p-4 rounded-xl bg-white/5 border border-white/5">
              <div className="text-xs text-gray-400 mb-1">Precision</div>
              <div className="text-2xl font-black text-emerald-400 font-mono">{pct(modelMetrics.precision)}</div>
              <div className="text-[10px] text-gray-500 mt-1">FPR {pct(modelMetrics.fpr)} of legitimate txns blocked</div>
            </div>
            <div className="p-4 rounded-xl bg-white/5 border border-white/5">
              <div className="text-xs text-gray-400 mb-1">Recall</div>
              <div className="text-2xl font-black text-purple-400 font-mono">{pct(modelMetrics.recall)}</div>
              <div className="text-[10px] text-gray-500 mt-1">{pct(1 - modelMetrics.recall)} of fraud missed</div>
            </div>
            <div className="p-4 rounded-xl bg-white/5 border border-white/5">
              <div className="text-xs text-gray-400 mb-1">Median Latency (p50)</div>
              <div className="text-2xl font-black text-amber-400 font-mono">{modelMetrics.latencyMedianMs.toFixed(1)} ms</div>
              <div className="text-[10px] text-gray-500 mt-1">Per transaction on CPU · p95 {modelMetrics.latencyP95Ms.toFixed(1)} ms</div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
