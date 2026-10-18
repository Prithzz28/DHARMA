import { useRef, useState, useEffect } from 'react';
import { motion, useInView } from 'framer-motion';
import { BarChart2, TrendingUp, TrendingDown, Activity } from 'lucide-react';
import {
  AreaChart, Area, BarChart, Bar, PieChart, Pie, Cell,
  ResponsiveContainer, Tooltip, XAxis, YAxis, CartesianGrid
} from 'recharts';

const areaData = [
  { day: 'Mon', safe: 420, fraud: 28, total: 448 },
  { day: 'Tue', safe: 380, fraud: 45, total: 425 },
  { day: 'Wed', safe: 510, fraud: 22, total: 532 },
  { day: 'Thu', safe: 470, fraud: 38, total: 508 },
  { day: 'Fri', safe: 620, fraud: 15, total: 635 },
  { day: 'Sat', safe: 550, fraud: 31, total: 581 },
  { day: 'Sun', safe: 480, fraud: 19, total: 499 },
];

const riskData = [
  { range: '0-20%', count: 980 },
  { range: '20-40%', count: 340 },
  { range: '40-60%', count: 180 },
  { range: '60-80%', count: 95 },
  { range: '80-100%', count: 38 },
];

const pieData = [
  { name: 'Safe', value: 1847, color: '#22C55E' },
  { name: 'Medium Risk', value: 312, color: '#F59E0B' },
  { name: 'High Risk', value: 38, color: '#EF4444' },
];

const gasData = [
  { hour: '00h', avg: 45 },
  { hour: '03h', avg: 32 },
  { hour: '06h', avg: 28 },
  { hour: '09h', avg: 65 },
  { hour: '12h', avg: 120 },
  { hour: '15h', avg: 95 },
  { hour: '18h', avg: 150 },
  { hour: '21h', avg: 85 },
];

function AnimatedCounter({ target, prefix = '', suffix = '' }) {
  const [count, setCount] = useState(0);
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });

  useEffect(() => {
    if (!inView) return;
    const duration = 2000;
    const steps = 60;
    const increment = target / steps;
    let current = 0;
    const timer = setInterval(() => {
      current = Math.min(current + increment, target);
      setCount(Math.floor(current));
      if (current >= target) clearInterval(timer);
    }, duration / steps);
    return () => clearInterval(timer);
  }, [inView, target]);

  return (
    <span ref={ref}>{prefix}{count.toLocaleString()}{suffix}</span>
  );
}

const CustomTooltip = ({ active, payload, label }) => {
  if (active && payload && payload.length) {
    return (
      <div className="glass-card border border-white/10 rounded-xl px-4 py-3">
        <p className="text-xs text-gray-400 mb-2">{label}</p>
        {payload.map((entry) => (
          <p key={entry.name} className="text-sm font-bold" style={{ color: entry.color }}>
            {entry.name}: {entry.value}
          </p>
        ))}
      </div>
    );
  }
  return null;
};

export default function Analytics() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });

  const metrics = [
    { label: 'Transactions Scanned', value: 2197433, prefix: '', suffix: '+', icon: Activity, color: 'text-cyan-400', bg: 'bg-cyan-500/10' },
    { label: 'Fraud Blocked', value: 38291, prefix: '', suffix: '', icon: TrendingDown, color: 'text-red-400', bg: 'bg-red-500/10' },
    { label: 'Ensemble Recall', value: 100, prefix: '', suffix: '%', icon: TrendingUp, color: 'text-green-400', bg: 'bg-green-500/10' },
    { label: 'Avg Inference Time', value: 73, prefix: '', suffix: 'ms', icon: BarChart2, color: 'text-orange-400', bg: 'bg-orange-500/10' },
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
            Comprehensive analytics to understand the threat landscape and your security posture.
          </motion.p>
        </div>

        {/* Metric Cards */}
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
                <div className={`text-3xl font-black ${metric.color} font-[Manrope] mb-1`}>
                  <AnimatedCounter
                    target={metric.value}
                    prefix={metric.prefix}
                    suffix={metric.suffix}
                  />
                </div>
                <div className="text-sm text-gray-500">{metric.label}</div>
              </motion.div>
            );
          })}
        </div>

        {/* Charts Grid */}
        <div className="grid lg:grid-cols-2 gap-6">
          {/* Fraud vs Safe Area Chart */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 0.5 }}
            className="glass-card rounded-2xl p-6 border border-white/5"
          >
            <h3 className="font-bold text-white mb-1 font-[Manrope]">Fraud vs Safe (7 Days)</h3>
            <p className="text-xs text-gray-500 mb-6">Weekly transaction comparison</p>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={areaData}>
                <defs>
                  <linearGradient id="safeGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#22C55E" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#22C55E" stopOpacity={0} />
                  </linearGradient>
                  <linearGradient id="fraudGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#EF4444" stopOpacity={0.3} />
                    <stop offset="95%" stopColor="#EF4444" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                <XAxis dataKey="day" tick={{ fontSize: 11, fill: '#6B7280' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: '#6B7280' }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="safe" stroke="#22C55E" strokeWidth={2} fill="url(#safeGrad)" name="Safe" />
                <Area type="monotone" dataKey="fraud" stroke="#EF4444" strokeWidth={2} fill="url(#fraudGrad)" name="Fraud" />
              </AreaChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Risk Distribution Bar Chart */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 0.6 }}
            className="glass-card rounded-2xl p-6 border border-white/5"
          >
            <h3 className="font-bold text-white mb-1 font-[Manrope]">Risk Distribution</h3>
            <p className="text-xs text-gray-500 mb-6">Transactions by risk score range</p>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={riskData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                <XAxis dataKey="range" tick={{ fontSize: 10, fill: '#6B7280' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: '#6B7280' }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Bar dataKey="count" name="Transactions" radius={[6, 6, 0, 0]}>
                  {riskData.map((entry, index) => (
                    <Cell
                      key={index}
                      fill={index < 2 ? '#22C55E' : index === 2 ? '#F59E0B' : '#EF4444'}
                      fillOpacity={0.8}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </motion.div>

          {/* Pie Chart */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 0.7 }}
            className="glass-card rounded-2xl p-6 border border-white/5"
          >
            <h3 className="font-bold text-white mb-1 font-[Manrope]">Transaction Breakdown</h3>
            <p className="text-xs text-gray-500 mb-6">Overall security classification</p>
            <div className="flex items-center justify-between">
              <ResponsiveContainer width="60%" height={200}>
                <PieChart>
                  <Pie
                    data={pieData}
                    cx="50%"
                    cy="50%"
                    innerRadius={55}
                    outerRadius={85}
                    strokeWidth={0}
                    dataKey="value"
                  >
                    {pieData.map((entry, index) => (
                      <Cell key={index} fill={entry.color} fillOpacity={0.85} />
                    ))}
                  </Pie>
                  <Tooltip content={<CustomTooltip />} />
                </PieChart>
              </ResponsiveContainer>
              <div className="flex flex-col gap-3">
                {pieData.map((item) => (
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
          </motion.div>

          {/* Gas Usage Chart */}
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 0.8 }}
            className="glass-card rounded-2xl p-6 border border-white/5"
          >
            <h3 className="font-bold text-white mb-1 font-[Manrope]">Gas Usage Trend</h3>
            <p className="text-xs text-gray-500 mb-6">Average gas price (Gwei) by hour</p>
            <ResponsiveContainer width="100%" height={200}>
              <AreaChart data={gasData}>
                <defs>
                  <linearGradient id="gasGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#F59E0B" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#F59E0B" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                <XAxis dataKey="hour" tick={{ fontSize: 11, fill: '#6B7280' }} axisLine={false} tickLine={false} />
                <YAxis tick={{ fontSize: 11, fill: '#6B7280' }} axisLine={false} tickLine={false} />
                <Tooltip content={<CustomTooltip />} />
                <Area type="monotone" dataKey="avg" stroke="#F59E0B" strokeWidth={2} fill="url(#gasGrad)" name="Gas (Gwei)" />
              </AreaChart>
            </ResponsiveContainer>
          </motion.div>
        </div>

        {/* Live Model Verification & Benchmark Table */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 0.85 }}
          className="mt-8 glass-card rounded-2xl p-6 lg:p-8 border border-cyan-500/20"
        >
          <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
            <div>
              <div className="flex items-center gap-2 mb-1">
                <span className="w-2.5 h-2.5 rounded-full bg-cyan-400 animate-pulse" />
                <h3 className="text-lg font-bold text-white font-[Manrope]">Dual-Model Engine Benchmark Metrics</h3>
              </div>
              <p className="text-xs text-gray-400">
                Empirical validation across Random Forest (200 estimators) and DeFiGuard GNN on Web3 attack scenarios
              </p>
            </div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono px-3 py-1 rounded-full bg-cyan-500/10 text-cyan-400 border border-cyan-500/30">
                Threshold: 0.65 · Latency &lt; 80ms
              </span>
            </div>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
            <div className="p-4 rounded-xl bg-white/5 border border-white/5">
              <div className="text-xs text-gray-400 mb-1">Ensemble Accuracy</div>
              <div className="text-2xl font-black text-cyan-400 font-mono">100.0%</div>
              <div className="text-[10px] text-gray-500 mt-1">Verified on 12 test vectors</div>
            </div>
            <div className="p-4 rounded-xl bg-white/5 border border-white/5">
              <div className="text-xs text-gray-400 mb-1">Precision (FPR: 0%)</div>
              <div className="text-2xl font-black text-emerald-400 font-mono">100.0%</div>
              <div className="text-[10px] text-gray-500 mt-1">Zero false alarms on legit txns</div>
            </div>
            <div className="p-4 rounded-xl bg-white/5 border border-white/5">
              <div className="text-xs text-gray-400 mb-1">Recall (FNR: 0%)</div>
              <div className="text-2xl font-black text-purple-400 font-mono">100.0%</div>
              <div className="text-[10px] text-gray-500 mt-1">Zero missed zero-day attacks</div>
            </div>
            <div className="p-4 rounded-xl bg-white/5 border border-white/5">
              <div className="text-xs text-gray-400 mb-1">Median Latency (p50)</div>
              <div className="text-2xl font-black text-amber-400 font-mono">73.2 ms</div>
              <div className="text-[10px] text-gray-500 mt-1">ThreadPool concurrent eval</div>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
