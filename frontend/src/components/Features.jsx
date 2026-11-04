import { motion } from 'framer-motion';
import { useInView } from 'framer-motion';
import { useRef } from 'react';
import {
  Scan, Brain, Shield, Gauge, MessageSquare, Zap
} from 'lucide-react';
import { modelMetrics } from '../data/modelMetrics';

const features = [
  {
    icon: Scan,
    title: 'Wallet RPC Firewall',
    description: 'Add one RPC URL to your wallet. Every transaction is checked before it is broadcast; balance, gas and block lookups pass straight through.',
    color: 'from-cyan-500 to-blue-600',
    glow: 'rgba(6,182,212,0.2)',
    badge: 'Core',
  },
  {
    icon: Brain,
    title: 'DeFiGuard Sender-Pattern Model',
    description: 'Scores how closely a transaction resembles ones sent by wallets labelled fraudulent, measured on wallets it never saw in training. It warns; it does not block on its own.',
    color: 'from-purple-500 to-violet-600',
    glow: 'rgba(139,92,246,0.2)',
    badge: 'ML',
  },
  {
    icon: Shield,
    title: 'Counterparty Checks',
    description: 'Blocks payments and approvals to 1,763 wallets labelled fraudulent, and unlimited or approve-all grants to plain wallets — a common drainer pattern.',
    color: 'from-blue-500 to-indigo-600',
    glow: 'rgba(59,130,246,0.2)',
    badge: 'Security',
  },
  {
    icon: Gauge,
    title: 'Leakage-Controlled Evaluation',
    description: 'Address-disjoint, time-matched test splits, and no wallet identity, history or block height in the features — so the reported numbers reflect transaction signal, not memorised wallets.',
    color: 'from-orange-500 to-amber-600',
    glow: 'rgba(245,158,11,0.2)',
    badge: 'Research',
  },
  {
    icon: MessageSquare,
    title: 'Explainable Security',
    description: 'Every verdict says why: the counterparty rule that fired, or the transaction signals behind the model score.',
    color: 'from-green-500 to-emerald-600',
    glow: 'rgba(34,197,94,0.2)',
    badge: 'XAI',
  },
  {
    icon: Zap,
    title: 'Fast Enough to Sit Inline',
    description: `Pre-execution scoring takes ${modelMetrics.latencyMedianMs.toFixed(1)} ms median (${modelMetrics.latencyP95Ms.toFixed(1)} ms p95) on CPU — the round trip to the Ethereum node dominates. Optional LLM explanations take longer.`,
    color: 'from-yellow-500 to-orange-500',
    glow: 'rgba(234,179,8,0.2)',
    badge: 'Speed',
  },
];

function FeatureCard({ feature, index }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-100px' });
  const Icon = feature.icon;

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 40 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ duration: 0.6, delay: index * 0.1 }}
      whileHover={{ y: -8, scale: 1.02 }}
      className="group relative glass-card rounded-2xl p-6 border border-white/5 hover:border-white/10 transition-all duration-500 cursor-default overflow-hidden"
      style={{ '--glow': feature.glow }}
    >
      {/* Hover glow background */}
      <div
        className="absolute inset-0 opacity-0 group-hover:opacity-100 transition-opacity duration-500 rounded-2xl"
        style={{ background: `radial-gradient(circle at 30% 30%, ${feature.glow}, transparent 70%)` }}
      />

      {/* Badge */}
      <div className="absolute top-4 right-4">
        <span className={`text-xs font-bold px-2.5 py-1 rounded-full bg-gradient-to-r ${feature.color} text-white opacity-80`}>
          {feature.badge}
        </span>
      </div>

      {/* Icon */}
      <div className={`relative w-14 h-14 rounded-2xl bg-gradient-to-br ${feature.color} flex items-center justify-center mb-5 shadow-lg group-hover:scale-110 transition-transform duration-300`}>
        <Icon className="w-7 h-7 text-white" />
        <div className="absolute inset-0 rounded-2xl opacity-50 blur-md"
          style={{ background: `linear-gradient(135deg, ${feature.glow}, transparent)` }} />
      </div>

      {/* Content */}
      <h3 className="text-lg font-bold text-white mb-3 font-[Manrope] relative z-10">{feature.title}</h3>
      <p className="text-sm text-gray-400 leading-relaxed relative z-10">{feature.description}</p>

      {/* Bottom glow line */}
      <div className={`absolute bottom-0 left-0 right-0 h-0.5 bg-gradient-to-r ${feature.color} opacity-0 group-hover:opacity-60 transition-opacity duration-500`} />
    </motion.div>
  );
}

export default function Features() {
  const titleRef = useRef(null);
  const titleInView = useInView(titleRef, { once: true });

  return (
    <section id="features" className="py-28 relative">
      <div className="absolute inset-0 grid-bg opacity-20" />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* Section Header */}
        <div ref={titleRef} className="text-center mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={titleInView ? { opacity: 1, y: 0 } : {}}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass border border-cyan-500/20 text-sm text-cyan-400 font-medium mb-6"
          >
            <Shield className="w-4 h-4" />
            What It Does
          </motion.div>

          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            animate={titleInView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 0.1 }}
            className="text-4xl lg:text-5xl font-black text-white mb-4 font-[Manrope]"
          >
            Two Checks on Every{' '}
            <span className="text-gradient">Transaction</span>
          </motion.h2>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={titleInView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 0.2 }}
            className="text-gray-400 text-lg max-w-2xl mx-auto"
          >
            Rules about who you pay or approve, a model of how fraud-labelled wallets transact, and the evidence behind every verdict.
          </motion.p>
        </div>

        {/* Feature Grid */}
        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map((feature, idx) => (
            <FeatureCard key={feature.title} feature={feature} index={idx} />
          ))}
        </div>
      </div>
    </section>
  );
}
