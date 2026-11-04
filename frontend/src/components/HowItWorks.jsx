import { useRef } from 'react';
import { motion, useInView } from 'framer-motion';
import { Pen, Wifi, Cpu, Brain, BarChart3, CheckCircle } from 'lucide-react';
import { modelMetrics, pct } from '../data/modelMetrics';

const steps = [
  {
    icon: Pen,
    title: 'User Sends a Transaction',
    description: 'The user signs and sends as usual. With the DeFiGuard RPC selected, the wallet sends it to the firewall instead of straight to the network.',
    color: 'from-blue-500 to-cyan-500',
    step: '01',
  },
  {
    icon: Wifi,
    title: 'RPC Interception',
    description: 'The firewall decodes the signed transaction and recovers the sender from its signature. Every other RPC call is forwarded unchanged.',
    color: 'from-cyan-500 to-teal-500',
    step: '02',
  },
  {
    icon: Cpu,
    title: 'Counterparty Checks',
    description: 'Who does it pay or approve? Recipients, approved spenders and token recipients are checked against fraud-labelled wallets, and approvals to plain wallets are caught.',
    color: 'from-teal-500 to-green-500',
    step: '03',
  },
  {
    icon: Brain,
    title: 'DeFiGuard Score',
    description: 'The pre-execution model scores how closely the transaction resembles ones sent by fraud-labelled wallets, from transaction-local features only (no address history).',
    color: 'from-purple-500 to-violet-500',
    step: '04',
  },
  {
    icon: BarChart3,
    title: 'Verdict',
    description: 'A counterparty red flag blocks. A high model score or a softer counterparty finding warns. Everything else is allowed.',
    color: 'from-orange-500 to-amber-500',
    step: '05',
  },
  {
    icon: CheckCircle,
    title: 'Forward or Refuse',
    description: 'Allowed and warned transactions go to the network. Blocked ones are never broadcast: the wallet shows the reason, and re-sending within two minutes overrides.',
    color: 'from-green-500 to-emerald-500',
    step: '06',
  },
];

function StepCard({ step, index, total }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: '-80px' });
  const Icon = step.icon;

  return (
    <div ref={ref} className="relative flex flex-col items-center">
      {/* Connector line */}
      {index < total - 1 && (
        <motion.div
          initial={{ scaleX: 0 }}
          animate={inView ? { scaleX: 1 } : {}}
          transition={{ delay: 0.5, duration: 0.6 }}
          className="absolute top-10 left-1/2 w-full h-0.5 hidden lg:block origin-left"
          style={{ background: 'linear-gradient(90deg, #06B6D4, #3B82F6, transparent)', zIndex: 0 }}
        />
      )}

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={inView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 0.6, delay: index * 0.12 }}
        className="flex flex-col items-center relative z-10"
      >
        {/* Step number */}
        <div className="text-xs font-bold text-gray-600 mb-2 font-mono">{step.step}</div>

        {/* Icon circle */}
        <motion.div
          whileHover={{ scale: 1.15, rotate: 5 }}
          className={`w-20 h-20 rounded-2xl bg-gradient-to-br ${step.color} flex items-center justify-center shadow-2xl mb-4 relative`}
        >
          <Icon className="w-9 h-9 text-white" />
          <div className="absolute -inset-1 rounded-2xl opacity-30 blur-md"
            style={{ background: `linear-gradient(135deg, #06B6D4, #3B82F6)` }} />
        </motion.div>

        {/* Content */}
        <div className="text-center max-w-[150px]">
          <h3 className="font-bold text-white text-sm mb-2 font-[Manrope]">{step.title}</h3>
          <p className="text-xs text-gray-500 leading-relaxed">{step.description}</p>
        </div>

        {/* Down arrow for mobile */}
        {index < total - 1 && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={inView ? { opacity: 1 } : {}}
            transition={{ delay: 0.5 }}
            className="sm:hidden mt-4 text-cyan-500 text-xl"
          >
            ↓
          </motion.div>
        )}
      </motion.div>
    </div>
  );
}

export default function HowItWorks() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });

  return (
    <section id="how-it-works" className="py-28 relative overflow-hidden">
      {/* Background */}
      <div className="absolute inset-0"
        style={{ background: 'radial-gradient(ellipse 60% 50% at 50% 50%, rgba(6,182,212,0.05) 0%, transparent 70%)' }}
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        {/* Header */}
        <div ref={ref} className="text-center mb-20">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass border border-cyan-500/20 text-sm text-cyan-400 font-medium mb-6"
          >
            <Cpu className="w-4 h-4" />
            Under the Hood
          </motion.div>

          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 0.1 }}
            className="text-4xl lg:text-5xl font-black text-white mb-4 font-[Manrope]"
          >
            How <span className="text-gradient">DHARMA</span> Works
          </motion.h2>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 0.2 }}
            className="text-gray-400 text-lg max-w-2xl mx-auto"
          >
            Six steps between your wallet and the network — the model itself scores a transaction in{' '}
            {modelMetrics.latencyMedianMs.toFixed(1)} ms (median).
          </motion.p>
        </div>

        {/* Steps Timeline */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-6 gap-8 lg:gap-4 relative">
          {steps.map((step, idx) => (
            <StepCard key={step.step} step={step} index={idx} total={steps.length} />
          ))}
        </div>

        {/* Bottom info strip */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ delay: 0.8 }}
          className="mt-20 glass-card rounded-2xl p-6 border border-cyan-500/10 flex flex-wrap items-center justify-center gap-8 text-center"
        >
          {[
            { label: 'Median Inference Time', value: `${modelMetrics.latencyMedianMs.toFixed(1)} ms` },
            { label: 'Features Analyzed', value: `${modelMetrics.features} dimensions` },
            { label: 'Model Accuracy', value: pct(modelMetrics.accuracy) },
            { label: 'False Positive Rate', value: pct(modelMetrics.fpr) },
          ].map((item) => (
            <div key={item.label}>
              <div className="text-2xl font-black text-gradient-cyan font-[Manrope]">{item.value}</div>
              <div className="text-sm text-gray-500">{item.label}</div>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  );
}
