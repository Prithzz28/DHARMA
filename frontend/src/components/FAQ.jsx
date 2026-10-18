import { useState, useRef } from 'react';
import { motion, useInView, AnimatePresence } from 'framer-motion';
import { HelpCircle, ChevronDown } from 'lucide-react';

const faqs = [
  {
    q: 'How does DeFiShield work?',
    a: 'DeFiShield intercepts transactions before they are broadcast to the blockchain. It extracts 20 on-chain features using our DeFiTransLyzer pipeline, then evaluates them through a dual-model ensemble (Random Forest classifier and DeFiGuard neural network). The entire evaluation completes in milliseconds, providing a calibrated risk score and transparent explainability breakdown.',
  },
  {
    q: 'Can it detect new, unseen scams?',
    a: 'Yes. The models are trained on feature-level structural and behavioral signals rather than static blacklists or contract signatures. It detects anomalous gas dynamics, drainer self-transfer patterns, and unusual transaction parameters that characterize novel zero-day attacks.',
  },
  {
    q: 'Does it require wallet transaction history?',
    a: 'No. DeFiShield analyzes each transaction in isolation using the transaction payload and parameters at the time of signature. Prior wallet history is not required for accurate fraud detection.',
  },
  {
    q: 'Is DeFiShield open source?',
    a: 'Yes. The full codebase — including the React frontend dashboard, FastAPI dual-model inference service, feature extraction pipeline, and test suite — is fully open source on GitHub.',
  },
  {
    q: 'Which blockchains are supported?',
    a: 'DeFiShield supports Ethereum Mainnet and all EVM-compatible networks and testnets (such as Sepolia). Any transaction payload adhering to standard EVM parameters can be evaluated directly.',
  },
];

function FAQItem({ faq, index, inView }) {
  const [open, setOpen] = useState(false);

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ delay: index * 0.1 }}
      className="glass-card rounded-2xl border border-white/5 hover:border-cyan-500/15 transition-all duration-300 overflow-hidden"
    >
      <button
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between px-6 py-5 text-left group"
      >
        <span className={`font-semibold text-sm lg:text-base transition-colors ${open ? 'text-cyan-400' : 'text-white group-hover:text-cyan-300'}`}>
          {faq.q}
        </span>
        <motion.div
          animate={{ rotate: open ? 180 : 0 }}
          transition={{ duration: 0.3 }}
          className={`flex-shrink-0 ml-4 transition-colors ${open ? 'text-cyan-400' : 'text-gray-500 group-hover:text-gray-300'}`}
        >
          <ChevronDown className="w-5 h-5" />
        </motion.div>
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.35, ease: 'easeInOut' }}
          >
            <div className="px-6 pb-6">
              <div className="h-px bg-white/5 mb-4" />
              <p className="text-gray-400 text-sm leading-relaxed">{faq.a}</p>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

export default function FAQ() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true });

  return (
    <section id="faq" className="py-28 relative">
      <div className="max-w-4xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        <div ref={ref} className="text-center mb-16">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass border border-cyan-500/20 text-sm text-cyan-400 font-medium mb-6"
          >
            <HelpCircle className="w-4 h-4" />
            FAQ
          </motion.div>

          <motion.h2
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 0.1 }}
            className="text-4xl lg:text-5xl font-black text-white mb-4 font-[Manrope]"
          >
            Frequently Asked <span className="text-gradient">Questions</span>
          </motion.h2>

          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={inView ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 0.2 }}
            className="text-gray-400 text-lg"
          >
            Everything you need to know about DeFiShield.
          </motion.p>
        </div>

        <div className="space-y-3">
          {faqs.map((faq, idx) => (
            <FAQItem key={faq.q} faq={faq} index={idx} inView={inView} />
          ))}
        </div>
      </div>
    </section>
  );
}
