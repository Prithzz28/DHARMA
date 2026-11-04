import { useState, useRef } from 'react';
import { motion, useInView, AnimatePresence } from 'framer-motion';
import { HelpCircle, ChevronDown } from 'lucide-react';

const faqs = [
  {
    q: 'How does DeFiShield work?',
    a: 'You add the DeFiGuard RPC URL to your wallet as a network. When you send a transaction it reaches the firewall before the network. Two independent checks run: counterparty checks (is the recipient, the approved spender or the token recipient a wallet labelled fraudulent, and are you granting an unlimited or approve-all permission to a plain wallet?) and the DeFiGuard model. A counterparty red flag blocks the transaction; a high model score or a softer finding forwards it with a warning in the live feed. Every other wallet call passes straight through to the Ethereum node.',
  },
  {
    q: 'What exactly does the model detect?',
    a: 'DeFiGuard was trained on the BCCC-DeFiFraudTrans-2025 dataset, whose labels are per wallet: every transaction sent by a wallet labelled fraudulent counts as fraud. So the model scores how closely a transaction resembles ones sent by fraud-labelled wallets. That is not the same as "this transaction will hurt you" — the harm to a user usually comes from the other side of the transaction, which is why the firewall blocks on counterparty checks and lets the model only warn.',
  },
  {
    q: 'Why does my new wallet get a warning on a normal transfer?',
    a: 'Fraud-labelled wallets in the dataset tend to be young, so the model gives a high score to accounts with few past transactions: a plain transfer scores around 85% at nonce 0 and stays above the warning threshold for roughly the first 50 transactions. That is the honest limit of a sender-pattern model, and the reason it warns rather than blocks.',
  },
  {
    q: 'Can it detect new, unseen scams?',
    a: 'Partly. The model is evaluated only on wallets it never saw in training (and on later blocks than it was trained on), so the figures on this page are for unseen wallets — but forward-in-time accuracy is clearly lower than on the balanced sample. The counterparty list only knows the 1,763 wallets in the dataset; the approval rules catch the drainer pattern regardless of address. Scams that rely on off-chain signatures (permits, marketplace orders) never reach an RPC endpoint and cannot be seen.',
  },
  {
    q: 'Is it safe to use with real funds?',
    a: 'No — this is a testnet demo (Sepolia by default). Your wallet traffic, including addresses and balances, passes through this server, and the model flags too many legitimate transactions to stand guard over real money. Use a testnet wallet. (The model was trained on Ethereum mainnet transactions; its accuracy on testnets has not been measured.)',
  },
  {
    q: 'Does it use my wallet history?',
    a: 'The model does not: it sees only the transaction itself (value, gas, nonce, calldata intent), never address history or identity. The counterparty checks compare addresses with a fixed list and ask the node whether an approved spender is a contract. The server keeps the last 50 verdicts, with addresses masked, for the live feed.',
  },
  {
    q: 'Is DeFiShield open source?',
    a: 'Yes. The full codebase — the React dashboard, the FastAPI inference and RPC firewall service, feature extraction, counterparty rules and the test suite — is open source on GitHub.',
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
