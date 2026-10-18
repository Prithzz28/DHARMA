import { motion } from 'framer-motion';
import { Shield, ArrowUpRight, Heart } from 'lucide-react';

export default function Footer() {
  const year = new Date().getFullYear();

  return (
    <footer id="footer" className="relative pt-20 pb-8 border-t border-white/5">
      {/* Top glow */}
      <div className="absolute top-0 left-1/2 -translate-x-1/2 w-96 h-px bg-gradient-to-r from-transparent via-cyan-500/50 to-transparent" />
      <div
        className="absolute top-0 left-0 right-0 h-32"
        style={{ background: 'radial-gradient(ellipse 50% 100% at 50% 0%, rgba(6,182,212,0.04) 0%, transparent 70%)' }}
      />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
        {/* CTA Strip */}
        <motion.div
          initial={{ opacity: 0, y: 30 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass-card rounded-3xl p-8 lg:p-12 border border-cyan-500/15 mb-20 text-center relative overflow-hidden"
          style={{ background: 'linear-gradient(135deg, rgba(6,182,212,0.06) 0%, rgba(59,130,246,0.06) 100%)' }}
        >
          <div className="absolute inset-0 grid-bg opacity-20" />
          <div className="absolute -top-20 -right-20 w-60 h-60 bg-cyan-500/10 rounded-full blur-3xl" />
          <div className="absolute -bottom-20 -left-20 w-60 h-60 bg-blue-500/10 rounded-full blur-3xl" />
          <div className="relative z-10">
            <h2 className="text-3xl lg:text-4xl font-black text-white mb-4 font-[Manrope]">
              Start Protecting Your Assets{' '}
              <span className="text-gradient">Today</span>
            </h2>
            <p className="text-gray-400 mb-8 max-w-xl mx-auto">
              Protect your wallet transactions in real time with machine learning intelligence and transparent risk explainability.
            </p>
            <div className="flex flex-wrap gap-4 justify-center">
              <motion.a
                href="#analytics"
                className="flex items-center gap-2 px-8 py-3.5 rounded-xl font-bold text-white bg-gradient-to-r from-cyan-500 to-blue-600 hover:shadow-2xl hover:shadow-cyan-500/30 transition-all duration-300"
                whileHover={{ scale: 1.04, y: -2 }}
              >
                <Shield className="w-5 h-5" />
                View Analytics
              </motion.a>
              <motion.a
                href="https://github.com/harshal6426/DHARMA"
                target="_blank"
                rel="noopener noreferrer"
                className="flex items-center gap-2 px-8 py-3.5 rounded-xl font-bold text-gray-300 glass border border-white/10 hover:border-cyan-500/30 hover:text-white transition-all duration-300"
                whileHover={{ scale: 1.04, y: -2 }}
              >
                <svg className="w-5 h-5" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z"/></svg>
                View on GitHub
                <ArrowUpRight className="w-4 h-4" />
              </motion.a>
            </div>
          </div>
        </motion.div>

        {/* Brand Section */}
        <div className="flex flex-col items-center text-center mb-16">
          <a href="#home" className="flex items-center gap-2.5 mb-4">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center glow-cyan">
              <Shield className="w-5 h-5 text-white" />
            </div>
            <span className="text-xl font-bold text-white font-[Manrope]">
              DeFi<span className="text-gradient-cyan">Shield</span>
            </span>
          </a>
          <p className="text-gray-500 text-sm leading-relaxed mb-5 max-w-md">
            Real-time transaction firewall for the decentralized web.
          </p>
          <div className="flex gap-3">
            <a
              href="https://github.com/harshal6426/DHARMA"
              target="_blank"
              rel="noopener noreferrer"
              aria-label="GitHub"
              className="w-9 h-9 rounded-lg glass border border-white/8 hover:border-cyan-500/30 flex items-center justify-center text-gray-500 hover:text-cyan-400 transition-all duration-200"
            >
              <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 24 24"><path d="M12 0C5.374 0 0 5.373 0 12c0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23A11.509 11.509 0 0112 5.803c1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576C20.566 21.797 24 17.3 24 12c0-6.627-5.373-12-12-12z"/></svg>
            </a>
          </div>
        </div>

        {/* Bottom bar */}
        <div className="pt-6 border-t border-white/5 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div className="text-sm text-gray-600">
            © {year} DeFiShield. All rights reserved.
          </div>
          <div className="flex items-center gap-1.5 text-sm text-gray-600">
            Built with <Heart className="w-3.5 h-3.5 text-red-500 fill-red-500 mx-0.5" /> for Web3 security
          </div>
          <div className="flex gap-4 text-xs text-gray-600">
            <a href="#" className="hover:text-white transition-colors">Privacy</a>
            <a href="#" className="hover:text-white transition-colors">Terms</a>
            <a href="#" className="hover:text-white transition-colors">Security</a>
          </div>
        </div>
      </div>
    </footer>
  );
}
