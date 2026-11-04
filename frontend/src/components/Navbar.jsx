import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, Menu, X, Zap } from 'lucide-react';
import { checkHealth } from '../services/api';

const navLinks = [
  { name: 'Home', href: '#home' },
  { name: 'Connect Wallet', href: '#connect' },
  { name: 'Risk Scanner', href: '#ai-explanation' },
  { name: 'Features', href: '#features' },
  { name: 'How It Works', href: '#how-it-works' },
  { name: 'Analytics', href: '#analytics' },
  { name: 'FAQ', href: '#faq' },
];

export default function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  // Section to scroll to once the mobile menu has finished closing. Following
  // the anchor immediately does not work: the closing animation interrupts the
  // page's smooth scroll, leaving the user at the top with only the URL changed.
  const pendingTarget = useRef(null);

  const goToFromMenu = (event, href) => {
    event.preventDefault();
    pendingTarget.current = href;
    setMobileOpen(false);
  };

  const scrollToPendingTarget = () => {
    const href = pendingTarget.current;
    pendingTarget.current = null;
    if (!href) return;
    document.querySelector(href)?.scrollIntoView({ behavior: 'smooth' });
    window.history.pushState(null, '', href);
  };
  const [healthStatus, setHealthStatus] = useState({
    online: false,
    modelLoaded: false,
  });

  useEffect(() => {
    const onScroll = () => {
      setScrolled(window.scrollY > 20);
    };
    window.addEventListener('scroll', onScroll);
    return () => window.removeEventListener('scroll', onScroll);
  }, []);

  useEffect(() => {
    let isMounted = true;
    const fetchHealth = async () => {
      try {
        const res = await checkHealth();
        if (isMounted) {
          setHealthStatus({
            online: res.status === 'ok',
            modelLoaded: Boolean(res.model_loaded),
          });
        }
      } catch {
        if (isMounted) {
          setHealthStatus({ online: false, modelLoaded: false });
        }
      }
    };

    fetchHealth();
    const interval = setInterval(fetchHealth, 10000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  return (
    <motion.nav
      initial={{ y: -80, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      transition={{ duration: 0.6, ease: 'easeOut' }}
      className={`fixed top-0 left-0 right-0 z-50 transition-all duration-500 ${
        scrolled
          ? 'bg-[#0B1120]/90 backdrop-blur-xl border-b border-cyan-500/10 shadow-lg shadow-cyan-500/5'
          : 'bg-transparent'
      }`}
    >
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-16 lg:h-[72px]">
          {/* Logo & Backend Health */}
          <div className="flex items-center gap-4">
            <motion.a
              href="#home"
              className="flex items-center gap-2.5 group"
              whileHover={{ scale: 1.03 }}
            >
              <div className="relative">
                <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-cyan-500 to-blue-600 flex items-center justify-center glow-cyan">
                  <Shield className="w-5 h-5 text-white" />
                </div>
                <div
                  className={`absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full border-2 border-[#0B1120] ${
                    healthStatus.online ? 'bg-green-400 animate-pulse' : 'bg-red-400'
                  }`}
                />
              </div>
              <span className="text-xl font-bold text-white font-[Manrope]">
                DeFi<span className="text-gradient-cyan">Shield</span>
              </span>
            </motion.a>
          </div>

          {/* Desktop Nav */}
          <div className="hidden lg:flex items-center gap-1">
            {navLinks.map((link) => (
              <a
                key={link.name}
                href={link.href}
                className="px-3.5 py-2 text-sm text-gray-400 hover:text-white rounded-lg hover:bg-white/5 transition-all duration-200 font-medium"
              >
                {link.name}
              </a>
            ))}
          </div>

          {/* CTA */}
          <div className="hidden lg:flex items-center gap-3">
            <motion.a
              href="#ai-explanation"
              className="flex items-center gap-2 px-5 py-2 rounded-lg text-sm font-semibold bg-gradient-to-r from-cyan-500 to-blue-600 text-white transition-all duration-300 hover:shadow-lg hover:shadow-cyan-500/30 hover:-translate-y-0.5"
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
            >
              <Zap className="w-4 h-4" />
              Scan Transaction
            </motion.a>
          </div>

          {/* Mobile Menu Toggle */}
          <button
            className="lg:hidden p-2 -mr-2 text-gray-400 hover:text-white"
            onClick={() => setMobileOpen(!mobileOpen)}
            aria-label="Toggle menu"
            aria-expanded={mobileOpen}
          >
            {mobileOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>
        </div>
      </div>

      {/* Mobile Menu */}
      <AnimatePresence onExitComplete={scrollToPendingTarget}>
        {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="lg:hidden bg-[#0B1120]/95 backdrop-blur-xl border-b border-cyan-500/10"
          >
            <div className="px-4 py-4 flex flex-col gap-2">
              {navLinks.map((link) => (
                <a
                  key={link.name}
                  href={link.href}
                  onClick={(event) => goToFromMenu(event, link.href)}
                  className="px-4 py-3 text-sm text-gray-300 hover:text-white rounded-lg hover:bg-white/5 transition-all duration-200 font-medium"
                >
                  {link.name}
                </a>
              ))}
              <a
                href="#ai-explanation"
                onClick={(event) => goToFromMenu(event, '#ai-explanation')}
                className="mt-2 flex items-center justify-center gap-2 px-5 py-2.5 rounded-lg text-sm font-semibold bg-gradient-to-r from-cyan-500 to-blue-600 text-white"
              >
                <Zap className="w-4 h-4" />
                Scan Transaction
              </a>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.nav>
  );
}
