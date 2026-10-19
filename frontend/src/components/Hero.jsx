import { useEffect, useRef } from 'react';
import { motion } from 'framer-motion';
import { ArrowRight, Shield, Zap, Brain, Activity } from 'lucide-react';
import { modelMetrics, pct } from '../data/modelMetrics';

// Animated blockchain canvas
function BlockchainCanvas() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    let animFrame;
    let time = 0;

    const resize = () => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    };
    resize();
    window.addEventListener('resize', resize);

    const nodes = Array.from({ length: 18 }, () => ({
      x: Math.random() * canvas.width,
      y: Math.random() * canvas.height,
      vx: (Math.random() - 0.5) * 0.4,
      vy: (Math.random() - 0.5) * 0.4,
      r: Math.random() * 3 + 2,
      pulse: Math.random() * Math.PI * 2,
      color: Math.random() > 0.5 ? '#06B6D4' : '#3B82F6',
    }));

    const draw = () => {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      time += 0.012;

      nodes.forEach((n) => {
        n.x += n.vx;
        n.y += n.vy;
        if (n.x < 0 || n.x > canvas.width) n.vx *= -1;
        if (n.y < 0 || n.y > canvas.height) n.vy *= -1;
      });

      // Draw connections
      for (let i = 0; i < nodes.length; i++) {
        for (let j = i + 1; j < nodes.length; j++) {
          const dx = nodes[i].x - nodes[j].x;
          const dy = nodes[i].y - nodes[j].y;
          const dist = Math.sqrt(dx * dx + dy * dy);
          if (dist < 180) {
            const alpha = (1 - dist / 180) * 0.4;
            const grad = ctx.createLinearGradient(nodes[i].x, nodes[i].y, nodes[j].x, nodes[j].y);
            grad.addColorStop(0, `rgba(6,182,212,${alpha})`);
            grad.addColorStop(1, `rgba(59,130,246,${alpha})`);
            ctx.beginPath();
            ctx.moveTo(nodes[i].x, nodes[i].y);
            ctx.lineTo(nodes[j].x, nodes[j].y);
            ctx.strokeStyle = grad;
            ctx.lineWidth = 1;
            ctx.stroke();

            // Animated pulse along the line
            if (dist < 120 && Math.random() > 0.97) {
              const t = (Math.sin(time) * 0.5) + 0.5;
              const px = nodes[i].x + (nodes[j].x - nodes[i].x) * t;
              const py = nodes[i].y + (nodes[j].y - nodes[i].y) * t;
              ctx.beginPath();
              ctx.arc(px, py, 2, 0, Math.PI * 2);
              ctx.fillStyle = '#06B6D4';
              ctx.fill();
            }
          }
        }
      }

      // Draw nodes
      nodes.forEach((n) => {
        const pulse = Math.sin(time * 2 + n.pulse) * 0.5 + 0.5;
        const glowRadius = n.r + pulse * 8;

        const gradient = ctx.createRadialGradient(n.x, n.y, 0, n.x, n.y, glowRadius * 3);
        gradient.addColorStop(0, n.color + 'CC');
        gradient.addColorStop(1, n.color + '00');
        ctx.beginPath();
        ctx.arc(n.x, n.y, glowRadius * 3, 0, Math.PI * 2);
        ctx.fillStyle = gradient;
        ctx.fill();

        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r + pulse * 2, 0, Math.PI * 2);
        ctx.fillStyle = n.color;
        ctx.fill();

        // Node ring
        ctx.beginPath();
        ctx.arc(n.x, n.y, n.r + 5, 0, Math.PI * 2);
        ctx.strokeStyle = n.color + '44';
        ctx.lineWidth = 1;
        ctx.stroke();
      });

      animFrame = requestAnimationFrame(draw);
    };

    draw();
    return () => {
      cancelAnimationFrame(animFrame);
      window.removeEventListener('resize', resize);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="absolute inset-0 w-full h-full opacity-50"
      style={{ zIndex: 0 }}
    />
  );
}

// Dashboard card showing one measured model figure (bar = the value as a share of 100%)
function DashboardCard({ icon: Icon, label, value, fraction, barColor, color, delay }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5 }}
      className="glass-card rounded-xl p-4 border border-white/5 hover:border-cyan-500/20 transition-all duration-300"
    >
      <div className="flex items-center justify-between mb-3">
        <div className={`w-9 h-9 rounded-lg flex items-center justify-center ${color}`}>
          <Icon className="w-4 h-4 text-white" />
        </div>
      </div>
      <div className="text-2xl font-bold text-white mb-1">{value}</div>
      <div className="text-xs text-gray-500 font-medium">{label}</div>
      {fraction != null && (
        <div className="mt-3 h-1 bg-gray-800 rounded-full overflow-hidden">
          <motion.div
            initial={{ width: 0 }}
            animate={{ width: `${Math.round(fraction * 100)}%` }}
            transition={{ delay: delay + 0.3, duration: 0.8 }}
            className={`h-full rounded-full ${barColor}`}
          />
        </div>
      )}
    </motion.div>
  );
}

// Score distribution of the evaluation sample (all labels), as bar heights
const scoreBars = modelMetrics.scoreHistogram.map((b) => b.fraud + b.legit);
const maxScoreBar = Math.max(...scoreBars);

export default function Hero() {
  const dashboardCards = [
    { icon: Shield, label: 'Model Precision', value: pct(modelMetrics.precision), fraction: modelMetrics.precision, barColor: 'bg-green-400', color: 'bg-gradient-to-br from-green-500 to-emerald-600', delay: 0.3 },
    { icon: Activity, label: 'Inference Latency (median)', value: `${modelMetrics.latencyMedianMs.toFixed(1)} ms`, fraction: null, color: 'bg-gradient-to-br from-cyan-500 to-blue-600', delay: 0.45 },
    { icon: Brain, label: 'Model Recall', value: pct(modelMetrics.recall), fraction: modelMetrics.recall, barColor: 'bg-purple-400', color: 'bg-gradient-to-br from-purple-500 to-violet-600', delay: 0.6 },
    { icon: Zap, label: 'False-Positive Rate', value: pct(modelMetrics.fpr), fraction: modelMetrics.fpr, barColor: 'bg-red-400', color: 'bg-gradient-to-br from-orange-500 to-red-600', delay: 0.75 },
  ];

  return (
    <section id="home" className="relative min-h-screen flex items-center overflow-hidden pt-16">
      {/* Background */}
      <div className="absolute inset-0 bg-[#0B1120]">
        <BlockchainCanvas />
        {/* Radial gradient overlay */}
        <div className="absolute inset-0 bg-radial-gradient" style={{
          background: 'radial-gradient(ellipse 80% 60% at 50% 0%, rgba(6,182,212,0.08) 0%, transparent 70%)'
        }} />
        <div className="absolute inset-0 grid-bg opacity-30" />
      </div>

      <div className="relative z-10 max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-20 lg:py-0">
        <div className="grid lg:grid-cols-2 gap-12 items-center">
          {/* Left Content */}
          <div>
            {/* Badge */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6 }}
              className="inline-flex items-center gap-2 px-4 py-2 rounded-full glass border border-cyan-500/20 text-sm text-cyan-400 font-medium mb-8"
            >
              <span className="w-2 h-2 bg-cyan-400 rounded-full animate-blink" />
              Web3 Security Platform
              <span className="px-2 py-0.5 bg-cyan-500/20 rounded-full text-xs">LIVE</span>
            </motion.div>

            {/* Headline */}
            <motion.h1
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.1 }}
              className="text-5xl lg:text-7xl font-black text-white leading-tight mb-6 font-[Manrope]"
            >
              Protect Your{' '}
              <span className="text-gradient">Crypto</span>{' '}
              Before You Sign.
            </motion.h1>

            {/* Subheading */}
            <motion.p
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.2 }}
              className="text-lg text-gray-400 leading-relaxed mb-10 max-w-xl"
            >
              DeFiShield is a real-time transaction firewall that analyzes blockchain
              transactions before execution and detects zero-day fraud using machine learning.
            </motion.p>

            {/* CTA Buttons */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.3 }}
              className="flex flex-wrap gap-4"
            >
              <motion.a
                href="#ai-explanation"
                className="flex items-center gap-2 px-8 py-3.5 rounded-xl font-semibold text-white bg-gradient-to-r from-cyan-500 to-blue-600 hover:shadow-2xl hover:shadow-cyan-500/30 transition-all duration-300 glow-cyan"
                whileHover={{ scale: 1.04, y: -2 }}
                whileTap={{ scale: 0.97 }}
              >
                <Zap className="w-5 h-5 text-yellow-300" />
                Launch Live Scanner
                <ArrowRight className="w-4 h-4" />
              </motion.a>

              <motion.a
                href="#features"
                className="flex items-center gap-2 px-8 py-3.5 rounded-xl font-semibold text-gray-300 glass border border-white/10 hover:border-cyan-500/30 hover:text-white transition-all duration-300"
                whileHover={{ scale: 1.04, y: -2 }}
                whileTap={{ scale: 0.97 }}
              >
                <Shield className="w-5 h-5 text-cyan-400" />
                Architecture
              </motion.a>
            </motion.div>

            {/* Stats Row */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.7, delay: 0.5 }}
              className="flex flex-wrap gap-8 mt-12 pt-8 border-t border-white/5"
            >
              {[
                { value: String(modelMetrics.features), label: 'Features Analyzed' },
                { value: pct(modelMetrics.recall), label: 'Fraud Recall' },
                { value: pct(modelMetrics.precision), label: 'Precision' },
                { value: `${modelMetrics.latencyMedianMs.toFixed(1)} ms`, label: 'Median Inference' },
              ].map((stat) => (
                <div key={stat.label}>
                  <div className="text-2xl font-bold text-white">{stat.value}</div>
                  <div className="text-sm text-gray-500">{stat.label}</div>
                </div>
              ))}
            </motion.div>
          </div>

          {/* Right: Dashboard Illustration */}
          <motion.div
            initial={{ opacity: 0, x: 60 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.9, delay: 0.2 }}
            className="relative hidden lg:block"
          >
            {/* Main Dashboard Panel */}
            <div className="relative">
              {/* Floating glow blobs */}
              <div className="absolute -top-10 -left-10 w-40 h-40 bg-cyan-500/20 rounded-full blur-3xl" />
              <div className="absolute -bottom-10 -right-10 w-40 h-40 bg-blue-500/20 rounded-full blur-3xl" />

              <div className="glass-card rounded-2xl p-6 border border-cyan-500/15 glow-cyan">
                {/* Header */}
                <div className="flex items-center justify-between mb-6">
                  <div>
                    <h3 className="font-bold text-white font-[Manrope]">Security Overview</h3>
                    <p className="text-xs text-gray-500">Real-time monitoring active</p>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-green-400 font-medium">
                    <span className="w-2 h-2 bg-green-400 rounded-full animate-pulse" />
                    PROTECTED
                  </div>
                </div>

                {/* Dashboard Cards Grid */}
                <div className="grid grid-cols-2 gap-3 mb-4">
                  {dashboardCards.map((card) => (
                    <DashboardCard key={card.label} {...card} />
                  ))}
                </div>

                {/* Mini chart bar */}
                <div className="glass rounded-xl p-4 border border-white/5">
                  <div className="flex items-center justify-between mb-3">
                    <span className="text-xs font-semibold text-gray-400">Fraud Score Distribution</span>
                    <span className="text-xs text-cyan-400">{modelMetrics.sampleSize.toLocaleString()} labelled txns</span>
                  </div>
                  <div className="flex items-end gap-1.5 h-16">
                    {scoreBars.map((count, i) => (
                      <motion.div
                        key={modelMetrics.scoreHistogram[i].range}
                        title={`${modelMetrics.scoreHistogram[i].range}: ${count.toLocaleString()} txns`}
                        initial={{ height: 0 }}
                        animate={{ height: `${Math.max((count / maxScoreBar) * 100, 3)}%` }}
                        transition={{ delay: 0.8 + i * 0.05, duration: 0.4 }}
                        className="flex-1 rounded-t bg-gradient-to-t from-cyan-600 to-cyan-400 opacity-80"
                      />
                    ))}
                  </div>
                </div>
              </div>

              {/* Floating Alert Badge */}
              <motion.div
                animate={{ y: [-5, 5, -5] }}
                transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut' }}
                className="absolute -top-4 -right-4 glass border border-red-500/30 rounded-xl px-4 py-2.5 shadow-lg shadow-red-500/10"
              >
                <div className="flex items-center gap-2">
                  <div className="w-2.5 h-2.5 bg-red-500 rounded-full animate-pulse" />
                  <div>
                    <div className="text-xs font-bold text-red-400">THREAT DETECTED</div>
                    <div className="text-xs text-gray-500">Risk Score: 94%</div>
                  </div>
                </div>
              </motion.div>

              {/* Floating Safe Badge */}
              <motion.div
                animate={{ y: [5, -5, 5] }}
                transition={{ duration: 3, repeat: Infinity, ease: 'easeInOut', delay: 1.5 }}
                className="absolute -bottom-4 -left-4 glass border border-green-500/30 rounded-xl px-4 py-2.5 shadow-lg shadow-green-500/10"
              >
                <div className="flex items-center gap-2">
                  <div className="w-2.5 h-2.5 bg-green-500 rounded-full animate-pulse" />
                  <div>
                    <div className="text-xs font-bold text-green-400">TRANSACTION SAFE</div>
                    <div className="text-xs text-gray-500">Verified Secure</div>
                  </div>
                </div>
              </motion.div>
            </div>
          </motion.div>
        </div>
      </div>

      {/* Bottom gradient fade */}
      <div className="absolute bottom-0 left-0 right-0 h-32 bg-gradient-to-t from-[#0B1120] to-transparent" />
    </section>
  );
}
