import Navbar from './components/Navbar';
import Hero from './components/Hero';
import ConnectWallet from './components/ConnectWallet';
import FirewallFeed from './components/FirewallFeed';
import Features from './components/Features';
import HowItWorks from './components/HowItWorks';
import Analytics from './components/Analytics';
import AIExplanation from './components/AIExplanation';
import FAQ from './components/FAQ';
import Footer from './components/Footer';
import SectionBoundary from './components/SectionBoundary';

// Each section gets its own error boundary, so one failing section (e.g. an
// unexpected API response) cannot blank the whole page.
const MAIN_SECTIONS = [
  ['Hero', Hero],
  ['Connect Your Wallet', ConnectWallet],
  ['Live firewall feed', FirewallFeed],
  ['Features', Features],
  ['How it works', HowItWorks],
  ['Analytics', Analytics],
  ['Risk scanner', AIExplanation],
  ['FAQ', FAQ],
];

export default function App() {
  return (
    <div className="min-h-screen bg-[#0B1120] font-inter overflow-x-hidden">
      <SectionBoundary name="Navigation">
        <Navbar />
      </SectionBoundary>
      <main>
        {MAIN_SECTIONS.map(([name, Section]) => (
          <SectionBoundary key={name} name={name}>
            <Section />
          </SectionBoundary>
        ))}
      </main>
      <SectionBoundary name="Footer">
        <Footer />
      </SectionBoundary>
    </div>
  );
}
