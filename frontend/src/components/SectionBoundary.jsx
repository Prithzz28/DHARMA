import { Component } from 'react';
import { AlertTriangle } from 'lucide-react';

/**
 * Error boundary around one page section: if the section throws while
 * rendering, it is replaced by a short notice and the rest of the page keeps
 * working (without it, React unmounts the whole app).
 */
export default class SectionBoundary extends Component {
  constructor(props) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error) {
    return { error };
  }

  componentDidCatch(error, info) {
    console.error(`Section "${this.props.name}" failed to render:`, error, info.componentStack);
  }

  render() {
    if (!this.state.error) return this.props.children;
    return (
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10">
        <div className="flex items-start gap-3 p-4 rounded-xl border border-amber-500/20 bg-amber-500/5 text-sm text-gray-400">
          <AlertTriangle className="w-5 h-5 text-amber-400 shrink-0" />
          <div>
            <p>
              <span className="text-amber-300 font-semibold">{this.props.name}</span> could not be displayed
              ({this.state.error.message}). The rest of the page still works.
            </p>
            <button
              type="button"
              onClick={() => this.setState({ error: null })}
              className="mt-2 text-xs text-cyan-400 hover:text-cyan-300"
            >
              Try again
            </button>
          </div>
        </div>
      </div>
    );
  }
}
