import { describe, expect, it } from 'vitest';
import { renderToString } from 'react-dom/server';

import evaluation from './modelEvaluation.json';
import { modelMetrics, pct } from './modelMetrics';
import Analytics from '../components/Analytics';
import Features from '../components/Features';
import Hero from '../components/Hero';
import HowItWorks from '../components/HowItWorks';

describe('modelMetrics', () => {
  it('reports the pending-view metrics at the configured threshold', () => {
    const row = evaluation.views.pending.metrics.find((m) => m.threshold === evaluation.threshold);
    expect(row).toBeDefined();
    expect(modelMetrics.precision).toBe(row.precision);
    expect(modelMetrics.recall).toBe(row.recall);
    expect(modelMetrics.fpr).toBe(row.fpr);
  });

  it('has outcome counts consistent with the sample and the metrics', () => {
    const o = modelMetrics.outcomes;
    expect(o.fraud_caught + o.fraud_missed).toBe(evaluation.sample.fraud);
    expect(o.legit_flagged + o.legit_passed).toBe(evaluation.sample.legit);
    expect(o.fraud_caught / evaluation.sample.fraud).toBeCloseTo(modelMetrics.recall, 3);
    expect(o.legit_flagged / evaluation.sample.legit).toBeCloseTo(modelMetrics.fpr, 3);
  });

  it('has a score histogram covering every evaluated transaction', () => {
    const total = modelMetrics.scoreHistogram.reduce((n, b) => n + b.fraud + b.legit, 0);
    expect(total).toBe(modelMetrics.sampleSize);
  });

  it('formats percentages', () => {
    expect(pct(0.7797)).toBe('78.0%');
  });
});

describe('components display measured figures', () => {
  const text = (Component) => renderToString(<Component />).replace(/<!-- -->/g, '');

  it.each([
    ['Hero', Hero],
    ['Analytics', Analytics],
    ['HowItWorks', HowItWorks],
    ['Features', Features],
  ])('%s renders without errors', (_name, Component) => {
    expect(() => text(Component)).not.toThrow();
  });

  it('shows no previously hard-coded marketing figures', () => {
    const html = [Hero, Analytics, HowItWorks, Features].map(text).join('\n');
    for (const fake of ['99.7%', '$4.2M', '2M+', '100.0%', '73.2 ms', '&lt;0.1%', '2,197,433', '38,291']) {
      expect(html).not.toContain(fake);
    }
    expect(html).toContain(pct(modelMetrics.precision));
    expect(html).toContain(pct(modelMetrics.recall));
  });
});
