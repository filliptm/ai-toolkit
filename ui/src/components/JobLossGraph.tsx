'use client';

import { Job } from '@prisma/client';
import useJobLossLog, { LossPoint } from '@/hooks/useJobLossLog';
import { useMemo, useState, useEffect, useRef, useCallback } from 'react';
import uPlot from 'uplot';
import 'uplot/dist/uPlot.min.css';

interface Props {
  job: Job;
  embedded?: boolean;
}

function formatNum(v: number) {
  if (!Number.isFinite(v)) return '';
  if (Math.abs(v) >= 1000) return v.toFixed(0);
  if (Math.abs(v) >= 10) return v.toFixed(3);
  if (Math.abs(v) >= 1) return v.toFixed(4);
  return v.toPrecision(4);
}

function formatStep(v: number) {
  if (!Number.isFinite(v)) return '';
  const abs = Math.abs(v);
  if (abs >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}m`;
  if (abs >= 1000) return `${(v / 1000).toFixed(abs >= 10_000 ? 0 : 1)}k`;
  return v.toFixed(0);
}

function clamp01(x: number) {
  return Math.max(0, Math.min(1, x));
}

// Fallback canvas height used before the container has been measured.
const FALLBACK_CANVAS_HEIGHT = 360;
const MIN_CANVAS_HEIGHT = 160;

// Compute canvas size so uPlot's canvas + its HTML legend fit inside `host`.
// `host` should be a layout-controlled wrapper (NOT the uPlot mount node, since
// uPlot's stylesheet sets `width: min-content` on its mount node).
function computeCanvasSize(host: HTMLElement): { width: number; height: number } | null {
  const { width, height } = host.getBoundingClientRect();
  if (width <= 0 || height <= 0) return null;
  const legend = host.querySelector('.u-legend') as HTMLElement | null;
  const legendH = legend?.getBoundingClientRect().height ?? 0;
  return { width, height: Math.max(MIN_CANVAS_HEIGHT, height - legendH) };
}

// EMA over a (number|null)[] series. Nulls are preserved as gaps and do not
// advance the running average.
function emaWithNulls(ys: (number | null)[], alpha: number): (number | null)[] {
  const out: (number | null)[] = new Array(ys.length);
  let prev: number | null = null;
  for (let i = 0; i < ys.length; i++) {
    const v = ys[i];
    if (v === null || !Number.isFinite(v)) {
      out[i] = null;
      continue;
    }
    if (prev === null) prev = v as number;
    else prev = alpha * (v as number) + (1 - alpha) * prev;
    out[i] = prev;
  }
  return out;
}

function hashToIndex(str: string, mod: number) {
  let h = 2166136261;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  return Math.abs(h) % mod;
}

const PALETTE = [
  'rgba(96,165,250,1)', // blue-400
  'rgba(52,211,153,1)', // emerald-400
  'rgba(167,139,250,1)', // purple-400
  'rgba(251,191,36,1)', // amber-400
  'rgba(244,114,182,1)', // pink-400
  'rgba(248,113,113,1)', // red-400
  'rgba(34,211,238,1)', // cyan-400
  'rgba(129,140,248,1)', // indigo-400
];

function strokeForKey(key: string) {
  return PALETTE[hashToIndex(key, PALETTE.length)];
}

function dulledColor(rgba: string): string {
  const m = rgba.match(/rgba?\((\d+),(\d+),(\d+)/);
  if (!m) return 'rgba(120,120,120,1)';
  const r = Math.round(Number(m[1]) * 0.55);
  const g = Math.round(Number(m[2]) * 0.55);
  const b = Math.round(Number(m[3]) * 0.55);
  return `rgba(${r},${g},${b},1)`;
}

export default function JobLossGraph({ job, embedded = false }: Props) {
  const { series, lossKeys, status, refreshLoss } = useJobLossLog(job.id, 2000);

  // Controls
  const [useLogScale, setUseLogScale] = useState(false);
  const [showRaw, setShowRaw] = useState(false);
  const [showSmoothed, setShowSmoothed] = useState(true);

  // 0..100 slider. 100 = no smoothing, 0 = heavy smoothing.
  const [smoothing, setSmoothing] = useState(80);

  // UI-only downsample for rendering speed
  const [plotStride, setPlotStride] = useState(1);

  // show only last N points in the chart (0 = all)
  const [windowSize] = useState<number>(0);

  // quick y clipping for readability
  const [clipOutliers, setClipOutliers] = useState(false);

  // which loss series are enabled (default: all enabled)
  const [enabled, setEnabled] = useState<Record<string, boolean>>({});

  const [isZoomed, setIsZoomed] = useState(false);

  // keep enabled map in sync with discovered keys (enable new ones automatically)
  useEffect(() => {
    setEnabled(prev => {
      const next = { ...prev };
      for (const k of lossKeys) {
        if (next[k] === undefined) next[k] = true;
      }
      for (const k of Object.keys(next)) {
        if (!lossKeys.includes(k)) delete next[k];
      }
      return next;
    });
  }, [lossKeys]);

  const activeKeys = useMemo(() => lossKeys.filter(k => enabled[k] !== false), [lossKeys, enabled]);

  // Build uPlot-aligned data + series configs.
  const built = useMemo(() => {
    const stride = Math.max(1, plotStride | 0);
    const t = clamp01(smoothing / 100);
    const alpha = 1.0 - t * 0.98; // 1.0 -> 0.02
    const fullAlpha = 0.005;

    // Union of all steps across active series.
    const stepSet = new Set<number>();
    for (const key of activeKeys) {
      const pts: LossPoint[] = series[key] ?? [];
      for (const p of pts) {
        if (p.value === null || !Number.isFinite(p.value as number)) continue;
        if (useLogScale && (p.value as number) <= 0) continue;
        stepSet.add(p.step);
      }
    }
    let xs = Array.from(stepSet).sort((a, b) => a - b);
    if (stride > 1) xs = xs.filter((_, i) => i % stride === 0);
    if (windowSize > 0 && xs.length > windowSize) xs = xs.slice(xs.length - windowSize);

    const xsSet = new Set(xs);

    const data: (number[] | (number | null)[])[] = [xs];
    const seriesConfigs: uPlot.Series[] = [{}]; // x

    for (const key of activeKeys) {
      const pts: LossPoint[] = series[key] ?? [];
      const map = new Map<number, number>();
      for (const p of pts) {
        if (p.value === null || !Number.isFinite(p.value as number)) continue;
        if (useLogScale && (p.value as number) <= 0) continue;
        if (!xsSet.has(p.step)) continue;
        map.set(p.step, p.value as number);
      }
      const raw: (number | null)[] = xs.map(s => (map.has(s) ? (map.get(s) as number) : null));
      const smooth = emaWithNulls(raw, alpha);
      const fullSmooth = emaWithNulls(raw, fullAlpha);

      const color = strokeForKey(key);
      const colorFaded = color.replace('1)', '0.40)');
      const colorDull = dulledColor(color);

      if (showRaw) {
        data.push(raw);
        seriesConfigs.push({
          label: `${key} (raw)`,
          stroke: colorFaded,
          width: 1.25,
          spanGaps: false,
          points: { show: false },
        });
      }
      if (showSmoothed) {
        data.push(smooth);
        seriesConfigs.push({
          label: key,
          stroke: color,
          width: 2,
          spanGaps: false,
          points: { show: false },
        });
      }
      data.push(fullSmooth);
      seriesConfigs.push({
        label: `${key} (trend)`,
        stroke: colorDull,
        width: 2.5,
        spanGaps: false,
        points: { show: false },
      });
    }

    // y-domain clipping (2nd–98th percentile of all visible y values).
    let yClip: { min: number; max: number } | null = null;
    if (clipOutliers && xs.length >= 10) {
      const vals: number[] = [];
      for (let s = 1; s < data.length; s++) {
        const arr = data[s] as (number | null)[];
        for (const v of arr) {
          if (v !== null && Number.isFinite(v)) vals.push(v as number);
        }
      }
      if (vals.length >= 10) {
        vals.sort((a, b) => a - b);
        const lo = vals[Math.floor(vals.length * 0.02)];
        const hi = vals[Math.ceil(vals.length * 0.98) - 1];
        if (Number.isFinite(lo) && Number.isFinite(hi) && lo !== hi) {
          yClip = { min: lo, max: hi };
        }
      }
    }

    return { data: data as uPlot.AlignedData, seriesConfigs, yClip };
  }, [series, activeKeys, smoothing, plotStride, windowSize, useLogScale, showRaw, showSmoothed, clipOutliers]);

  // Layout wrapper we measure for sizing — uPlot collapses its own mount node
  // to width:min-content, so we can't read sizes off it.
  const chartHostRef = useRef<HTMLDivElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);
  const uplotRef = useRef<uPlot | null>(null);

  // Latest yClip read by the y-scale range fn — kept current via effect.
  const yClipRef = useRef<{ min: number; max: number } | null>(null);
  useEffect(() => {
    yClipRef.current = built.yClip;
  }, [built.yClip]);

  // Track zoom state via ref so the data-update effect can decide whether to refit scales.
  const isZoomedRef = useRef(false);
  useEffect(() => {
    isZoomedRef.current = isZoomed;
  }, [isZoomed]);

  // Structural recreate key — recreate uPlot only when the series shape or
  // axis distribution changes. Data updates go through setData.
  const hasData = (built.data[0]?.length ?? 0) > 1;
  const structuralKey = useMemo(
    () => `${activeKeys.join('|')}|raw=${showRaw}|sm=${showSmoothed}|log=${useLogScale}|has=${hasData}`,
    [activeKeys, showRaw, showSmoothed, useLogScale, hasData],
  );

  useEffect(() => {
    if (uplotRef.current) {
      uplotRef.current.destroy();
      uplotRef.current = null;
    }
    if (!containerRef.current || !chartHostRef.current) return;
    if (!hasData) return;

    const host = chartHostRef.current;
    const rect = host.getBoundingClientRect();
    const initialHeight = rect.height > 0 ? Math.max(MIN_CANVAS_HEIGHT, rect.height - 40) : FALLBACK_CANVAS_HEIGHT;
    const opts: uPlot.Options = {
      width: rect.width || 800,
      height: initialHeight,
      padding: [10, 16, 2, 4],
      series: built.seriesConfigs,
      scales: {
        x: { time: false },
        y: {
          distr: useLogScale ? 3 : 1,
          range: (_u, dataMin, dataMax) => {
            const c = yClipRef.current;
            if (c) return [c.min, c.max];
            return [dataMin, dataMax];
          },
        },
      },
      axes: [
        {
          stroke: 'rgba(255,255,255,0.55)',
          grid: { stroke: 'rgba(255,255,255,0.06)' },
          ticks: { stroke: 'rgba(255,255,255,0.15)' },
          size: embedded ? 24 : 32,
          values: (_u, ticks) => ticks.map(tk => formatStep(tk)),
          label: embedded ? undefined : 'Step',
          labelSize: embedded ? 0 : 20,
        },
        {
          stroke: 'rgba(255,255,255,0.55)',
          grid: { stroke: 'rgba(255,255,255,0.06)' },
          ticks: { stroke: 'rgba(255,255,255,0.15)' },
          size: 60,
          values: (_u, ticks) => ticks.map(tk => formatNum(tk)),
        },
      ],
      cursor: {
        drag: { x: true, y: false, setScale: true },
        points: { size: 6 },
      },
      legend: { show: !embedded },
      hooks: {
        setScale: [
          (u, key) => {
            if (key !== 'x') return;
            const xs = u.data[0] as number[];
            if (!xs || !xs.length) return;
            const sx = u.scales.x;
            const zoomed = sx.min !== xs[0] || sx.max !== xs[xs.length - 1];
            setIsZoomed(zoomed);
          },
        ],
      },
    };

    uplotRef.current = new uPlot(opts, built.data, containerRef.current);
    setIsZoomed(false);

    // After uPlot mounts its legend, right-size the canvas against the actual
    // legend height so the canvas fills the remaining vertical space.
    const fitted = computeCanvasSize(host);
    if (fitted) uplotRef.current.setSize(fitted);

    return () => {
      uplotRef.current?.destroy();
      uplotRef.current = null;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [structuralKey]);

  // Push new data without recreating — preserves zoom & cursor state.
  useEffect(() => {
    const u = uplotRef.current;
    if (!u) return;
    // When zoomed, pass resetScales=false so the user's view stays put. uPlot
    // skips its commit() in that branch though, so force a redraw to actually
    // re-render the new smoothed/strided values within the zoom window.
    if (isZoomedRef.current) {
      u.setData(built.data, false);
      u.redraw(true, true);
    } else {
      u.setData(built.data, true);
    }
  }, [built]);

  // Resize observer — fit canvas to the wrapper's available space minus the
  // HTML legend uPlot renders below it. Observe the layout wrapper, not the
  // uPlot mount node (which uPlot pins to width:min-content).
  // Re-runs on `hasData` because the observed element only exists once data
  // has loaded (see the `!hasData` branch below).
  useEffect(() => {
    const el = chartHostRef.current;
    if (!el) return;
    const ro = new ResizeObserver(() => {
      const u = uplotRef.current;
      if (!u) return;
      const fitted = computeCanvasSize(el);
      if (fitted) u.setSize(fitted);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, [hasData]);

  const handleResetZoom = useCallback(() => {
    const u = uplotRef.current;
    if (!u) return;
    const xs = u.data[0] as number[];
    if (!xs || !xs.length) return;
    u.setScale('x', { min: xs[0], max: xs[xs.length - 1] });
  }, []);

  const totalPoints = built.data[0]?.length ?? 0;
  const xSteps = built.data[0] as number[] | undefined;
  const firstStep = xSteps && xSteps.length > 0 ? xSteps[0] : null;
  const lastStep = xSteps && xSteps.length > 0 ? xSteps[xSteps.length - 1] : null;
  const outerClassName = embedded
    ? 'flex h-full min-h-0 flex-col overflow-hidden'
    : 'bg-gray-900 rounded-xl shadow-lg overflow-hidden border border-gray-800 flex flex-col h-full';
  const chartPaddingClassName = embedded ? 'px-2 pt-2 pb-1' : 'px-4 pt-4 pb-4';
  const controlsPaddingClassName = embedded ? 'px-2 pb-2' : 'px-4 pb-2';
  const controlsGridClassName = embedded ? 'grid grid-cols-2 gap-1' : 'grid grid-cols-1 md:grid-cols-2 gap-3';
  const controlPanelClassName = embedded
    ? 'min-w-0 bg-gray-950 border border-gray-800 rounded-md p-1.5'
    : 'bg-gray-950 border border-gray-800 rounded-lg p-3';
  const controlLabelClassName = embedded
    ? 'block text-[9px] uppercase leading-3 text-gray-500 mb-0.5'
    : 'block text-xs text-gray-400 mb-2';
  const sliderLabelClassName = embedded ? 'block text-[10px] text-gray-400' : 'block text-xs text-gray-400';

  return (
    <div className={outerClassName}>
      {!embedded && (
        <div className="bg-gray-800 px-4 py-3 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <div className="h-2 w-2 rounded-full bg-blue-400" />
            <h2 className="text-gray-100 text-sm font-medium">Loss graph</h2>
            <span className="text-xs text-gray-400">
              {status === 'loading' && 'Loading...'}
              {status === 'refreshing' && 'Refreshing...'}
              {status === 'error' && 'Error'}
              {status === 'success' && hasData && `${totalPoints.toLocaleString()} steps`}
              {status === 'success' && !hasData && 'No data yet'}
            </span>
          </div>

          <button
            type="button"
            onClick={refreshLoss}
            className="px-3 py-1 rounded-md text-xs bg-gray-700/60 hover:bg-gray-700 text-gray-200 border border-gray-700"
          >
            Refresh
          </button>
        </div>
      )}

      {/* Chart */}
      <div className={`${chartPaddingClassName} flex-1 min-h-0 flex flex-col`}>
        <div className="bg-gray-950 rounded-lg border border-gray-800 relative select-none flex-1 min-h-0" style={{ minHeight: embedded ? 150 : 240 }}>
          {!hasData ? (
            <div className="absolute inset-0 flex items-center justify-center text-sm text-gray-400">
              {status === 'error' ? 'Failed to load loss logs.' : 'Waiting for loss points...'}
            </div>
          ) : (
            <>
              {isZoomed && (
                <button
                  type="button"
                  onClick={handleResetZoom}
                  className="absolute top-2 right-2 z-10 px-2 py-1 rounded text-xs bg-blue-600/80 hover:bg-blue-600 text-white border border-blue-500/50"
                >
                  Reset zoom
                </button>
              )}
              <div ref={chartHostRef} className="absolute inset-0 overflow-hidden">
                <div ref={containerRef} />
              </div>
            </>
          )}
        </div>
      </div>

      {/* Controls */}
      <div className={`${controlsPaddingClassName} shrink-0`}>
        {embedded && (
          <div className="mb-1 flex items-center justify-between gap-2 text-[10px] leading-3 text-gray-400">
            <span className="min-w-0 truncate">
              {status === 'loading' && 'Loading...'}
              {status === 'refreshing' && 'Refreshing...'}
              {status === 'error' && 'Error'}
              {status === 'success' &&
                hasData &&
                `${totalPoints.toLocaleString()} pts${firstStep != null && lastStep != null ? `, steps ${formatStep(firstStep)}-${formatStep(lastStep)}` : ''}`}
              {status === 'success' && !hasData && 'No data yet'}
            </span>
            <button
              type="button"
              onClick={refreshLoss}
              className="shrink-0 rounded border border-gray-800 bg-gray-950 px-1.5 py-0.5 text-[10px] text-gray-300 hover:bg-gray-800"
            >
              Refresh
            </button>
          </div>
        )}
        <div className={embedded ? 'space-y-1' : controlsGridClassName}>
          <div className={controlPanelClassName}>
            <div className={embedded ? 'flex items-center gap-1.5' : ''}>
              <label className={embedded ? 'shrink-0 text-[9px] uppercase text-gray-500' : controlLabelClassName}>Display</label>
              <div className={embedded ? 'flex flex-wrap gap-1' : 'flex flex-wrap gap-2'}>
              <ToggleButton checked={showSmoothed} onClick={() => setShowSmoothed(v => !v)} label="Smoothed" compact={embedded} />
              <ToggleButton checked={showRaw} onClick={() => setShowRaw(v => !v)} label="Raw" compact={embedded} />
              <ToggleButton checked={useLogScale} onClick={() => setUseLogScale(v => !v)} label="Log Y" compact={embedded} />
              <ToggleButton checked={clipOutliers} onClick={() => setClipOutliers(v => !v)} label="Clip" compact={embedded} />
              </div>
            </div>
          </div>

          <div className={controlPanelClassName}>
            <div className={embedded ? 'flex items-start gap-1.5' : ''}>
              <label className={embedded ? 'shrink-0 pt-1 text-[9px] uppercase text-gray-500' : controlLabelClassName}>Series</label>
              {lossKeys.length === 0 ? (
                <div className={embedded ? 'text-[10px] text-gray-400' : 'text-sm text-gray-400'}>No loss keys found yet.</div>
              ) : (
                <div className={embedded ? 'flex max-h-10 flex-1 flex-wrap gap-1 overflow-y-auto pr-1' : 'flex flex-wrap gap-2'}>
                  {lossKeys.map(k => (
                    <button
                      key={k}
                      type="button"
                      onClick={() => setEnabled(prev => ({ ...prev, [k]: !(prev[k] ?? true) }))}
                      className={[
                        embedded
                          ? 'min-w-0 px-1.5 py-0.5 rounded text-[10px] leading-3 border transition-colors truncate'
                          : 'px-3 py-1 rounded-md text-xs border transition-colors',
                        enabled[k] === false
                          ? 'bg-gray-900 text-gray-400 border-gray-800 hover:bg-gray-800/60'
                          : 'bg-gray-900 text-gray-200 border-gray-800 hover:bg-gray-800/60',
                      ].join(' ')}
                      aria-pressed={enabled[k] !== false}
                      title={k}
                    >
                      <span className="inline-block h-1.5 w-1.5 rounded-full mr-1" style={{ background: strokeForKey(k) }} />
                      {k}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className={embedded ? 'grid grid-cols-2 gap-1' : controlsGridClassName}>
            <div className={controlPanelClassName}>
              <div className="flex items-center justify-between gap-2 mb-0.5">
                <label className={sliderLabelClassName}>Smoothing</label>
                <span className={embedded ? 'text-[10px] text-gray-300' : 'text-xs text-gray-300'}>{smoothing}%</span>
              </div>
              <input
                type="range"
                min={0}
                max={100}
                value={smoothing}
                onChange={e => setSmoothing(Number(e.target.value))}
                className="block h-3 w-full accent-blue-500"
                disabled={!showSmoothed}
              />
            </div>

            <div className={controlPanelClassName}>
              <div className="flex items-center justify-between gap-2 mb-0.5">
                <label className={sliderLabelClassName}>Stride</label>
                <span className={embedded ? 'text-[10px] text-gray-300' : 'text-xs text-gray-300'}>{plotStride} pt</span>
              </div>
              <input
                type="range"
                min={1}
                max={20}
                value={plotStride}
                onChange={e => setPlotStride(Number(e.target.value))}
                className="block h-3 w-full accent-blue-500"
              />
              {!embedded && <div className="mt-2 text-[11px] text-gray-500">UI downsample for huge runs.</div>}
            </div>
          </div>
        </div>
      </div>

      <style jsx global>{`
        .uplot,
        .uplot * {
          font-family: inherit;
        }
        .uplot .u-legend {
          color: rgba(255, 255, 255, 0.85);
          font-size: 12px;
          margin-top: 4px;
        }
        .uplot .u-legend th,
        .uplot .u-legend td {
          color: rgba(255, 255, 255, 0.85);
        }
        .uplot .u-legend .u-marker {
          border-radius: 2px;
        }
        .uplot .u-select {
          background: rgba(59, 130, 246, 0.15);
          border: 1px solid rgba(59, 130, 246, 0.4);
        }
      `}</style>
    </div>
  );
}

function ToggleButton({
  checked,
  onClick,
  label,
  compact = false,
}: {
  checked: boolean;
  onClick: () => void;
  label: string;
  compact?: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={[
        compact
          ? 'px-1.5 py-0.5 rounded text-[10px] leading-3 border transition-colors'
          : 'px-3 py-1 rounded-md text-xs border transition-colors',
        checked
          ? 'bg-blue-500/10 text-blue-300 border-blue-500/30 hover:bg-blue-500/15'
          : 'bg-gray-900 text-gray-300 border-gray-800 hover:bg-gray-800/60',
      ].join(' ')}
      aria-pressed={checked}
    >
      {label}
    </button>
  );
}
