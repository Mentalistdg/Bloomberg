import { useState, useCallback, useRef } from 'react';

type RulerState = 'idle' | 'placing_a' | 'placing_b' | 'locked';

interface RulerPoint {
  date: string;
  value: number;
  /** pixel X relative to chart container */
  cx: number;
  /** pixel Y relative to chart container */
  cy: number;
}

export interface Measurement {
  a: RulerPoint;
  b: RulerPoint;
  pctChange: number;
  absChange: number;
}

export interface ChartRuler {
  state: RulerState;
  isActive: boolean;
  pointA: RulerPoint | null;
  pointB: RulerPoint | null;
  measurement: Measurement | null;
  activate: () => void;
  deactivate: () => void;
  handleChartClick: (e: any) => void;
  handleChartMouseMove: (e: any) => void;
  containerRef: React.RefObject<HTMLDivElement>;
}

function extractPoint(e: any, containerRef: React.RefObject<HTMLDivElement | null>): RulerPoint | null {
  if (!e?.activePayload?.length) return null;
  const payload = e.activePayload[0].payload;
  const equity = payload.equity ?? payload.equity_trailing ?? payload.equity_forward ?? payload.optimal;
  if (!payload?.date || equity == null) return null;

  // Get pixel coords relative to container
  const container = containerRef.current;
  if (!container) return null;

  // chartX/chartY are relative to the chart area
  // e.chartX and e.chartY come from Recharts
  const cx = e.chartX ?? (e.activeCoordinate?.x ?? 0);
  const cy = e.chartY ?? (e.activeCoordinate?.y ?? 0);

  // Add Recharts margin offset (default left=5, top=5 for ResponsiveContainer)
  // These are approximate — Recharts doesn't expose exact offsets easily
  return {
    date: payload.date,
    value: equity,
    cx,
    cy,
  };
}

export function useChartRuler(): ChartRuler {
  const [state, setState] = useState<RulerState>('idle');
  const [pointA, setPointA] = useState<RulerPoint | null>(null);
  const [pointB, setPointB] = useState<RulerPoint | null>(null);
  const containerRef = useRef<HTMLDivElement>(null!);

  const activate = useCallback(() => {
    setState('placing_a');
    setPointA(null);
    setPointB(null);
  }, []);

  const deactivate = useCallback(() => {
    setState('idle');
    setPointA(null);
    setPointB(null);
  }, []);

  const handleChartClick = useCallback((e: any) => {
    if (state === 'idle') return;

    if (state === 'locked') {
      // Reset: start placing again
      setState('placing_a');
      setPointA(null);
      setPointB(null);
      return;
    }

    const point = extractPoint(e, containerRef);
    if (!point) return;

    if (state === 'placing_a') {
      setPointA(point);
      setPointB(null);
      setState('placing_b');
    } else if (state === 'placing_b') {
      setPointB(point);
      setState('locked');
    }
  }, [state]);

  const handleChartMouseMove = useCallback((e: any) => {
    if (state !== 'placing_b') return;
    const point = extractPoint(e, containerRef);
    if (point) setPointB(point);
  }, [state]);

  const measurement: Measurement | null =
    pointA && pointB && pointA.value !== 0
      ? {
          a: pointA,
          b: pointB,
          pctChange: ((pointB.value - pointA.value) / pointA.value) * 100,
          absChange: pointB.value - pointA.value,
        }
      : null;

  return {
    state,
    isActive: state !== 'idle',
    pointA,
    pointB,
    measurement,
    activate,
    deactivate,
    handleChartClick,
    handleChartMouseMove,
    containerRef,
  };
}
