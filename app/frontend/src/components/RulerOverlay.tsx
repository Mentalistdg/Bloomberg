import type { Measurement } from '../hooks/useChartRuler';

interface RulerOverlayProps {
  measurement: Measurement | null;
  /** Recharts left margin (to offset SVG coords) */
  marginLeft?: number;
  marginTop?: number;
}

export default function RulerOverlay({ measurement, marginLeft = 0, marginTop = 0 }: RulerOverlayProps) {
  if (!measurement) return null;

  const { a, b, pctChange } = measurement;
  const ax = a.cx + marginLeft;
  const ay = a.cy + marginTop;
  const bx = b.cx + marginLeft;
  const by = b.cy + marginTop;

  // Label position: midpoint of A-B, nudged up
  const labelX = (ax + bx) / 2;
  const labelY = Math.min(ay, by) - 12;

  const sign = pctChange >= 0 ? '+' : '';
  const color = pctChange >= 0 ? '#00c853' : '#CF2141';

  return (
    <svg
      className="absolute inset-0 pointer-events-none"
      style={{ width: '100%', height: '100%', overflow: 'visible' }}
    >
      {/* Dashed line A→B */}
      <line
        x1={ax} y1={ay} x2={bx} y2={by}
        stroke={color} strokeWidth={1.5} strokeDasharray="6 3"
      />
      {/* Point A marker */}
      <circle cx={ax} cy={ay} r={4} fill={color} stroke="#111" strokeWidth={1.5} />
      {/* Point B marker */}
      <circle cx={bx} cy={by} r={4} fill={color} stroke="#111" strokeWidth={1.5} />
      {/* Label background */}
      <rect
        x={labelX - 52} y={labelY - 14}
        width={104} height={20}
        rx={4}
        fill="#111" stroke="#333" strokeWidth={0.5}
      />
      {/* Label text */}
      <text
        x={labelX} y={labelY}
        textAnchor="middle"
        fill={color}
        fontSize={11}
        fontFamily="monospace"
        fontWeight="bold"
      >
        {sign}{pctChange.toFixed(1)}%
      </text>
    </svg>
  );
}
