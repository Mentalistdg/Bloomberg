import { useEffect, useState, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Legend, CartesianGrid, ReferenceLine, ReferenceArea,
} from 'recharts';
import { BacktestPoint, Meta, FoldBoundary, getBacktest, getMeta } from '../services/api';
import LoadingScreen from '../components/LoadingScreen';

const pct = (v: number | null | undefined) =>
  v == null ? '—' : `${(v * 100).toFixed(1)}%`;
const fmtSharpeSigned = (v: number | null | undefined) =>
  v == null ? '—' : `${v >= 0 ? '+' : ''}${v.toFixed(2)}`;

const TICK_STYLE = { fill: '#737373', fontSize: 10 } as const;
const AXIS_STYLE = { stroke: '#222' } as const;

function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  const point = payload[0]?.payload;
  const suffix = point?.is_production ? ' (producción)' : '';
  return (
    <div className="bg-[#111] border border-[#333] rounded-lg px-4 py-3 text-xs shadow-xl">
      <p className="text-[#aaa] mb-1.5 font-medium">{label}{suffix}</p>
      {payload.map((p: any) => (
        <p key={p.dataKey} style={{ color: p.stroke }} className="leading-relaxed">
          {p.name}: <span className="font-mono">{fmtSharpeSigned(p.value)}</span>
        </p>
      ))}
      {point && <p className="text-[#555] mt-1">{point.n_funds} fondos</p>}
    </div>
  );
}


export default function BacktestPage() {
  const [data, setData] = useState<BacktestPoint[]>([]);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([getBacktest(), getMeta()])
      .then(([bt, m]) => { setData(bt); setMeta(m); })
      .finally(() => setLoading(false));
  }, []);

  const foldBoundaries: FoldBoundary[] = useMemo(
    () => meta?.fold_boundaries ?? [], [meta]);

  // Production zone boundaries
  const productionStart = useMemo(
    () => data.find(d => d.is_production)?.date, [data]);

  const productionEnd = useMemo(() => {
    const prod = data.filter(d => d.is_production);
    return prod.length ? prod[prod.length - 1].date : undefined;
  }, [data]);


  if (loading) return <LoadingScreen message="Cargando backtest..." />;

  const elMetrics = meta?.metrics_summary?.elastic;

  /* Shared chart elements */
  const renderFoldLines = (boundaries: FoldBoundary[]) =>
    boundaries.map(fb => (
      <ReferenceLine
        key={`fold-${fb.fold}`}
        x={fb.val_start}
        stroke="#444"
        strokeDasharray="3 3"
        label={{ value: `F${fb.fold}`, position: 'top', fill: '#555', fontSize: 9 }}
      />
    ));

  const renderProductionArea = () =>
    productionStart && productionEnd ? (
      <ReferenceArea
        x1={productionStart}
        x2={productionEnd}
        fill="#CF2141"
        fillOpacity={0.07}
        label={{ value: 'Producción (OOS)', position: 'insideTopRight', fill: '#CF2141', fontSize: 10 }}
      />
    ) : null;

  return (
    <div>
      {/* ── Section A: Header ── */}
      <h1 className="text-xl font-bold text-text mb-1">
        Backtest: separaci&oacute;n D10 vs D1 (ElasticNet)
      </h1>
      <p className="text-sm text-muted mb-3">
        Comparaci&oacute;n del <strong>Sortino</strong> realizado a 6m entre el decil superior (D10, top 10%)
        e inferior (D1, bottom 10%) seg&uacute;n score ElasticNet.
      </p>
      <div className="text-xs text-muted bg-[#0a0a0a] border border-[#1a1a1a] rounded-lg px-4 py-2.5 mb-6 max-w-2xl">
        Datos out-of-sample (walk-forward). Periodo posterior al &uacute;ltimo fold = scoring de producci&oacute;n
        (modelo entrenado con datos hasta {foldBoundaries.length > 0
          ? foldBoundaries[foldBoundaries.length - 1].val_end
          : '2024-12'}).
      </div>

      {/* ── Explanation: D10 vs D1 ── */}
      <div className="bg-[#0a0a0a] border border-[#1a1a1a] rounded-lg px-4 py-3 mb-6 max-w-2xl">
        <div className="flex items-center gap-2 mb-1.5">
          <span className="w-3 h-0.5 bg-[#CF2141] inline-block rounded" />
          <span className="text-sm font-semibold text-text">ElasticNet &mdash; D10 vs D1</span>
        </div>
        <p className="text-xs text-muted leading-relaxed">
          <strong className="text-[#aaa]">D10</strong> = fondos en el decil superior (top 10%) por score.{' '}
          <strong className="text-[#aaa]">D1</strong> = decil inferior (bottom 10%).
          El spread mide la capacidad del modelo de separar fondos de alta calidad
          ajustada por riesgo de los de baja calidad. Un spread consistentemente positivo
          indica que el scoring aporta informaci&oacute;n predictiva sobre el <strong>Sortino</strong> futuro a 6 meses.
        </p>
      </div>

      {/* ── Section B: Returns by decile ── */}
      <h2 className="text-base font-semibold text-text mb-2">Sortino rolling 6m por decil (ElasticNet)</h2>
      <div className="card p-5 mb-8" style={{ height: 380 }}>
        <ResponsiveContainer>
          <LineChart data={data} margin={{ top: 20, right: 20, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
            <XAxis dataKey="date" tick={TICK_STYLE} axisLine={AXIS_STYLE} tickLine={AXIS_STYLE}
                   tickFormatter={(d: string) => d.slice(5, 7) === '01' ? d.slice(0, 4) : ''}
                   interval={Math.max(0, Math.floor(data.length / 12))} />
            <YAxis tick={TICK_STYLE} axisLine={AXIS_STYLE} tickLine={AXIS_STYLE}
                   tickFormatter={(v: number) => v.toFixed(1)} />
            {renderProductionArea()}
            {renderFoldLines(foldBoundaries)}
            <ReferenceLine y={0} stroke="#555" strokeDasharray="2 2" />
            <Tooltip content={<ChartTooltip />} />
            <Legend wrapperStyle={{ color: '#737373', paddingTop: 8 }} />
            <Line type="monotone" dataKey="top_d10" stroke="#00c853" name="Top D10"
                  dot={false} strokeWidth={2} connectNulls />
            <Line type="monotone" dataKey="bot_d1" stroke="#CF2141" name="Bottom D1"
                  dot={false} strokeWidth={2} connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* ── Section D: Metrics table ── */}
      {meta && elMetrics && (
        <div className="card p-5 mb-8 overflow-x-auto">
          <h2 className="text-base font-semibold text-text mb-4">M&eacute;tricas del modelo (OOS)</h2>
          <table className="w-full text-sm max-w-lg">
            <thead>
              <tr className="border-b border-[#222]">
                <th className="text-left text-muted font-medium py-2 pr-6">M&eacute;trica</th>
                <th className="text-right text-muted font-medium py-2 px-4">ElasticNet</th>
              </tr>
            </thead>
            <tbody className="font-mono text-text">
              <tr className="border-b border-[#151515]">
                <td className="py-2 pr-6 text-muted font-sans">IC medio</td>
                <td className="text-right px-4">{elMetrics.ic_mean.toFixed(3)}</td>
              </tr>
              <tr className="border-b border-[#151515]">
                <td className="py-2 pr-6 text-muted font-sans">ICIR (IC medio / std)</td>
                <td className="text-right px-4">{elMetrics.ic_ir.toFixed(2)}</td>
              </tr>
              <tr className="border-b border-[#151515]">
                <td className="py-2 pr-6 text-muted font-sans">IC hit (% meses &gt; 0)</td>
                <td className="text-right px-4">{pct(elMetrics.ic_hit_meses)}</td>
              </tr>
              <tr className="border-b border-[#151515]">
                <td className="py-2 pr-6 text-muted font-sans">IC 95% CI</td>
                <td className="text-right px-4">
                  [{elMetrics.ic_ci95_low.toFixed(3)}, {elMetrics.ic_ci95_high.toFixed(3)}]
                </td>
              </tr>
              <tr>
                <td className="py-2 pr-6 text-muted font-sans">Hit top-25%</td>
                <td className="text-right px-4">{pct(elMetrics.hit_top25)}</td>
              </tr>
            </tbody>
          </table>
        </div>
      )}

      {/* ── Section E: Footnotes ── */}
      <div className="text-xs text-muted space-y-1.5 mt-4 mb-8 max-w-3xl leading-relaxed">
        <p><strong className="text-[#888]">Deciles:</strong> D10 = top 10% por score, D1 = bottom 10%.</p>
        <p><strong className="text-[#888]">Producci&oacute;n:</strong> Zona sombreada roja = modelo reentrenado con datos hasta {foldBoundaries.length > 0
          ? foldBoundaries[foldBoundaries.length - 1].val_end
          : '2024-12'},
          aplicado al periodo posterior. No es walk-forward; es scoring &ldquo;en vivo&rdquo;.</p>
        <p><strong className="text-[#888]">Sortino:</strong> Sortino anualizado = (media &minus; Rf) / downside_dev &times; &radic;12, con Rf = 2% anual constante.
          downside_dev = &radic;(mean(min(r,0)&sup2;)). Los meses m&aacute;s recientes usan Sortino forward estimado con los meses disponibles (rolling).</p>
        <p><strong className="text-[#888]">Walk-forward:</strong> L&iacute;neas verticales = inicio de cada fold de validaci&oacute;n.</p>
      </div>
    </div>
  );
}
