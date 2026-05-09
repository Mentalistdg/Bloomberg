import { useEffect, useState, useMemo } from 'react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Legend, CartesianGrid, ReferenceLine, ReferenceArea,
} from 'recharts';
import { BacktestPoint, Meta, FoldBoundary, getBacktest, getMeta } from '../services/api';
import LoadingScreen from '../components/LoadingScreen';

const pct = (v: number | null | undefined) =>
  v == null ? '—' : `${(v * 100).toFixed(1)}%`;
const pctSigned = (v: number | null | undefined) =>
  v == null ? '—' : `${v >= 0 ? '+' : ''}${(v * 100).toFixed(1)}%`;

const TICK_STYLE = { fill: '#737373', fontSize: 10 } as const;
const AXIS_STYLE = { stroke: '#222' } as const;

/* Custom tooltip shared by both charts */
function ChartTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  // Filter out null-valued series (the split full/partial duplicates)
  const visible = payload.filter((p: any) => p.value != null);
  if (!visible.length) return null;
  const point: BacktestPoint | undefined = visible[0]?.payload;
  const suffix = point?.is_partial
    ? ` (parcial: ${point.months_forward}m)`
    : point?.is_production ? ' (producción)' : '';
  return (
    <div className="bg-[#111] border border-[#333] rounded-lg px-4 py-3 text-xs shadow-xl">
      <p className="text-[#aaa] mb-1.5 font-medium">{label}{suffix}</p>
      {visible.map((p: any) => (
        <p key={p.dataKey} style={{ color: p.stroke }} className="leading-relaxed">
          {p.name}: <span className="font-mono">{pctSigned(p.value)}</span>
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

  // Prepare split data: full (solid lines) vs partial (dashed lines)
  // To connect visually, the last full point is duplicated into the partial series
  const chartData = useMemo(() => {
    const lastFullIdx = data.reduce((acc, d, i) => (!d.is_partial ? i : acc), -1);
    return data.map((d, i) => ({
      ...d,
      // Full series: show values for non-partial points + transition overlap
      top_d10_full: !d.is_partial ? d.top_d10 : null,
      bot_d1_full: !d.is_partial ? d.bot_d1 : null,
      universe_full: !d.is_partial ? d.universe : null,
      spread_full: !d.is_partial ? d.spread : null,
      naive_spread_full: !d.is_partial ? d.naive_spread : null,
      // Partial series: show values for partial points + overlap at transition
      top_d10_partial: d.is_partial || i === lastFullIdx ? d.top_d10 : null,
      bot_d1_partial: d.is_partial || i === lastFullIdx ? d.bot_d1 : null,
      universe_partial: d.is_partial || i === lastFullIdx ? d.universe : null,
      spread_partial: d.is_partial || i === lastFullIdx ? d.spread : null,
      naive_spread_partial: d.is_partial || i === lastFullIdx ? d.naive_spread : null,
    }));
  }, [data]);

  // Mean spreads for summary
  const meanSpread = useMemo(() => {
    const valid = data.filter(d => d.spread != null && !d.is_production);
    return valid.length ? valid.reduce((s, d) => s + d.spread!, 0) / valid.length : null;
  }, [data]);
  const meanNaiveSpread = useMemo(() => {
    const valid = data.filter(d => d.naive_spread != null && !d.is_production);
    return valid.length ? valid.reduce((s, d) => s + d.naive_spread!, 0) / valid.length : null;
  }, [data]);

  if (loading) return <LoadingScreen message="Cargando backtest..." />;

  const elMetrics = meta?.metrics_summary?.elastic;
  const naiveMetrics = meta?.metrics_summary?.benchmark;
  const dm = meta?.diebold_mariano_elastic_vs_benchmark;

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
        fillOpacity={0.06}
        label={{ value: 'Producción', position: 'insideTopRight', fill: '#CF2141', fontSize: 11 }}
      />
    ) : null;

  return (
    <div>
      {/* ── Section A: Header ── */}
      <h1 className="text-xl font-bold text-text mb-1">
        Backtest: ElasticNet vs Benchmark naive
      </h1>
      <p className="text-sm text-muted mb-3">
        Spread D10&minus;D1: diferencia de retorno realizado a 12m entre el decil superior e inferior
        por score. Media OOS: ElasticNet {pctSigned(meanSpread)}, naive {pctSigned(meanNaiveSpread)}.
      </p>
      <div className="text-xs text-muted bg-[#0a0a0a] border border-[#1a1a1a] rounded-lg px-4 py-2.5 mb-6 max-w-2xl">
        Datos out-of-sample (walk-forward). Periodo 2025+ = scoring de producci&oacute;n
        (modelo entrenado con datos hasta 2024-12).
      </div>

      {/* ── Section B: Spread comparison chart ── */}
      <h2 className="text-base font-semibold text-text mb-2">Spread D10&minus;D1 comparativo</h2>
      <div className="card p-5 mb-8" style={{ height: 380 }}>
        <ResponsiveContainer>
          <LineChart data={chartData} margin={{ top: 20, right: 20, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
            <XAxis dataKey="date" tick={TICK_STYLE} axisLine={AXIS_STYLE} tickLine={AXIS_STYLE}
                   interval={Math.max(0, Math.floor(chartData.length / 8))} />
            <YAxis tick={TICK_STYLE} axisLine={AXIS_STYLE} tickLine={AXIS_STYLE}
                   tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
            {renderProductionArea()}
            {renderFoldLines(foldBoundaries)}
            <ReferenceLine y={0} stroke="#555" strokeDasharray="2 2" />
            <Tooltip content={<ChartTooltip />} />
            <Legend wrapperStyle={{ color: '#737373', paddingTop: 8 }} />
            <Line type="monotone" dataKey="spread_full" stroke="#CF2141" name="ElasticNet spread"
                  dot={false} strokeWidth={2} connectNulls />
            <Line type="monotone" dataKey="spread_partial" stroke="#CF2141" name="ElasticNet spread (parcial)"
                  dot={false} strokeWidth={1.5} strokeDasharray="4 4" connectNulls />
            <Line type="monotone" dataKey="naive_spread_full" stroke="#4fc3f7" name="Naive spread"
                  dot={false} strokeWidth={1.5} strokeDasharray="4 4" connectNulls />
            <Line type="monotone" dataKey="naive_spread_partial" stroke="#4fc3f7" name="Naive spread (parcial)"
                  dot={false} strokeWidth={1} strokeDasharray="2 2" connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* ── Section C: Returns by decile ── */}
      <h2 className="text-base font-semibold text-text mb-2">Retornos por decil (ElasticNet)</h2>
      <div className="card p-5 mb-8" style={{ height: 380 }}>
        <ResponsiveContainer>
          <LineChart data={chartData} margin={{ top: 20, right: 20, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
            <XAxis dataKey="date" tick={TICK_STYLE} axisLine={AXIS_STYLE} tickLine={AXIS_STYLE}
                   interval={Math.max(0, Math.floor(chartData.length / 8))} />
            <YAxis tick={TICK_STYLE} axisLine={AXIS_STYLE} tickLine={AXIS_STYLE}
                   tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`} />
            {renderProductionArea()}
            {renderFoldLines(foldBoundaries)}
            <ReferenceLine y={0} stroke="#555" strokeDasharray="2 2" />
            <Tooltip content={<ChartTooltip />} />
            <Legend wrapperStyle={{ color: '#737373', paddingTop: 8 }} />
            {/* Full (solid) lines */}
            <Line type="monotone" dataKey="top_d10_full" stroke="#00c853" name="Top D10"
                  dot={false} strokeWidth={2} connectNulls />
            <Line type="monotone" dataKey="universe_full" stroke="#525252" name="Universo"
                  dot={false} strokeWidth={1.5} strokeDasharray="4 4" connectNulls />
            <Line type="monotone" dataKey="bot_d1_full" stroke="#CF2141" name="Bot D1"
                  dot={false} strokeWidth={2} connectNulls />
            {/* Partial (dashed) lines */}
            <Line type="monotone" dataKey="top_d10_partial" stroke="#00c853" name="Top D10 (parcial)"
                  dot={false} strokeWidth={1.5} strokeDasharray="4 4" connectNulls />
            <Line type="monotone" dataKey="universe_partial" stroke="#525252" name="Universo (parcial)"
                  dot={false} strokeWidth={1} strokeDasharray="2 2" connectNulls />
            <Line type="monotone" dataKey="bot_d1_partial" stroke="#CF2141" name="Bot D1 (parcial)"
                  dot={false} strokeWidth={1.5} strokeDasharray="4 4" connectNulls />
          </LineChart>
        </ResponsiveContainer>
      </div>

      {/* ── Section D: Metrics comparison table ── */}
      {meta && elMetrics && naiveMetrics && (
        <div className="card p-5 mb-8 overflow-x-auto">
          <h2 className="text-base font-semibold text-text mb-4">M&eacute;tricas comparativas (OOS)</h2>
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#222]">
                <th className="text-left text-muted font-medium py-2 pr-6">M&eacute;trica</th>
                <th className="text-right text-muted font-medium py-2 px-4">ElasticNet</th>
                <th className="text-right text-muted font-medium py-2 px-4">Naive</th>
              </tr>
            </thead>
            <tbody className="font-mono text-text">
              <tr className="border-b border-[#151515]">
                <td className="py-2 pr-6 text-muted font-sans">IC medio</td>
                <td className="text-right px-4">{elMetrics.ic_mean.toFixed(3)}</td>
                <td className="text-right px-4">{naiveMetrics.ic_mean.toFixed(3)}</td>
              </tr>
              <tr className="border-b border-[#151515]">
                <td className="py-2 pr-6 text-muted font-sans">IC Information Ratio</td>
                <td className="text-right px-4">{elMetrics.ic_ir.toFixed(2)}</td>
                <td className="text-right px-4">{naiveMetrics.ic_ir.toFixed(2)}</td>
              </tr>
              <tr className="border-b border-[#151515]">
                <td className="py-2 pr-6 text-muted font-sans">IC hit (% meses &gt; 0)</td>
                <td className="text-right px-4">{pct(elMetrics.ic_hit_meses)}</td>
                <td className="text-right px-4">{pct(naiveMetrics.ic_hit_meses)}</td>
              </tr>
              <tr className="border-b border-[#151515]">
                <td className="py-2 pr-6 text-muted font-sans">IC 95% CI</td>
                <td className="text-right px-4">
                  [{elMetrics.ic_ci95_low.toFixed(3)}, {elMetrics.ic_ci95_high.toFixed(3)}]
                </td>
                <td className="text-right px-4">
                  [{naiveMetrics.ic_ci95_low.toFixed(3)}, {naiveMetrics.ic_ci95_high.toFixed(3)}]
                </td>
              </tr>
              <tr className="border-b border-[#151515]">
                <td className="py-2 pr-6 text-muted font-sans">Hit top-25%</td>
                <td className="text-right px-4">{pct(elMetrics.hit_top25)}</td>
                <td className="text-right px-4">{pct(naiveMetrics.hit_top25)}</td>
              </tr>
              {dm && (
                <tr>
                  <td className="py-2 pr-6 text-muted font-sans">D-M p-value</td>
                  <td className="text-right px-4">&mdash;</td>
                  <td className="text-right px-4">{dm.p_value.toFixed(3)}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {/* ── Section E: Footnotes ── */}
      <div className="text-xs text-muted space-y-1.5 mt-4 mb-8 max-w-3xl leading-relaxed">
        <p><strong className="text-[#888]">Deciles:</strong> D10 = top 10% por score, D1 = bottom 10%.</p>
        <p><strong className="text-[#888]">Producci&oacute;n:</strong> Zona sombreada = modelo reentrenado con datos hasta 2024-12,
          aplicado a 2025+. No es walk-forward; es scoring &ldquo;en vivo&rdquo;.</p>
        <p><strong className="text-[#888]">Datos parciales:</strong> L&iacute;neas punteadas = retorno forward con los meses disponibles (&lt; 12m).
          Magnitudes no directamente comparables con retornos 12m completos.</p>
        <p><strong className="text-[#888]">Naive:</strong> score = ret_12m_rank &minus; fee_rank. Sin entrenamiento.</p>
        <p><strong className="text-[#888]">Walk-forward:</strong> L&iacute;neas verticales = inicio de cada fold de validaci&oacute;n.</p>
      </div>
    </div>
  );
}
