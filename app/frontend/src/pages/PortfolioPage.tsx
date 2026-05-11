import { useEffect, useState, useMemo } from 'react';
import { Link } from 'react-router-dom';
import { Ruler } from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer,
  Legend, CartesianGrid, PieChart, Pie, Cell, ReferenceLine,
} from 'recharts';
import { PortfolioData, RebalancePeriod, getPortfolio } from '../services/api';
import { useChartRuler } from '../hooks/useChartRuler';
import RulerOverlay from '../components/RulerOverlay';
import LoadingScreen from '../components/LoadingScreen';

const pct = (v: number | null | undefined) =>
  v == null ? '\u2014' : `${(v * 100).toFixed(1)}%`;
const fmtFee = (v: number | null | undefined) =>
  v == null ? '\u2014' : `${v.toFixed(2)}%`;
const fmtNum = (v: number | null | undefined, decimals = 2) =>
  v == null ? '\u2014' : v.toFixed(decimals);

const TICK_STYLE = { fill: '#737373', fontSize: 10 } as const;
const AXIS_STYLE = { stroke: '#222' } as const;

const COLORS = [
  '#CF2141', '#4fc3f7', '#66bb6a', '#ffa726', '#ab47bc',
  '#26c6da', '#ef5350', '#7e57c2', '#42a5f5', '#ec407a',
  '#8d6e63', '#78909c', '#d4e157', '#ffca28', '#29b6f6',
];

function EquityTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null;
  return (
    <div className="bg-[#111] border border-[#333] rounded-lg px-4 py-3 text-xs shadow-xl">
      <p className="text-[#aaa] mb-1.5 font-medium">{label}</p>
      {payload.map((p: any) => (
        <p key={p.dataKey} style={{ color: p.stroke || p.fill }} className="leading-relaxed">
          {p.name}: <span className="font-mono">{fmtNum(p.value, 1)}</span>
        </p>
      ))}
    </div>
  );
}

function HoldingsTable({ holdings, maxContrib }: { holdings: { fondo: string; weight: number; period_return: number; contribution: number }[]; maxContrib: number }) {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="border-b border-[#222]">
          <th className="text-left text-muted font-medium py-1 pr-4">Fondo</th>
          <th className="text-right text-muted font-medium py-1 px-3">Peso</th>
          <th className="text-right text-muted font-medium py-1 px-3">Retorno</th>
          <th className="text-right text-muted font-medium py-1 px-3">Contrib.</th>
          <th className="text-left text-muted font-medium py-1 pl-4 w-32"></th>
        </tr>
      </thead>
      <tbody className="font-mono text-text">
        {holdings.map((h) => {
          const barWidth = maxContrib > 0 ? (Math.abs(h.contribution) / maxContrib) * 100 : 0;
          const barColor = h.contribution >= 0 ? '#66bb6a' : '#ef5350';
          return (
            <tr key={h.fondo} className="border-b border-[#111]">
              <td className="py-1 pr-4 font-sans">
                <Link to={`/detail/${h.fondo}`} className="text-accent hover:underline">{h.fondo}</Link>
              </td>
              <td className="text-right px-3">{pct(h.weight)}</td>
              <td className={`text-right px-3 ${h.period_return < 0 ? 'text-red-400' : ''}`}>
                {pct(h.period_return)}
              </td>
              <td className={`text-right px-3 ${h.contribution < 0 ? 'text-red-400' : ''}`}>
                {pct(h.contribution)}
              </td>
              <td className="pl-4 py-1">
                <div className="h-2 bg-[#1a1a1a] rounded-full overflow-hidden">
                  <div
                    className="h-full rounded-full"
                    style={{ width: `${barWidth}%`, backgroundColor: barColor }}
                  />
                </div>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

const OPT_METHOD_LABELS: Record<string, string> = {
  max_sharpe_rf0: 'RF=0',
  max_quadratic_utility: 'Quad. Utility',
  equal_weight: 'EW fallback',
};

function RebalanceRow({ period, isLast }: { period: RebalancePeriod; isLast: boolean }) {
  const [expanded, setExpanded] = useState(false);
  const nFunds = period.holdings.length;
  const allHoldings = [...period.holdings, ...(period.ew_holdings ?? [])];
  const maxContrib = Math.max(...allHoldings.map(h => Math.abs(h.contribution)), 0);
  const isFallback = period.optimization_method && period.optimization_method !== 'max_sharpe';

  return (
    <>
      <tr
        className={`cursor-pointer hover:bg-[#111] transition-colors ${!isLast ? 'border-b border-[#151515]' : ''}`}
        onClick={() => setExpanded(!expanded)}
      >
        <td className="py-2 pr-4 text-xs">
          <span className="text-muted mr-2">{expanded ? '\u25BC' : '\u25B6'}</span>
          {period.rebal_date}
          {isFallback && (
            <span className="ml-2 px-1.5 py-0.5 rounded text-[10px] bg-yellow-900/40 text-yellow-400 border border-yellow-800/50"
                  title={`Optimización: ${period.optimization_method}`}>
              {OPT_METHOD_LABELS[period.optimization_method!] ?? period.optimization_method}
            </span>
          )}
        </td>
        <td className="text-right px-3 font-mono text-xs">{period.n_months}m</td>
        <td className="text-right px-3 font-mono text-xs">{nFunds}</td>
        <td className={`text-right px-3 font-mono text-xs ${period.portfolio_return < 0 ? 'text-red-400' : ''}`}>
          {pct(period.portfolio_return)}
          {period.ew_portfolio_return != null && (
            <span className="text-muted ml-1.5">(EW: {pct(period.ew_portfolio_return)})</span>
          )}
        </td>
        <td className="text-right px-3 text-xs text-muted">{period.period_end}</td>
      </tr>
      {expanded && (
        <tr className="border-b border-[#1a1a1a]">
          <td colSpan={5} className="p-0">
            <div className="bg-[#0a0a0a] px-6 py-3">
              <p className="text-xs text-muted mb-2 font-semibold">&Oacute;ptimo ({nFunds} fondos)</p>
              <HoldingsTable holdings={period.holdings} maxContrib={maxContrib} />
              {period.ew_holdings && period.ew_holdings.length > 0 && (
                <>
                  <p className="text-xs text-muted mt-4 mb-2 font-semibold">
                    Equal-Weight Benchmark ({period.ew_holdings.length} fondos)
                  </p>
                  <HoldingsTable holdings={period.ew_holdings} maxContrib={maxContrib} />
                </>
              )}
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export default function PortfolioPage() {
  const [data, setData] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(true);
  const ruler = useChartRuler();

  useEffect(() => {
    getPortfolio()
      .then(setData)
      .finally(() => setLoading(false));
  }, []);

  const equityData = useMemo(() => {
    if (!data) return [];
    return data.backtest.dates.map((d, i) => ({
      date: d,
      optimal: data.backtest.opt_equity[i],
      equal_weight: data.backtest.ew_equity[i],
    }));
  }, [data]);

  const donutData = useMemo(() => {
    if (!data) return [];
    return data.current_weights.map(w => ({
      name: w.fondo,
      value: w.weight,
    }));
  }, [data]);

  const rebalanceDates = useMemo(() => {
    if (!data?.backtest.rebalance_dates) return [];
    // Solo incluir fechas que existan en el eje X
    const dateSet = new Set(data.backtest.dates);
    return data.backtest.rebalance_dates.filter(d => dateSet.has(d));
  }, [data]);

  const rebalanceHistory = useMemo(() => {
    if (!data?.rebalance_history) return [];
    // Orden cronologico inverso (mas reciente arriba)
    return [...data.rebalance_history].reverse();
  }, [data]);

  if (loading) return <LoadingScreen message="Cargando portafolio..." />;
  if (!data) return <p className="text-muted">No hay datos de portafolio disponibles.</p>;

  const { metrics, config } = data;

  return (
    <div>
      {/* Header */}
      <h1 className="text-xl font-bold text-text mb-1">
        Portafolio &Oacute;ptimo &mdash; Decil 10
      </h1>
      <p className="text-sm text-muted mb-3">
        Portafolio optimizado con los fondos del D10 (top 10% por score ElasticNet).
        Retornos EMA, semicovarianza como modelo de riesgo, rebalanceo anual, max peso {pct(config.max_weight)} por fondo.
      </p>
      <div className="text-xs text-muted bg-[#0a0a0a] border border-[#1a1a1a] rounded-lg px-4 py-2.5 mb-6 max-w-3xl">
        <strong className="text-[#888]">Anti-leakage:</strong> Cada mes usa scores OOS (walk-forward) para seleccionar D10,
        y covarianza trailing {config.lookback_months}m (backward-looking) para optimizar pesos.
        El retorno mostrado es el realizado en el mes siguiente.
      </div>

      {/* Equity curves */}
      <div className="flex items-baseline justify-between mb-2">
        <h2 className="text-base font-semibold text-text">Equity curves (base 100)</h2>
        <button
          onClick={() => ruler.isActive ? ruler.deactivate() : ruler.activate()}
          className={`flex items-center gap-1.5 text-xs px-2.5 py-1 rounded border transition-all ${
            ruler.isActive
              ? 'border-accent text-accent bg-accent/10'
              : 'border-[#333] text-muted hover:text-text hover:border-[#555]'
          }`}
        >
          <Ruler size={13} />
          Medir
        </button>
      </div>
      <p className="text-xs text-muted mb-2">
        <span className="text-[#CF2141]">Rojo</span> = portafolio &oacute;ptimo (semicov + max Sharpe, EMA returns),&nbsp;
        <span className="text-[#4fc3f7]">azul</span> = equal-weight D10.
        L&iacute;neas verticales punteadas indican fechas de rebalanceo.
      </p>
      <div className="card p-5 mb-8">
        <div className="relative" ref={ruler.containerRef}
             style={{ height: 340, ...(ruler.isActive ? { cursor: 'crosshair' } : {}) }}>
          <ResponsiveContainer>
            <LineChart data={equityData} margin={{ top: 10, right: 20, bottom: 5, left: 10 }}
                       onClick={ruler.handleChartClick} onMouseMove={ruler.handleChartMouseMove}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
              <XAxis dataKey="date" tick={TICK_STYLE} axisLine={AXIS_STYLE} tickLine={AXIS_STYLE}
                     tickFormatter={(d: string) => d.slice(5, 7) === '01' ? d.slice(0, 4) : ''}
                     interval={Math.max(0, Math.floor(equityData.length / 12))} />
              <YAxis tick={TICK_STYLE} axisLine={AXIS_STYLE} tickLine={AXIS_STYLE}
                     domain={['auto', 'auto']} />
              <Tooltip content={<EquityTooltip />} />
              <Legend wrapperStyle={{ color: '#737373', paddingTop: 8 }} />
              {rebalanceDates.map((rd) => (
                <ReferenceLine key={rd} x={rd} stroke="#444" strokeDasharray="3 3" />
              ))}
              <Line type="monotone" dataKey="optimal" stroke="#CF2141" name="Portafolio \u00d3ptimo"
                    dot={false} strokeWidth={2} connectNulls />
              <Line type="monotone" dataKey="equal_weight" stroke="#4fc3f7" name="Equal-Weight D10"
                    dot={false} strokeWidth={1.5} strokeDasharray="4 4" connectNulls />
            </LineChart>
          </ResponsiveContainer>
          <RulerOverlay measurement={ruler.measurement} />
        </div>
        {ruler.isActive && (
          <p className="text-xs text-muted mt-2">
            {ruler.state === 'placing_a' && 'Haz clic en el punto inicial de la medici\u00f3n.'}
            {ruler.state === 'placing_b' && 'Haz clic en el punto final.'}
            {ruler.state === 'locked' && ruler.measurement && (
              <>
                <span className={ruler.measurement.pctChange >= 0 ? 'text-[#00c853]' : 'text-[#CF2141]'}>
                  {ruler.measurement.pctChange >= 0 ? '+' : ''}{ruler.measurement.pctChange.toFixed(1)}%
                </span>
                {' '}({ruler.measurement.a.date} → {ruler.measurement.b.date})
                {' '}<span className="text-[#555]">&middot; Clic para reiniciar</span>
              </>
            )}
          </p>
        )}
      </div>

      {/* Metrics comparison table */}
      <h2 className="text-base font-semibold text-text mb-3">M&eacute;tricas comparativas</h2>
      <div className="card p-5 mb-8 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[#222]">
              <th className="text-left text-muted font-medium py-2 pr-6">M&eacute;trica</th>
              <th className="text-right text-muted font-medium py-2 px-4">&Oacute;ptimo</th>
              <th className="text-right text-muted font-medium py-2 px-4">Equal-Weight</th>
            </tr>
          </thead>
          <tbody className="font-mono text-text">
            <tr className="border-b border-[#151515]">
              <td className="py-2 pr-6 text-muted font-sans">Retorno anualizado</td>
              <td className="text-right px-4">{pct(metrics.optimal.annual_return)}</td>
              <td className="text-right px-4">{pct(metrics.equal_weight.annual_return)}</td>
            </tr>
            <tr className="border-b border-[#151515]">
              <td className="py-2 pr-6 text-muted font-sans">Volatilidad anualizada</td>
              <td className="text-right px-4">{pct(metrics.optimal.annual_vol)}</td>
              <td className="text-right px-4">{pct(metrics.equal_weight.annual_vol)}</td>
            </tr>
            <tr className="border-b border-[#151515]">
              <td className="py-2 pr-6 text-muted font-sans">Sortino ratio</td>
              <td className="text-right px-4">{fmtNum(metrics.optimal.sortino)}</td>
              <td className="text-right px-4">{fmtNum(metrics.equal_weight.sortino)}</td>
            </tr>
            <tr className="border-b border-[#151515]">
              <td className="py-2 pr-6 text-muted font-sans">Max drawdown</td>
              <td className="text-right px-4">{pct(metrics.optimal.max_drawdown)}</td>
              <td className="text-right px-4">{pct(metrics.equal_weight.max_drawdown)}</td>
            </tr>
            <tr>
              <td className="py-2 pr-6 text-muted font-sans">Hit rate (% meses &gt; 0)</td>
              <td className="text-right px-4">{pct(metrics.optimal.hit_rate)}</td>
              <td className="text-right px-4">{pct(metrics.equal_weight.hit_rate)}</td>
            </tr>
          </tbody>
        </table>
      </div>

      {/* Current weights + donut */}
      <h2 className="text-base font-semibold text-text mb-3">
        Pesos actuales {data.last_rebalance_date && (
          <span className="text-muted font-normal text-sm">&mdash; rebalanceo {data.last_rebalance_date}</span>
        )}
      </h2>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-8">
        {/* Table */}
        <div className="lg:col-span-2 card p-5 overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#222]">
                <th className="text-left text-muted font-medium py-2 pr-4">Fondo</th>
                <th className="text-right text-muted font-medium py-2 px-3">Peso</th>
                <th className="text-right text-muted font-medium py-2 px-3">Score</th>
                <th className="text-right text-muted font-medium py-2 px-3">Ret 12m</th>
                <th className="text-right text-muted font-medium py-2 px-3">Vol 12m</th>
                <th className="text-right text-muted font-medium py-2 px-3">Fee</th>
              </tr>
            </thead>
            <tbody className="font-mono text-text">
              {data.current_weights.map((w, i) => (
                <tr key={w.fondo} className="border-b border-[#151515]">
                  <td className="py-1.5 pr-4 font-sans text-xs">
                    <span className="inline-block w-2 h-2 rounded-full mr-2"
                          style={{ backgroundColor: COLORS[i % COLORS.length] }} />
                    <Link to={`/detail/${w.fondo}`} className="text-accent hover:underline">{w.fondo}</Link>
                  </td>
                  <td className="text-right px-3">{pct(w.weight)}</td>
                  <td className="text-right px-3">{fmtNum(w.score, 3)}</td>
                  <td className="text-right px-3">{pct(w.ret_12m)}</td>
                  <td className="text-right px-3">{pct(w.vol_12m)}</td>
                  <td className="text-right px-3">{fmtFee(w.fee)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Donut chart */}
        <div className="card p-5 flex items-center justify-center">
          <ResponsiveContainer width="100%" height={280}>
            <PieChart>
              <Pie
                data={donutData}
                cx="50%"
                cy="50%"
                innerRadius={60}
                outerRadius={100}
                paddingAngle={2}
                dataKey="value"
                nameKey="name"
              >
                {donutData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip
                formatter={(value: number) => pct(value)}
                contentStyle={{ backgroundColor: '#111', border: '1px solid #333', borderRadius: 8, fontSize: 11 }}
                itemStyle={{ color: '#ccc' }}
              />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Rebalance history accordion */}
      {rebalanceHistory.length > 0 && (
        <>
          <h2 className="text-base font-semibold text-text mb-2">Historial de carteras</h2>
          <p className="text-xs text-muted mb-3">
            {rebalanceHistory.length} periodos de rebalanceo. Click en una fila para ver los fondos, pesos, retornos y contribuci&oacute;n.
          </p>
          <div className="card p-5 mb-6 overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-[#222]">
                  <th className="text-left text-muted font-medium py-2 pr-4">Rebalanceo</th>
                  <th className="text-right text-muted font-medium py-2 px-3">Duraci&oacute;n</th>
                  <th className="text-right text-muted font-medium py-2 px-3">Fondos</th>
                  <th className="text-right text-muted font-medium py-2 px-3">Retorno</th>
                  <th className="text-right text-muted font-medium py-2 px-3">Fin periodo</th>
                </tr>
              </thead>
              <tbody className="text-text">
                {rebalanceHistory.map((period, i) => (
                  <RebalanceRow
                    key={period.rebal_date}
                    period={period}
                    isLast={i === rebalanceHistory.length - 1}
                  />
                ))}
              </tbody>
            </table>
          </div>
          {rebalanceHistory.some(p => p.optimization_method && p.optimization_method !== 'max_sharpe') && (
            <div className="text-xs text-muted bg-[#0a0a0a] border border-[#1a1a1a] rounded-lg px-4 py-2.5 mb-8 max-w-3xl leading-relaxed">
              <span className="px-1.5 py-0.5 rounded text-[10px] bg-yellow-900/40 text-yellow-400 border border-yellow-800/50 mr-1.5">RF=0</span>
              <code className="text-[#aaa]">max_sharpe</code> con tasa libre = 0: se usa cuando todos los retornos esperados son &asymp; RF y no hay margen para optimizar sobre ella.
              <br className="mb-1" />
              <span className="px-1.5 py-0.5 rounded text-[10px] bg-yellow-900/40 text-yellow-400 border border-yellow-800/50 mr-1.5">Quad. Utility</span>
              <code className="text-[#aaa]">max_quadratic_utility</code>: se usa cuando <code className="text-[#aaa]">max_sharpe</code> es infactible (crisis o matriz mal condicionada). Siempre tiene soluci&oacute;n.
              <br className="mb-1" />
              Periodos sin etiqueta usaron <code className="text-[#aaa]">max_sharpe</code> est&aacute;ndar.
            </div>
          )}
        </>
      )}

      {/* Footnotes */}
      <div className="text-xs text-muted space-y-1.5 mt-4 mb-8 max-w-3xl leading-relaxed">
        <p><strong className="text-[#888]">Semicovarianza:</strong> Covarianza calculada solo con retornos negativos (downside).
          <code className="text-[#aaa]"> max_sharpe()</code> sobre semicov equivale a maximizar Sortino ratio.</p>
        <p><strong className="text-[#888]">Retornos esperados:</strong> EMA (span={config.ema_span ?? 9} meses) da mayor peso a meses recientes vs media simple.</p>
        <p><strong className="text-[#888]">Timing:</strong> Cada {config.rebalance_months ?? 12} meses, los scores OOS seleccionan D10. Con retornos trailing {config.lookback_months}m
          se estima semicov y retornos EMA, y se optimizan pesos. Entre rebalanceos, los pesos se mantienen congelados. El retorno es el realizado del mes siguiente (sin look-ahead).</p>
        <p><strong className="text-[#888]">Constraints:</strong> Peso m&aacute;ximo {pct(config.max_weight)} por fondo.
          Rebalanceo {config.rebalance}. Rf = {pct(config.rf)} anual.</p>
        <p><strong className="text-[#888]">Equal-weight:</strong> Benchmark que asigna 1/N a cada fondo D10,
          sin optimizaci&oacute;n. Permite evaluar si la optimizaci&oacute;n de pesos agrega valor sobre la mera selecci&oacute;n.</p>
      </div>
    </div>
  );
}
