import { useEffect, useState } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Ruler } from 'lucide-react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid, ReferenceLine, ReferenceArea } from 'recharts';
import { Fund, getFundDetail, getFunds } from '../services/api';
import { useChartRuler } from '../hooks/useChartRuler';
import RulerOverlay from '../components/RulerOverlay';

export default function DetailPage() {
  const { fondo } = useParams();
  const navigate = useNavigate();
  const [data, setData] = useState<any>(null);
  const [pickList, setPickList] = useState<Fund[]>([]);
  const ruler = useChartRuler();

  useEffect(() => {
    getFunds().then(fs => setPickList(fs.slice(0, 30)));
  }, []);
  useEffect(() => {
    if (fondo) getFundDetail(fondo).then(setData);
  }, [fondo]);

  if (!fondo) {
    return (
      <div>
        <h1 className="text-xl font-bold text-text mb-3">Ficha de fondo</h1>
        <p className="text-muted mb-4">Selecciona un fondo (top 30 por score):</p>
        <div className="grid grid-cols-3 gap-2">
          {pickList.map((f, i) => {
            const dBadge = f.decil != null && f.decil >= 8 ? 'badge badge-success'
              : f.decil != null && f.decil <= 3 ? 'badge badge-danger'
              : 'badge badge-neutral';
            return (
              <a key={f.fondo} href={`/detail/${f.fondo}`}
                 className="card-elevated px-3 py-2 flex items-center gap-2 hover:border-accent transition-all">
                <span className="text-xs font-bold text-accent w-5 shrink-0">#{i + 1}</span>
                <span className="text-sm font-medium text-text truncate flex-1">{f.fondo}</span>
                <span className="text-xs text-muted tabular-nums shrink-0">{f.score.toFixed(3)}</span>
                {f.decil != null && <span className={`${dBadge} shrink-0`}>D{f.decil}</span>}
              </a>
            );
          })}
        </div>
      </div>
    );
  }

  if (!data) return <div className="flex items-center gap-3 text-muted"><div className="spinner" />Cargando...</div>;

  const s = data.summary;
  const fechaScore: string | null = s?.fecha_score ?? null;

  const series = data.dates.map((d: string, i: number) => {
    const atOrBefore = !fechaScore || d <= fechaScore;
    const atOrAfter = fechaScore && d >= fechaScore;
    return {
      date: d,
      equity_trailing: atOrBefore ? data.equity[i] : undefined,
      equity_forward: atOrAfter ? data.equity[i] : undefined,
    };
  });
  const growthX = data.equity.length > 0 ? data.equity[data.equity.length - 1] / 100 : null;

  const forwardEndDate: string | null = fechaScore
    ? (() => {
        const d = new Date(fechaScore);
        d.setMonth(d.getMonth() + 6);
        return d.toISOString().slice(0, 10);
      })()
    : null;

  const decil = s?.decil;
  const badgeClass =
    decil != null && decil >= 8 ? 'badge badge-success' :
    decil != null && decil <= 3 ? 'badge badge-danger' :
    'badge badge-neutral';

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-3">
          <h1 className="text-xl font-bold text-text">{fondo}</h1>
          {decil != null && <span className={badgeClass}>D{decil}</span>}
          <span className="text-sm text-muted">Score: <span className="font-semibold text-accent">{s?.score?.toFixed(3) ?? '—'}</span></span>
          <span className="text-xs text-muted border-l border-line pl-3">
            {fechaScore ?? '—'} &middot; Fee {s?.fee != null ? s.fee.toFixed(2) + '%' : '—'} &middot; {s?.n_instrumentos ?? '—'} instr.
          </span>
        </div>
        <button onClick={() => navigate('/detail')} className="btn-secondary flex items-center gap-1.5 !py-1.5 !px-3 !text-xs">
          <ArrowLeft size={14} />
          Volver
        </button>
      </div>

      {/* Info box */}
      <p className="text-xs text-muted mb-3">
        <strong className="text-muted">Trailing</strong> = hasta fecha score &middot; <strong className="text-[#4fc3f7]">Forward (OOS)</strong> = per&iacute;odo de validaci&oacute;n, 6 meses despu&eacute;s del score.
      </p>

      {/* Gráfico equity */}
      <div className="card p-5">
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-muted">Retorno total acumulado (base 100, incluye dividendos)</h2>
          <div className="flex items-center gap-3">
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
            {growthX != null && (
              <span className="text-sm font-bold text-accent tabular-nums">{growthX.toFixed(1)}x</span>
            )}
          </div>
        </div>
        <div className="h-80 relative" ref={ruler.containerRef} style={ruler.isActive ? { cursor: 'crosshair' } : undefined}>
          <ResponsiveContainer>
            <LineChart data={series} onClick={ruler.handleChartClick} onMouseMove={ruler.handleChartMouseMove}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
              <XAxis dataKey="date" tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#222' }} tickLine={{ stroke: '#222' }} />
              <YAxis tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#222' }} tickLine={{ stroke: '#222' }} />
              <Tooltip contentStyle={{ background: '#111111', border: '1px solid #222222', borderRadius: '8px', color: '#f5f5f5' }} />
              {fechaScore && forwardEndDate && (
                <ReferenceArea x1={fechaScore} x2={forwardEndDate} fill="#CF2141" fillOpacity={0.08} />
              )}
              {fechaScore && (
                <ReferenceLine x={fechaScore} stroke="#CF2141" strokeDasharray="4 4" label={{ value: 'Score', position: 'top', fill: '#CF2141', fontSize: 11 }} />
              )}
              <Line type="monotone" dataKey="equity_trailing" stroke="#CF2141" dot={false} strokeWidth={2} connectNulls={false} name="Trailing" />
              {fechaScore && (
                <Line type="monotone" dataKey="equity_forward" stroke="#4fc3f7" dot={false} strokeWidth={2} strokeDasharray="6 3" connectNulls={false} name="Forward (OOS)" />
              )}
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

      {/* KPI strip */}
      <div className="card px-5 py-3 mt-4 flex items-center gap-6 flex-wrap text-sm">
        <div className="flex items-center gap-4 border-l-2 border-line pl-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-muted">Trailing</span>
          <KPI label="Ret 12m" value={s?.ret_12m_trailing} pct />
          <KPI label="Vol" value={s?.vol_12m} pct />
          <KPI label="Sharpe" value={s?.sharpe_12m} />
          <KPI label="Sortino" value={s?.sortino_12m} />
          <KPI label="Max DD" value={s?.max_dd_12m} pct />
        </div>
        <div className="flex items-center gap-4 border-l-2 border-accent pl-3">
          <span className="text-xs font-semibold uppercase tracking-wider text-accent">Forward 6m</span>
          <KPI label="Ret realizado" value={s?.target_realizado_6m} pct />
          <KPI label="Sortino" value={s?.sortino_realizado_6m} />
        </div>
      </div>
    </div>
  );
}

function KPI({ label, value, pct }: { label: string; value: number | string | null | undefined; pct?: boolean }) {
  const isNeg = typeof value === 'number' && value < 0;
  let display: string;
  if (value == null) {
    display = '\u2014';
  } else if (typeof value === 'string') {
    display = value;
  } else if (pct) {
    display = `${(value * 100).toFixed(1)}%`;
  } else if (Number.isInteger(value)) {
    display = value.toString();
  } else {
    display = value.toFixed(2);
  }
  return (
    <span className="flex items-baseline gap-1.5">
      <span className="text-xs text-muted">{label}</span>
      <span className={`font-semibold tabular-nums ${isNeg ? 'text-negative' : 'text-text'}`}>{display}</span>
    </span>
  );
}
