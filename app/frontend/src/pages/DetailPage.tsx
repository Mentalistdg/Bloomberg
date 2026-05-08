import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { getFundDetail, getFunds } from '../services/api';

export default function DetailPage() {
  const { fondo } = useParams();
  const [data, setData] = useState<any>(null);
  const [pickList, setPickList] = useState<string[]>([]);

  useEffect(() => {
    getFunds().then(fs => setPickList(fs.slice(0, 30).map(f => f.fondo)));
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
          {pickList.map(f => (
            <a key={f} href={`/detail/${f}`}
               className="card-elevated px-3 py-2 text-sm font-medium text-text hover:border-accent transition-all">
              {f}
            </a>
          ))}
        </div>
      </div>
    );
  }

  if (!data) return <div className="flex items-center gap-3 text-muted"><div className="spinner" />Cargando...</div>;

  const series = data.dates.map((d: string, i: number) => ({
    date: d,
    equity: data.equity[i],
  }));
  const s = data.summary;

  return (
    <div>
      <h1 className="text-xl font-bold text-text mb-1">{fondo}</h1>
      <p className="text-sm text-muted mb-6">Score actual: <span className="font-semibold text-accent">{s?.score?.toFixed(3)}</span> (decil {s?.decil ?? '\u2014'})</p>
      <div className="grid grid-cols-4 gap-3 mb-6">
        <Stat label="Ret 12m" value={s?.ret_12m_trailing} pct />
        <Stat label="Vol 12m" value={s?.vol_12m} pct />
        <Stat label="Sharpe 12m" value={s?.sharpe_12m} />
        <Stat label="Max DD 12m" value={s?.max_dd_12m} pct />
      </div>
      <div className="card p-5 h-80">
        <ResponsiveContainer>
          <LineChart data={series}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
            <XAxis dataKey="date" tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#222' }} tickLine={{ stroke: '#222' }} />
            <YAxis tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#222' }} tickLine={{ stroke: '#222' }} />
            <Tooltip contentStyle={{ background: '#111111', border: '1px solid #222222', borderRadius: '8px', color: '#f5f5f5' }} />
            <Line type="monotone" dataKey="equity" stroke="#CF2141" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Stat({ label, value, pct }: { label: string; value: number | null | undefined; pct?: boolean }) {
  const isNeg = value != null && value < 0;
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className={`metric-value ${isNeg ? 'text-negative' : 'text-text'}`}>
        {value == null ? '\u2014' : pct ? `${(value * 100).toFixed(1)}%` : value.toFixed(2)}
      </div>
    </div>
  );
}
