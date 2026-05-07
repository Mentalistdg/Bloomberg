import { useEffect, useState } from 'react';
import { useParams } from 'react-router-dom';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer } from 'recharts';
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
        <h1 className="text-xl font-semibold mb-3">Ficha de fondo</h1>
        <p className="text-muted mb-4">Elegí un fondo (top 30 por score):</p>
        <div className="grid grid-cols-3 gap-2">
          {pickList.map(f => (
            <a key={f} href={`/detail/${f}`} className="px-3 py-2 bg-panel rounded border border-line hover:border-accent text-sm">
              {f}
            </a>
          ))}
        </div>
      </div>
    );
  }

  if (!data) return <p className="text-muted">Cargando…</p>;

  const series = data.dates.map((d: string, i: number) => ({
    date: d,
    equity: data.equity[i],
  }));
  const s = data.summary;

  return (
    <div>
      <h1 className="text-xl font-semibold mb-1">{fondo}</h1>
      <p className="text-sm text-muted mb-6">Score actual: {s?.score?.toFixed(3)} (decil {s?.decil ?? '—'})</p>
      <div className="grid grid-cols-4 gap-3 mb-6">
        <Stat label="Ret 12m" value={s?.ret_12m_trailing} pct />
        <Stat label="Vol 12m" value={s?.vol_12m} pct />
        <Stat label="Sharpe 12m" value={s?.sharpe_12m} />
        <Stat label="Max DD 12m" value={s?.max_dd_12m} pct />
      </div>
      <div className="bg-panel rounded border border-line p-4 h-80">
        <ResponsiveContainer>
          <LineChart data={series}>
            <XAxis dataKey="date" tick={{ fill: '#9ca3af', fontSize: 10 }} />
            <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} />
            <Tooltip contentStyle={{ background: '#111827', border: '1px solid #1f2937' }} />
            <Line type="monotone" dataKey="equity" stroke="#2563eb" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function Stat({ label, value, pct }: { label: string; value: number | null | undefined; pct?: boolean }) {
  return (
    <div className="bg-panel rounded border border-line p-3">
      <div className="text-xs text-muted">{label}</div>
      <div className="text-lg font-mono">
        {value == null ? '—' : pct ? `${(value * 100).toFixed(1)}%` : value.toFixed(2)}
      </div>
    </div>
  );
}
