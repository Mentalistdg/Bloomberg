import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Cell, Tooltip, ResponsiveContainer } from 'recharts';
import { Drivers, getDrivers } from '../services/api';

export default function DriversPage() {
  const [data, setData] = useState<Drivers | null>(null);

  useEffect(() => { getDrivers().then(setData); }, []);
  if (!data) return <p className="text-muted">Cargando…</p>;

  const series = data.elastic_net_coefs.features
    .map((f, i) => ({ feature: f, coef: data.elastic_net_coefs.mean[i] }))
    .sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef));

  return (
    <div>
      <h1 className="text-xl font-semibold mb-1">Drivers del score (ElasticNet)</h1>
      <p className="text-sm text-muted mb-6">
        Coeficientes promedio entre folds de walk-forward CV. Variables estandarizadas.
        α≈{data.elastic_net_hyperparams.alpha_mean.toFixed(3)}, l1_ratio≈{data.elastic_net_hyperparams.l1_ratio_mean.toFixed(2)}
      </p>
      <div className="bg-panel rounded border border-line p-4 h-[500px]">
        <ResponsiveContainer>
          <BarChart data={series} layout="vertical" margin={{ left: 100 }}>
            <XAxis type="number" tick={{ fill: '#9ca3af', fontSize: 10 }} />
            <YAxis type="category" dataKey="feature" tick={{ fill: '#9ca3af', fontSize: 10 }} />
            <Tooltip contentStyle={{ background: '#111827', border: '1px solid #1f2937' }} />
            <Bar dataKey="coef">
              {series.map((d, i) => (
                <Cell key={i} fill={d.coef >= 0 ? '#10b981' : '#ef4444'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
