import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Cell, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { Drivers, getDrivers } from '../services/api';

export default function DriversPage() {
  const [data, setData] = useState<Drivers | null>(null);

  useEffect(() => { getDrivers().then(setData); }, []);
  if (!data) return <div className="flex items-center gap-3 text-muted"><div className="spinner" />Cargando...</div>;

  const series = data.elastic_net_coefs.features
    .map((f, i) => ({ feature: f, coef: data.elastic_net_coefs.mean[i] }))
    .sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef));

  return (
    <div>
      <h1 className="text-xl font-bold text-text mb-1">Drivers del score (ElasticNet)</h1>
      <p className="text-sm text-muted mb-6">
        Coeficientes promedio entre folds de walk-forward CV. Variables estandarizadas.
        &alpha;&asymp;{data.elastic_net_hyperparams.alpha_mean.toFixed(3)}, l1_ratio&asymp;{data.elastic_net_hyperparams.l1_ratio_mean.toFixed(2)}
      </p>
      <div className="card p-5 h-[500px]">
        <ResponsiveContainer>
          <BarChart data={series} layout="vertical" margin={{ left: 100 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" horizontal={false} />
            <XAxis type="number" tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#222' }} tickLine={{ stroke: '#222' }} />
            <YAxis type="category" dataKey="feature" tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#222' }} tickLine={{ stroke: '#222' }} />
            <Tooltip contentStyle={{ background: '#111111', border: '1px solid #222222', borderRadius: '8px', color: '#f5f5f5' }} />
            <Bar dataKey="coef" radius={[0, 4, 4, 0]}>
              {series.map((d, i) => (
                <Cell key={i} fill={d.coef >= 0 ? '#00c853' : '#CF2141'} />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
