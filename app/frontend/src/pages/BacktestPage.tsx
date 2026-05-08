import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend, CartesianGrid } from 'recharts';
import { BacktestPoint, getBacktest } from '../services/api';

export default function BacktestPage() {
  const [data, setData] = useState<BacktestPoint[]>([]);

  useEffect(() => { getBacktest().then(setData); }, []);

  return (
    <div>
      <h1 className="text-xl font-bold text-text mb-1">Backtest top-Q5 vs bottom-Q1</h1>
      <p className="text-sm text-muted mb-6">
        Retorno realizado a 12m promedio del quintil superior, quintil inferior, y universo
        equiponderado. La diferencia (spread) es el "valor de la se&ntilde;al".
      </p>
      <div className="card p-5 h-96">
        <ResponsiveContainer>
          <LineChart data={data}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" />
            <XAxis dataKey="date" tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#222' }} tickLine={{ stroke: '#222' }} />
            <YAxis tick={{ fill: '#737373', fontSize: 10 }} axisLine={{ stroke: '#222' }} tickLine={{ stroke: '#222' }}
                   tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
            <Tooltip contentStyle={{ background: '#111111', border: '1px solid #222222', borderRadius: '8px', color: '#f5f5f5' }}
                     formatter={(v: number) => `${(v * 100).toFixed(2)}%`} />
            <Legend wrapperStyle={{ color: '#737373' }} />
            <Line type="monotone" dataKey="top_q5" stroke="#00c853" name="Top Q5" dot={false} strokeWidth={2} />
            <Line type="monotone" dataKey="universe" stroke="#525252" name="Universo" dot={false} strokeWidth={1.5} strokeDasharray="4 4" />
            <Line type="monotone" dataKey="bot_q1" stroke="#CF2141" name="Bot Q1" dot={false} strokeWidth={2} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
