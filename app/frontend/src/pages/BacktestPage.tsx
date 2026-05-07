import { useEffect, useState } from 'react';
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Legend } from 'recharts';
import { BacktestPoint, getBacktest } from '../services/api';

export default function BacktestPage() {
  const [data, setData] = useState<BacktestPoint[]>([]);

  useEffect(() => { getBacktest().then(setData); }, []);

  return (
    <div>
      <h1 className="text-xl font-semibold mb-1">Backtest top-Q5 vs bottom-Q1</h1>
      <p className="text-sm text-muted mb-6">
        Retorno realizado a 12m promedio del quintil superior, quintil inferior, y universo
        equiponderado. La diferencia (spread) es el "valor de la señal".
      </p>
      <div className="bg-panel rounded border border-line p-4 h-96">
        <ResponsiveContainer>
          <LineChart data={data}>
            <XAxis dataKey="date" tick={{ fill: '#9ca3af', fontSize: 10 }} />
            <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }}
                   tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
            <Tooltip contentStyle={{ background: '#111827', border: '1px solid #1f2937' }}
                     formatter={(v: number) => `${(v * 100).toFixed(2)}%`} />
            <Legend />
            <Line type="monotone" dataKey="top_q5" stroke="#10b981" name="Top Q5" dot={false} strokeWidth={1.5} />
            <Line type="monotone" dataKey="universe" stroke="#9ca3af" name="Universo" dot={false} strokeWidth={1} />
            <Line type="monotone" dataKey="bot_q1" stroke="#ef4444" name="Bot Q1" dot={false} strokeWidth={1.5} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
