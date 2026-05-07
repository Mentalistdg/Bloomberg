import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Fund, Meta, getFunds, getMeta } from '../services/api';

const fmt = (v: number | null, p = 2) => (v == null ? '—' : v.toFixed(p));
const pct = (v: number | null) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);

export default function OverviewPage() {
  const [funds, setFunds] = useState<Fund[]>([]);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [decilFilter, setDecilFilter] = useState<number | ''>('');

  useEffect(() => {
    Promise.all([getFunds(), getMeta()]).then(([f, m]) => {
      setFunds(f);
      setMeta(m);
    });
  }, []);

  const filtered = decilFilter ? funds.filter(f => f.decil === decilFilter) : funds;

  return (
    <div>
      <div className="flex items-baseline justify-between mb-6">
        <div>
          <h1 className="text-xl font-semibold">Ranking de fondos por score</h1>
          {meta && (
            <p className="text-sm text-muted mt-1">
              {meta.n_funds} fondos · cierre {meta.as_of} · modelo principal: {meta.primary_model}
            </p>
          )}
        </div>
        <div className="text-sm">
          <label className="text-muted mr-2">Decil:</label>
          <select
            value={decilFilter}
            onChange={(e) => setDecilFilter(e.target.value === '' ? '' : Number(e.target.value))}
            className="bg-panel border border-line rounded px-2 py-1"
          >
            <option value="">Todos</option>
            {[10, 9, 8, 7, 6, 5, 4, 3, 2, 1].map(d => (
              <option key={d} value={d}>D{d}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="bg-panel rounded border border-line overflow-hidden">
        <table>
          <thead>
            <tr>
              <th>Fondo</th>
              <th>Decil</th>
              <th className="text-right">Score</th>
              <th className="text-right">Ret 12m</th>
              <th className="text-right">Vol 12m</th>
              <th className="text-right">Sharpe 12m</th>
              <th className="text-right">Max DD 12m</th>
              <th className="text-right">Fee</th>
              <th className="text-right">N instr.</th>
            </tr>
          </thead>
          <tbody>
            {filtered.map(f => (
              <tr key={f.fondo}>
                <td>
                  <Link to={`/detail/${f.fondo}`} className="text-accent hover:underline">
                    {f.fondo}
                  </Link>
                </td>
                <td>{f.decil ?? '—'}</td>
                <td className="text-right font-mono">{fmt(f.score, 3)}</td>
                <td className="text-right font-mono">{pct(f.ret_12m_trailing)}</td>
                <td className="text-right font-mono">{pct(f.vol_12m)}</td>
                <td className="text-right font-mono">{fmt(f.sharpe_12m, 2)}</td>
                <td className="text-right font-mono">{pct(f.max_dd_12m)}</td>
                <td className="text-right font-mono">{f.fee == null ? '—' : `${f.fee.toFixed(2)}%`}</td>
                <td className="text-right font-mono">{f.n_instrumentos ?? '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
