import { useEffect, useMemo, useState, useCallback } from 'react';
import { Link } from 'react-router-dom';
import { Download, Info } from 'lucide-react';
import * as XLSX from 'xlsx';
import { Fund, Meta, getFunds, getMeta } from '../services/api';
import LoadingScreen from '../components/LoadingScreen';

/* ── Formatters ── */
const fmt = (v: number | null, p = 2) => (v == null ? '—' : v.toFixed(p));
const pct = (v: number | null) => (v == null ? '—' : `${(v * 100).toFixed(1)}%`);

/* ── Column definitions ── */
type SortKey = keyof Fund;

/* Columns where negative values should render in red */
const NEG_RED_KEYS = new Set<SortKey>([
  'ret_12m_trailing', 'sharpe_12m', 'max_dd_12m', 'target_realizado_12m',
]);

interface ColDef {
  key: SortKey;
  label: string;
  tooltip: string;
  group: 'id' | 'score' | 'perf' | 'risk' | 'struct';
  align: 'left' | 'right';
  format: (f: Fund) => string | null;
}

const COLUMNS: ColDef[] = [
  {
    key: 'fondo', label: 'Fondo', tooltip: 'Identificador del fondo mutuo',
    group: 'id', align: 'left', format: f => f.fondo,
  },
  {
    key: 'decil', label: 'D', tooltip: 'Decil del score (D10 = mejor)',
    group: 'id', align: 'left', format: f => f.decil != null ? `D${f.decil}` : null,
  },
  {
    key: 'score', label: 'Score', tooltip: 'Score compuesto del modelo (0-1)',
    group: 'score', align: 'right', format: f => fmt(f.score, 3),
  },
  {
    key: 'ret_12m_trailing', label: 'Ret 12m', tooltip: 'Retorno trailing 12 meses',
    group: 'perf', align: 'right', format: f => pct(f.ret_12m_trailing),
  },
  {
    key: 'vol_12m', label: 'Vol 12m', tooltip: 'Volatilidad anualizada 12 meses',
    group: 'risk', align: 'right', format: f => pct(f.vol_12m),
  },
  {
    key: 'sharpe_12m', label: 'Sharpe', tooltip: 'Ratio Sharpe 12 meses (ret/vol)',
    group: 'perf', align: 'right', format: f => fmt(f.sharpe_12m, 2),
  },
  {
    key: 'max_dd_12m', label: 'Max DD', tooltip: 'Máximo drawdown 12 meses',
    group: 'risk', align: 'right', format: f => pct(f.max_dd_12m),
  },
  {
    key: 'fee', label: 'Fee', tooltip: 'Comisión anual (%)',
    group: 'struct', align: 'right', format: f => f.fee == null ? null : `${f.fee.toFixed(2)}%`,
  },
  {
    key: 'n_instrumentos', label: 'N instr.', tooltip: 'Número de instrumentos subyacentes',
    group: 'struct', align: 'right', format: f => f.n_instrumentos != null ? String(f.n_instrumentos) : null,
  },
  {
    key: 'target_realizado_12m', label: 'Realizado 12m', tooltip: 'Retorno efectivamente observado en los 12 meses posteriores al score (validaci\u00f3n out-of-sample)',
    group: 'perf', align: 'right', format: f => pct(f.target_realizado_12m),
  },
];

/* Group border: show left border when a column starts a new group */
function groupBorder(idx: number): string {
  if (idx === 0) return '';
  if (COLUMNS[idx].group !== COLUMNS[idx - 1].group) return 'group-separator';
  return '';
}

/* ── Component ── */
export default function OverviewPage() {
  const [funds, setFunds] = useState<Fund[]>([]);
  const [meta, setMeta] = useState<Meta | null>(null);
  const [loading, setLoading] = useState(true);

  /* Sorting */
  const [sortCol, setSortCol] = useState<SortKey>('score');
  const [sortDir, setSortDir] = useState<'asc' | 'desc'>('desc');

  /* Filters */
  const [search, setSearch] = useState('');
  const [activeDeciles, setActiveDeciles] = useState<Set<number>>(new Set());
  const [scoreMin, setScoreMin] = useState('');
  const [scoreMax, setScoreMax] = useState('');

  useEffect(() => {
    async function loadData() {
      const splashMin = new Promise(resolve => setTimeout(resolve, 4000));
      try {
        const [f, m] = await Promise.all([getFunds(), getMeta()]);
        setFunds(f);
        setMeta(m);
        await splashMin;
      } catch (err) {
        console.error('Failed to load data:', err);
        await splashMin;
      } finally {
        setLoading(false);
      }
    }
    loadData();
  }, []);

  /* Toggle decile */
  const toggleDecil = (d: number) => {
    setActiveDeciles(prev => {
      const next = new Set(prev);
      if (next.has(d)) next.delete(d); else next.add(d);
      return next;
    });
  };

  /* Clear all filters */
  const clearFilters = () => {
    setSearch('');
    setActiveDeciles(new Set());
    setScoreMin('');
    setScoreMax('');
  };

  const hasFilters = search !== '' || activeDeciles.size > 0 || scoreMin !== '' || scoreMax !== '';

  /* Filtered + sorted data */
  const filtered = useMemo(() => {
    let data = funds;

    if (search) {
      const q = search.toLowerCase();
      data = data.filter(f => f.fondo.toLowerCase().includes(q));
    }

    if (activeDeciles.size > 0) {
      data = data.filter(f => f.decil != null && activeDeciles.has(f.decil));
    }

    const sMin = scoreMin !== '' ? parseFloat(scoreMin) : null;
    const sMax = scoreMax !== '' ? parseFloat(scoreMax) : null;
    if (sMin != null) data = data.filter(f => f.score >= sMin);
    if (sMax != null) data = data.filter(f => f.score <= sMax);

    /* Sort */
    data = [...data].sort((a, b) => {
      const va = a[sortCol];
      const vb = b[sortCol];
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === 'string' && typeof vb === 'string') {
        return sortDir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
      }
      const diff = (va as number) - (vb as number);
      return sortDir === 'asc' ? diff : -diff;
    });

    return data;
  }, [funds, search, activeDeciles, scoreMin, scoreMax, sortCol, sortDir]);

  /* Sort handler */
  const handleSort = (key: SortKey) => {
    if (sortCol === key) {
      setSortDir(d => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortCol(key);
      setSortDir('desc');
    }
  };

  /* Export to Excel */
  const exportToExcel = useCallback(() => {
    const rows = filtered.map(f =>
      Object.fromEntries(COLUMNS.map(col => [col.label, col.format(f) ?? '']))
    );
    const ws = XLSX.utils.json_to_sheet(rows);
    const wb = XLSX.utils.book_new();
    XLSX.utils.book_append_sheet(wb, ws, 'Ranking');
    XLSX.writeFile(wb, 'fund_scoring.xlsx');
  }, [filtered]);

  if (loading) {
    return <LoadingScreen />;
  }

  return (
    <div>
      {/* Header */}
      <div className="mb-4">
        <h1 className="text-xl font-bold text-text">Ranking de fondos &mdash; &Uacute;ltimo corte de validaci&oacute;n</h1>
        {meta && (
          <p className="text-sm text-muted mt-1">
            {meta.n_funds} fondos &middot; corte {meta.as_of} &middot; validado con datos hasta dic 2025 &middot; modelo: {meta.primary_model}
          </p>
        )}
        <div className="flex items-start gap-2 mt-3 px-3 py-2.5 rounded-lg bg-panel border border-line text-xs text-muted leading-relaxed max-w-3xl">
          <Info size={14} className="mt-0.5 shrink-0 text-accent" />
          <span>
            Los scores corresponden al &uacute;ltimo mes donde el modelo pudo ser validado
            out-of-sample (requiere 12 meses de datos forward). La columna
            &laquo;Realizado 12m&raquo; muestra el retorno que efectivamente ocurri&oacute;.
          </span>
        </div>
      </div>

      {/* Filter bar */}
      <div className="filter-bar">
        {/* Search */}
        <input
          type="text"
          value={search}
          onChange={e => setSearch(e.target.value)}
          placeholder="Buscar fondo…"
          className="filter-input w-48"
        />

        {/* Decile toggles */}
        <div className="flex items-center gap-1">
          <span className="text-xs text-muted font-medium mr-1">Decil:</span>
          {[10, 9, 8, 7, 6, 5, 4, 3, 2, 1].map(d => (
            <button
              key={d}
              onClick={() => toggleDecil(d)}
              className={`decil-toggle ${activeDeciles.has(d) ? 'active' : ''}`}
            >
              {d}
            </button>
          ))}
        </div>

        {/* Score range */}
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-muted font-medium">Score:</span>
          <input
            type="number"
            step="0.01"
            value={scoreMin}
            onChange={e => setScoreMin(e.target.value)}
            placeholder="min"
            className="filter-input w-16 text-center"
          />
          <span className="text-muted text-xs">–</span>
          <input
            type="number"
            step="0.01"
            value={scoreMax}
            onChange={e => setScoreMax(e.target.value)}
            placeholder="max"
            className="filter-input w-16 text-center"
          />
        </div>

        {/* Clear */}
        {hasFilters && (
          <button onClick={clearFilters} className="text-xs text-muted hover:text-accent transition-colors">
            Limpiar filtros
          </button>
        )}

        {/* Counter + Export */}
        <span className="text-xs text-muted ml-auto tabular-nums">
          Mostrando {filtered.length} de {funds.length} fondos
        </span>
        <button onClick={exportToExcel} className="btn-secondary flex items-center gap-1.5 !py-1.5 !px-3 !text-xs">
          <Download size={14} />
          Excel
        </button>
      </div>

      {/* Table */}
      <div className="card screener-container">
        <div className="screener-scroll">
          <table className="table-dark">
            <thead>
              <tr>
                {COLUMNS.map((col, i) => {
                  const isSorted = sortCol === col.key;
                  const arrow = isSorted ? (sortDir === 'desc' ? ' ▼' : ' ▲') : '';
                  return (
                    <th
                      key={col.key}
                      className={[
                        'sortable',
                        isSorted ? 'sorted' : '',
                        col.align === 'right' ? 'text-right' : '',
                        groupBorder(i),
                        i === 0 ? 'sticky-col sticky-col-head' : '',
                      ].filter(Boolean).join(' ')}
                      onClick={() => handleSort(col.key)}
                      title={col.tooltip}
                    >
                      {col.label}{arrow}
                    </th>
                  );
                })}
              </tr>
            </thead>
            <tbody>
              {filtered.map(f => (
                <tr key={f.fondo}>
                  {COLUMNS.map((col, i) => {
                    /* First column: link */
                    if (col.key === 'fondo') {
                      return (
                        <td key={col.key} className={`sticky-col ${groupBorder(i)}`}>
                          <Link to={`/detail/${f.fondo}`} className="text-accent font-medium hover:underline">
                            {f.fondo}
                          </Link>
                        </td>
                      );
                    }

                    /* Decil badge */
                    if (col.key === 'decil') {
                      return (
                        <td key={col.key} className={groupBorder(i)}>
                          <span className={`badge ${
                            (f.decil ?? 0) >= 8 ? 'badge-success'
                            : (f.decil ?? 0) <= 3 ? 'badge-danger'
                            : 'badge-neutral'
                          }`}>
                            {f.decil != null ? `D${f.decil}` : '—'}
                          </span>
                        </td>
                      );
                    }

                    /* Negative-red text for key columns */
                    const v = f[col.key];
                    const isNeg = NEG_RED_KEYS.has(col.key) && typeof v === 'number' && v < 0;

                    const formatted = col.format(f);
                    return (
                      <td
                        key={col.key}
                        className={[
                          col.align === 'right' ? 'text-right' : '',
                          'font-mono text-sm tabular-nums',
                          isNeg ? 'text-negative' : '',
                          groupBorder(i),
                        ].filter(Boolean).join(' ')}
                      >
                        {formatted ?? '—'}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
