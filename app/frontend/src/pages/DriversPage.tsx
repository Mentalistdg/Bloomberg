import { useEffect, useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, Cell, Tooltip, ResponsiveContainer, CartesianGrid } from 'recharts';
import { Drivers, getDrivers } from '../services/api';

const LABEL_MAP: Record<string, string> = {
  ret_1m: 'Retorno 1 mes',
  ret_3m: 'Retorno 3 meses',
  ret_6m: 'Retorno 6 meses',
  ret_12m: 'Retorno 12 meses',
  vol_12m: 'Volatilidad 12m',
  max_dd_12m: 'Máx. caída 12m',
  sharpe_12m: 'Sharpe 12m',
  vol_intrames: 'Volatilidad intramensual',
  autocorr_diaria: 'Autocorrelación diaria',
  ratio_dias_cero: 'Ratio días sin movimiento',
  skewness_12m: 'Asimetría 12m',
  hit_rate_12m: 'Tasa de acierto 12m',
  distribution_yield_12m: 'Yield distribuciones 12m',
  persistencia_rank_12m: 'Persistencia ranking 12m',
  fee: 'Comisión',
  log_n_instrumentos: 'Nº instrumentos (log)',
  pct_acum: 'Concentración acumulada',
  fee_observado: 'Fee observado (indicador)',
  concentracion_disponible: 'Concentración disponible (indicador)',
  ret_3m_rank: 'Ranking retorno 3m',
  ret_12m_rank: 'Ranking retorno 12m',
  vol_12m_rank: 'Ranking volatilidad 12m',
  sharpe_12m_rank: 'Ranking Sharpe 12m',
  max_dd_12m_rank: 'Ranking máx. caída 12m',
  fee_rank: 'Ranking comisión',
  log_n_instrumentos_rank: 'Ranking Nº instrumentos',
  pct_acum_rank: 'Ranking concentración',
  vol_intrames_rank: 'Ranking vol. intramensual',
  autocorr_diaria_rank: 'Ranking autocorrelación',
  ratio_dias_cero_rank: 'Ranking días sin movimiento',
  skewness_12m_rank: 'Ranking asimetría 12m',
  hit_rate_12m_rank: 'Ranking tasa de acierto',
};

const label = (f: string) => LABEL_MAP[f] ?? f;

const TOP_N = 10;

export default function DriversPage() {
  const [data, setData] = useState<Drivers | null>(null);

  useEffect(() => { getDrivers().then(setData); }, []);
  if (!data) return <div className="flex items-center gap-3 text-muted"><div className="spinner" />Cargando...</div>;

  const elasticSeries = data.elastic_net_coefs.features
    .map((f, i) => ({
      feature: f,
      label: label(f),
      coef: data.elastic_net_coefs.mean[i],
      std: data.elastic_net_coefs.std[i],
    }))
    .sort((a, b) => Math.abs(b.coef) - Math.abs(a.coef))
    .slice(0, TOP_N);

  return (
    <div>
      <h1 className="text-xl font-bold text-text mb-1">Variables que más influyen en el score</h1>
      <p className="text-sm text-muted mb-6">
        Top {TOP_N} variables por importancia. Positivo = sube el score, negativo = lo baja.
      </p>

      <div className="card p-5">
        <div className="flex items-center justify-between mb-3">
          <h2 className="text-sm font-semibold text-text">Coeficientes ElasticNet</h2>
          <div className="flex items-center gap-4 text-xs text-muted">
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-sm" style={{ background: '#00c853' }} />
              Sube el score
            </span>
            <span className="flex items-center gap-1.5">
              <span className="inline-block w-3 h-3 rounded-sm" style={{ background: '#CF2141' }} />
              Baja el score
            </span>
          </div>
        </div>

        <div className="h-[420px]">
          <ResponsiveContainer>
            <BarChart data={elasticSeries} layout="vertical" margin={{ left: 180, right: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1a1a1a" horizontal={false} />
              <XAxis
                type="number"
                tick={{ fill: '#737373', fontSize: 11 }}
                axisLine={{ stroke: '#222' }}
                tickLine={{ stroke: '#222' }}
              />
              <YAxis
                type="category"
                dataKey="label"
                tick={{ fill: '#a3a3a3', fontSize: 12 }}
                axisLine={{ stroke: '#222' }}
                tickLine={{ stroke: '#222' }}
                width={170}
              />
              <Tooltip
                content={({ active, payload }) => {
                  if (!active || !payload?.length) return null;
                  const d = payload[0].payload;
                  return (
                    <div style={{ background: '#111', border: '1px solid #222', borderRadius: 8, padding: '8px 12px', color: '#f5f5f5', fontSize: 12 }}>
                      <div style={{ fontWeight: 600 }}>{d.label}</div>
                      <div style={{ color: '#737373', fontSize: 11 }}>{d.feature}</div>
                      <div style={{ marginTop: 4 }}>
                        Coef: {d.coef >= 0 ? '+' : ''}{d.coef.toFixed(4)} &plusmn; {d.std.toFixed(4)}
                      </div>
                    </div>
                  );
                }}
              />
              <Bar dataKey="coef" radius={[0, 4, 4, 0]}>
                {elasticSeries.map((d, i) => (
                  <Cell key={i} fill={d.coef >= 0 ? '#00c853' : '#CF2141'} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <p className="text-[11px] text-muted mt-4">
        ElasticNet: &alpha;={data.elastic_net_hyperparams.alpha_mean.toFixed(4)},
        l1_ratio={data.elastic_net_hyperparams.l1_ratio_mean.toFixed(3)}.
        Coeficientes sobre variables estandarizadas; promedio entre folds de walk-forward CV con embargo = horizonte.
        <br />
        Sharpe 12m aparece con signo negativo: controlando por otras variables, fondos con alto Sharpe pasado tienden a revertir (reversión a la media).
      </p>
    </div>
  );
}
