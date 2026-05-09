import axios from 'axios';

export const api = axios.create({
  baseURL: '/api',
  timeout: 15000,
});

export interface Fund {
  fondo: string;
  fecha_score: string;
  score: number;
  decil: number | null;
  target_realizado_12m: number | null;
  ret_12m_trailing: number | null;
  vol_12m: number | null;
  sharpe_12m: number | null;
  max_dd_12m: number | null;
  fee: number | null;
  fee_observado: number | null;
  n_instrumentos: number | null;
}

export interface FundDetail extends Fund {
  dates: string[];
  ret_mensual: (number | null)[];
  equity: number[];
  score: any; // serie temporal del score (puede ser nulo en periodos sin fold)
}

export interface Drivers {
  elastic_net_coefs: { features: string[]; mean: number[]; std: number[]; };
  elastic_net_hyperparams: { alpha_mean: number; l1_ratio_mean: number; };
  lgbm_importance: { features: string[]; mean: number[]; };
}

export interface BacktestPoint {
  date: string;
  top_d10: number | null;
  bot_d1: number | null;
  universe: number | null;
  spread: number | null;
  naive_top_d10: number | null;
  naive_bot_d1: number | null;
  naive_spread: number | null;
  n_funds: number;
  is_production: boolean;
  is_partial: boolean;
  months_forward: number;
}

export interface FoldBoundary {
  fold: number;
  val_start: string;
  val_end: string;
}

export interface Meta {
  as_of: string;
  n_funds: number;
  n_months: number;
  n_folds: number;
  metrics_summary: Record<string, {
    ic_mean: number;
    ic_ir: number;
    ic_hit_meses: number;
    ic_ci95_low: number;
    ic_ci95_high: number;
    spread_q5_q1: number;
    hit_top25: number;
  }>;
  diebold_mariano_elastic_vs_benchmark?: { stat: number; p_value: number; n: number };
  fold_boundaries?: FoldBoundary[];
  primary_model: string;
  scorers_compared?: string[];
}

export const getFunds = (params?: { decil?: number; limit?: number }) =>
  api.get<Fund[]>('/funds', { params }).then(r => r.data);

export const getFundDetail = (fondo: string) =>
  api.get<any>(`/funds/${fondo}`).then(r => r.data);

export const getDrivers = () => api.get<Drivers>('/drivers').then(r => r.data);
export const getBacktest = () => api.get<BacktestPoint[]>('/backtest').then(r => r.data);
export const getMeta = () => api.get<Meta>('/meta').then(r => r.data);
