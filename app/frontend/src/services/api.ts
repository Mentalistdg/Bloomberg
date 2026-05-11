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
  target_realizado_6m: number | null;
  sortino_realizado_6m: number | null;
  ret_12m_trailing: number | null;
  vol_12m: number | null;
  sharpe_12m: number | null;
  sortino_12m: number | null;
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
  n_funds_total?: number;
  n_months: number;
  n_folds: number;
  metrics_summary: Record<string, {
    ic_mean: number;
    ic_ir: number;
    ic_hit_meses: number;
    ic_ci95_low: number;
    ic_ci95_high: number;
    spread_d10_d1: number;
    hit_top25: number;
  }>;
  fold_boundaries?: FoldBoundary[];
  primary_model: string;
  scorers_compared?: string[];
}

export interface PortfolioWeight {
  fondo: string;
  weight: number;
  score: number | null;
  ret_12m: number | null;
  vol_12m: number | null;
  fee: number | null;
}

export interface PortfolioMetrics {
  annual_return: number;
  annual_vol: number;
  sortino: number | null;
  max_drawdown: number;
  hit_rate: number;
}

export interface RebalanceHolding {
  fondo: string;
  weight: number;
  period_return: number;
  contribution: number;
}

export interface RebalancePeriod {
  rebal_date: string;
  period_end: string;
  n_months: number;
  portfolio_return: number;
  holdings: RebalanceHolding[];
  ew_portfolio_return?: number;
  ew_holdings?: RebalanceHolding[];
  optimization_method?: string;
}

export interface PortfolioData {
  as_of: string;
  last_rebalance_date?: string;
  config: {
    lookback_months: number;
    risk_model: string;
    max_weight: number;
    rebalance: string;
    rf: number;
    returns_model?: string;
    ema_span?: number;
    rebalance_months?: number;
  };
  backtest: {
    dates: string[];
    opt_ret: number[];
    ew_ret: number[];
    opt_equity: number[];
    ew_equity: number[];
    rebalance_dates?: string[];
  };
  metrics: {
    optimal: PortfolioMetrics;
    equal_weight: PortfolioMetrics;
  };
  current_weights: PortfolioWeight[];
  rebalance_history?: RebalancePeriod[];
}

export const getFunds = (params?: { decil?: number; limit?: number }) =>
  api.get<Fund[]>('/funds', { params }).then(r => r.data);

export const getFundDetail = (fondo: string) =>
  api.get<any>(`/funds/${fondo}`).then(r => r.data);

export const getDrivers = () => api.get<Drivers>('/drivers').then(r => r.data);
export const getBacktest = () => api.get<BacktestPoint[]>('/backtest').then(r => r.data);
export const getMeta = () => api.get<Meta>('/meta').then(r => r.data);
export const getPortfolio = () => api.get<PortfolioData>('/portfolio').then(r => r.data);
