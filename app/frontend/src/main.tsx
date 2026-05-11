import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import OverviewPage from './pages/OverviewPage';
import DetailPage from './pages/DetailPage';
import DriversPage from './pages/DriversPage';
import BacktestPage from './pages/BacktestPage';
import PortfolioPage from './pages/PortfolioPage';
import './index.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Navigate to="/overview" replace />} />
          <Route path="overview" element={<OverviewPage />} />
          <Route path="detail/:fondo?" element={<DetailPage />} />
          <Route path="drivers" element={<DriversPage />} />
          <Route path="backtest" element={<BacktestPage />} />
          <Route path="portfolio" element={<PortfolioPage />} />
          <Route path="*" element={<Navigate to="/overview" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
);
