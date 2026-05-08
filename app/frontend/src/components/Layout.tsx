import { NavLink, Outlet } from 'react-router-dom';
import { LayoutDashboard, FileText, Layers, LineChart } from 'lucide-react';
import clsx from 'clsx';
import logoImg from '../assets/logo-habitat.png';

const NAV = [
  { to: '/overview', label: 'Ranking', icon: LayoutDashboard },
  { to: '/detail', label: 'Ficha de fondo', icon: FileText },
  { to: '/drivers', label: 'Drivers', icon: Layers },
  { to: '/backtest', label: 'Backtest', icon: LineChart },
];

export default function Layout() {
  return (
    <div className="min-h-screen flex flex-col bg-[#0a0a0a]">
      {/* Header */}
      <header className="border-b border-line bg-panel">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <img src={logoImg} alt="AFP Habitat" className="h-16 w-auto" />
            <div>
              <div className="text-base font-bold text-text tracking-wide">FUND SCORING</div>
              <div className="text-[10px] text-muted uppercase tracking-widest">Renta Variable Internacional</div>
            </div>
          </div>
          <nav className="flex items-center gap-1">
            {NAV.map(({ to, label, icon: Icon }) => (
              <NavLink
                key={to}
                to={to}
                className={({ isActive }) =>
                  clsx(
                    'flex items-center gap-2 px-4 py-2 text-sm font-medium transition-all border-b-2',
                    isActive
                      ? 'border-accent text-accent'
                      : 'border-transparent text-muted hover:text-text'
                  )
                }
              >
                <Icon size={16} />
                {label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      {/* Main content */}
      <main className="flex-1 max-w-7xl mx-auto w-full p-6">
        <Outlet />
      </main>

      {/* Footer */}
      <footer className="border-t border-line py-4 text-center">
        <p className="text-[11px] text-muted">Scoring out-of-sample &middot; Datos hasta mayo 2026 &middot; &Uacute;ltimo corte validado: dic 2024</p>
      </footer>
    </div>
  );
}
