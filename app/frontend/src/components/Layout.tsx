import { NavLink, Outlet } from 'react-router-dom';
import { LayoutDashboard, FileText, Layers, LineChart } from 'lucide-react';
import clsx from 'clsx';

const NAV = [
  { to: '/overview', label: 'Ranking', icon: LayoutDashboard },
  { to: '/detail', label: 'Ficha de fondo', icon: FileText },
  { to: '/drivers', label: 'Drivers', icon: Layers },
  { to: '/backtest', label: 'Backtest', icon: LineChart },
];

export default function Layout() {
  return (
    <div className="min-h-screen flex">
      <aside className="w-60 bg-panel border-r border-line flex flex-col p-4">
        <div className="text-lg font-semibold mb-1">Fund Scoring</div>
        <div className="text-xs text-muted mb-6">Renta Variable Indirecta</div>
        <nav className="space-y-1">
          {NAV.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-2 px-3 py-2 rounded text-sm transition',
                  isActive
                    ? 'bg-accent/15 text-accent'
                    : 'text-muted hover:bg-line/40 hover:text-gray-100'
                )
              }
            >
              <Icon size={16} />
              {label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="flex-1 p-8 overflow-auto">
        <Outlet />
      </main>
    </div>
  );
}
