import { Link, useLocation } from 'react-router-dom';
import { ShieldCheck } from 'lucide-react';

export default function Navbar() {
  const { pathname } = useLocation();

  const links = [
    { to: '/', label: 'Home' },
    { to: '/blind', label: 'Blind Detection' },
    { to: '/compare', label: 'Comparative' },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 border-b border-dark-300 bg-dark-900/80 backdrop-blur-md">
      <div className="max-w-6xl mx-auto px-6 h-16 flex items-center justify-between">
        <Link to="/" className="flex items-center gap-2 font-bold text-lg">
          <ShieldCheck className="text-purple-500" size={22} />
          <span className="gradient-text">FakeGuard AI</span>
        </Link>

        <div className="flex items-center gap-1">
          {links.map(({ to, label }) => (
            <Link
              key={to}
              to={to}
              className={`px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200 ${
                pathname === to
                  ? 'bg-purple-600/20 text-purple-400'
                  : 'text-slate-400 hover:text-slate-200 hover:bg-dark-400'
              }`}
            >
              {label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
