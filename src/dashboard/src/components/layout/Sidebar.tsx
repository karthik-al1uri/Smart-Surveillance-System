import { NavLink } from 'react-router-dom';
import {
  HomeIcon,
  VideoCameraIcon,
  BellIcon,
  ListBulletIcon,
  ChartBarIcon,
  Cog6ToothIcon,
  UsersIcon,
} from '@heroicons/react/24/outline';
import { useAlertStore } from '../../stores/alertStore';
import { useAuthStore } from '../../stores/authStore';

const NAV = [
  { to: '/', label: 'Dashboard', icon: HomeIcon },
  { to: '/live', label: 'Live View', icon: VideoCameraIcon },
  { to: '/alerts', label: 'Alerts', icon: BellIcon, badge: true },
  { to: '/events', label: 'Events', icon: ListBulletIcon },
  { to: '/analytics', label: 'Analytics', icon: ChartBarIcon },
  { to: '/settings', label: 'Settings', icon: Cog6ToothIcon },
];

interface SidebarProps {
  open: boolean;
}

export function Sidebar({ open }: SidebarProps) {
  const { unreadCount } = useAlertStore();
  const { user } = useAuthStore();

  return (
    <aside
      className={`flex flex-col bg-gray-900 text-white transition-all duration-200 ${
        open ? 'w-56' : 'w-14'
      } min-h-screen flex-shrink-0`}
    >
      <div className={`flex items-center gap-2 px-3 py-4 border-b border-gray-700 ${open ? '' : 'justify-center'}`}>
        <span className="text-xl">🎥</span>
        {open && <span className="font-semibold text-sm truncate">SSS Dashboard</span>}
      </div>

      <nav className="flex-1 py-4 flex flex-col gap-1">
        {NAV.map(({ to, label, icon: Icon, badge }) => (
          <NavLink
            key={to}
            to={to}
            end={to === '/'}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 mx-1 rounded-lg text-sm transition-colors ${
                isActive
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-300 hover:bg-gray-700 hover:text-white'
              } ${open ? '' : 'justify-center'}`
            }
          >
            <div className="relative flex-shrink-0">
              <Icon className="w-5 h-5" />
              {badge && unreadCount > 0 && (
                <span className="absolute -top-1 -right-1 bg-red-500 text-white text-xs rounded-full w-4 h-4 flex items-center justify-center leading-none">
                  {unreadCount > 9 ? '9+' : unreadCount}
                </span>
              )}
            </div>
            {open && <span>{label}</span>}
          </NavLink>
        ))}

        {user?.role === 'admin' && (
          <NavLink
            to="/users"
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2 mx-1 rounded-lg text-sm transition-colors ${
                isActive
                  ? 'bg-indigo-600 text-white'
                  : 'text-gray-300 hover:bg-gray-700 hover:text-white'
              } ${open ? '' : 'justify-center'}`
            }
          >
            <UsersIcon className="w-5 h-5 flex-shrink-0" />
            {open && <span>Users</span>}
          </NavLink>
        )}
      </nav>
    </aside>
  );
}
