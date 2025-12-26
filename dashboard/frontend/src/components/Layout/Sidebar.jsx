import { NavLink } from 'react-router-dom'
import {
  Home,
  Boxes,
  BarChart3,
  Database,
  Play,
  Zap
} from 'lucide-react'

const navItems = [
  { path: '/', icon: Home, label: 'Overview' },
  { path: '/architecture', icon: Boxes, label: 'Architecture' },
  { path: '/results', icon: BarChart3, label: 'Results' },
  { path: '/dataset', icon: Database, label: 'Dataset' },
  { path: '/inference', icon: Play, label: 'Live Demo' },
]

export default function Sidebar() {
  return (
    <aside className="w-64 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 flex flex-col">
      {/* Logo / Title */}
      <div className="p-6 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-blue-700 rounded-lg flex items-center justify-center">
            <Zap className="w-6 h-6 text-white" />
          </div>
          <div>
            <h1 className="font-bold text-lg text-gray-900 dark:text-white">Physics-SSL</h1>
            <p className="text-xs text-gray-500 dark:text-gray-400">Power Grid GNN</p>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 p-4">
        <ul className="space-y-2">
          {navItems.map(({ path, icon: Icon, label }) => (
            <li key={path}>
              <NavLink
                to={path}
                className={({ isActive }) =>
                  `nav-link ${isActive ? 'active' : ''}`
                }
              >
                <Icon className="w-5 h-5" />
                <span>{label}</span>
              </NavLink>
            </li>
          ))}
        </ul>
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        <div className="text-xs text-gray-500 dark:text-gray-400 text-center">
          <p>IEEE TPWRS 2025</p>
          <p className="mt-1">Physics-Guided SSL for Power Grids</p>
        </div>
      </div>
    </aside>
  )
}
