import { useContext } from 'react'
import { useLocation } from 'react-router-dom'
import { Sun, Moon, Users } from 'lucide-react'
import { ThemeContext, AudienceContext } from '../../App'

const pageTitles = {
  '/': 'Overview',
  '/architecture': 'Model Architecture',
  '/results': 'Experimental Results',
  '/dataset': 'Dataset Explorer',
  '/inference': 'Live Inference Demo',
}

const audienceLevels = [
  { value: 'beginner', label: 'Beginner', description: 'New to ML/Power Systems' },
  { value: 'engineer', label: 'Engineer', description: 'Power systems background' },
  { value: 'researcher', label: 'Researcher', description: 'ML/GNN expert' },
]

export default function Header() {
  const location = useLocation()
  const { darkMode, toggleDarkMode } = useContext(ThemeContext)
  const { audienceLevel, setAudienceLevel } = useContext(AudienceContext)

  const currentTitle = pageTitles[location.pathname] || 'Dashboard'

  return (
    <header className="h-16 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between px-6">
      {/* Page Title */}
      <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
        {currentTitle}
      </h2>

      {/* Right side controls */}
      <div className="flex items-center gap-4">
        {/* Audience Level Selector */}
        <div className="flex items-center gap-2">
          <Users className="w-4 h-4 text-gray-500" />
          <select
            value={audienceLevel}
            onChange={(e) => setAudienceLevel(e.target.value)}
            className="text-sm bg-gray-100 dark:bg-gray-700 border-none rounded-lg px-3 py-1.5 text-gray-700 dark:text-gray-200 focus:ring-2 focus:ring-blue-500"
          >
            {audienceLevels.map(({ value, label }) => (
              <option key={value} value={value}>
                {label}
              </option>
            ))}
          </select>
        </div>

        {/* Dark Mode Toggle */}
        <button
          onClick={toggleDarkMode}
          className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
          aria-label="Toggle dark mode"
        >
          {darkMode ? (
            <Sun className="w-5 h-5 text-yellow-500" />
          ) : (
            <Moon className="w-5 h-5 text-gray-500" />
          )}
        </button>
      </div>
    </header>
  )
}
