import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { useState, createContext } from 'react'

import Layout from './components/Layout/Layout'
import Overview from './components/Overview/Overview'
import Architecture from './components/Architecture/Architecture'
import Results from './components/Results/Results'
import Dataset from './components/Dataset/Dataset'
import Inference from './components/Inference/Inference'

// Create contexts
export const ThemeContext = createContext()
export const AudienceContext = createContext()

// Query client for data fetching
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      refetchOnWindowFocus: false,
    },
  },
})

function App() {
  const [darkMode, setDarkMode] = useState(false)
  const [audienceLevel, setAudienceLevel] = useState('researcher') // 'beginner' | 'engineer' | 'researcher'

  const toggleDarkMode = () => {
    setDarkMode(!darkMode)
    document.documentElement.classList.toggle('dark')
  }

  return (
    <QueryClientProvider client={queryClient}>
      <ThemeContext.Provider value={{ darkMode, toggleDarkMode }}>
        <AudienceContext.Provider value={{ audienceLevel, setAudienceLevel }}>
          <Router>
            <div className={`min-h-screen bg-gray-50 dark:bg-gray-900 text-gray-900 dark:text-gray-100 ${darkMode ? 'dark' : ''}`}>
              <Layout>
                <Routes>
                  <Route path="/" element={<Overview />} />
                  <Route path="/architecture" element={<Architecture />} />
                  <Route path="/results" element={<Results />} />
                  <Route path="/dataset" element={<Dataset />} />
                  <Route path="/inference" element={<Inference />} />
                </Routes>
              </Layout>
            </div>
          </Router>
        </AudienceContext.Provider>
      </ThemeContext.Provider>
    </QueryClientProvider>
  )
}

export default App
