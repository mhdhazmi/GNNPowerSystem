import { motion } from 'framer-motion'
import { TrendingUp, Target, Shield, Layers } from 'lucide-react'

const metrics = [
  {
    label: 'F1 Improvement',
    value: '+35.5%',
    description: 'IEEE-24 @ 10% labels',
    icon: TrendingUp,
    color: 'from-green-500 to-emerald-600',
  },
  {
    label: 'Peak F1 Score',
    value: '0.994',
    description: 'IEEE-118 cascade detection',
    icon: Target,
    color: 'from-blue-500 to-indigo-600',
  },
  {
    label: 'Variance Reduction',
    value: '-54%',
    description: 'IEEE-118 stability',
    icon: Shield,
    color: 'from-purple-500 to-violet-600',
  },
  {
    label: 'Tasks Tested',
    value: '3',
    description: 'Cascade, PF, Line Flow',
    icon: Layers,
    color: 'from-orange-500 to-red-500',
  },
]

export default function KeyMetrics() {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
      {metrics.map((metric, index) => {
        const Icon = metric.icon
        return (
          <motion.div
            key={metric.label}
            className={`relative overflow-hidden rounded-xl bg-gradient-to-br ${metric.color} p-6 text-white shadow-lg`}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.5, delay: index * 0.1 }}
          >
            <div className="absolute top-0 right-0 p-3 opacity-20">
              <Icon className="w-16 h-16" />
            </div>
            <div className="relative">
              <p className="text-white/80 text-sm font-medium">{metric.label}</p>
              <p className="text-3xl font-bold mt-1">{metric.value}</p>
              <p className="text-white/70 text-xs mt-2">{metric.description}</p>
            </div>
          </motion.div>
        )
      })}
    </div>
  )
}
