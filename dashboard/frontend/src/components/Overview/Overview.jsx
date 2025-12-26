import { useContext } from 'react'
import { Link } from 'react-router-dom'
import { motion } from 'framer-motion'
import {
  Zap,
  TrendingUp,
  Shield,
  ArrowRight,
  CheckCircle,
  AlertTriangle,
  Lightbulb
} from 'lucide-react'
import { AudienceContext } from '../../App'
import KeyMetrics from './KeyMetrics'

const fadeInUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
  transition: { duration: 0.5 }
}

export default function Overview() {
  const { audienceLevel } = useContext(AudienceContext)

  return (
    <div className="space-y-8">
      {/* Hero Section */}
      <motion.section
        className="bg-gradient-to-br from-blue-600 via-blue-700 to-indigo-800 rounded-2xl p-8 text-white shadow-xl"
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-start justify-between">
          <div className="max-w-2xl">
            <h1 className="text-3xl font-bold mb-4">
              Physics-Guided Self-Supervised Learning for Power Grid Analysis
            </h1>
            <p className="text-blue-100 text-lg mb-6">
              {audienceLevel === 'beginner' ? (
                "Using the physics of power grids to help AI learn better with less labeled data."
              ) : audienceLevel === 'engineer' ? (
                "A GNN approach that learns grid structure via power flow physics before downstream tasks like cascade prediction."
              ) : (
                "Self-supervised pretraining with physics-based pretext tasks (PF/OPF reconstruction) improves sample efficiency for power grid GNNs."
              )}
            </p>
            <div className="flex gap-4">
              <Link
                to="/results"
                className="inline-flex items-center gap-2 bg-white text-blue-700 px-6 py-3 rounded-lg font-medium hover:bg-blue-50 transition-colors"
              >
                View Results <ArrowRight className="w-4 h-4" />
              </Link>
              <Link
                to="/inference"
                className="inline-flex items-center gap-2 border border-white/30 px-6 py-3 rounded-lg font-medium hover:bg-white/10 transition-colors"
              >
                Try Live Demo
              </Link>
            </div>
          </div>
          <div className="hidden lg:block">
            <div className="w-32 h-32 bg-white/10 rounded-full flex items-center justify-center">
              <Zap className="w-16 h-16 text-yellow-300" />
            </div>
          </div>
        </div>
      </motion.section>

      {/* Key Metrics */}
      <KeyMetrics />

      {/* Problem → Solution → Results Flow */}
      <motion.section
        className="grid md:grid-cols-3 gap-6"
        initial="initial"
        animate="animate"
        variants={{
          animate: { transition: { staggerChildren: 0.1 } }
        }}
      >
        {/* Problem */}
        <motion.div
          className="card border-l-4 border-red-500"
          variants={fadeInUp}
        >
          <div className="flex items-center gap-3 mb-4">
            <AlertTriangle className="w-6 h-6 text-red-500" />
            <h3 className="font-semibold text-lg">The Problem</h3>
          </div>
          <p className="text-gray-600 dark:text-gray-300">
            {audienceLevel === 'beginner' ? (
              "Labeling power grid data is expensive. We need AI that can learn from limited labeled examples."
            ) : audienceLevel === 'engineer' ? (
              "Labeled data for grid events (cascades, contingencies) is scarce. Traditional ML requires many labeled samples."
            ) : (
              "Limited labeled data for critical grid events. Supervised GNNs require extensive annotation effort."
            )}
          </p>
          <ul className="mt-4 space-y-2 text-sm text-gray-500 dark:text-gray-400">
            <li className="flex items-center gap-2">
              <span className="w-2 h-2 bg-red-500 rounded-full" />
              Only 5.7% cascade events in IEEE-118
            </li>
            <li className="flex items-center gap-2">
              <span className="w-2 h-2 bg-red-500 rounded-full" />
              Annotation requires domain expertise
            </li>
            <li className="flex items-center gap-2">
              <span className="w-2 h-2 bg-red-500 rounded-full" />
              GNNs overfit with few labels
            </li>
          </ul>
        </motion.div>

        {/* Solution */}
        <motion.div
          className="card border-l-4 border-blue-500"
          variants={fadeInUp}
        >
          <div className="flex items-center gap-3 mb-4">
            <Lightbulb className="w-6 h-6 text-blue-500" />
            <h3 className="font-semibold text-lg">Our Solution</h3>
          </div>
          <p className="text-gray-600 dark:text-gray-300">
            {audienceLevel === 'beginner' ? (
              "Pretrain the AI to understand power flow physics first, then fine-tune on the actual task."
            ) : audienceLevel === 'engineer' ? (
              "SSL pretraining with physics-guided tasks: reconstruct bus voltages (PF) and line flows (LF) from masked inputs."
            ) : (
              "Physics-guided pretext tasks (PF voltage reconstruction, LF power reconstruction) enable effective representation learning."
            )}
          </p>
          <ul className="mt-4 space-y-2 text-sm text-gray-500 dark:text-gray-400">
            <li className="flex items-center gap-2">
              <span className="w-2 h-2 bg-blue-500 rounded-full" />
              No additional labels needed
            </li>
            <li className="flex items-center gap-2">
              <span className="w-2 h-2 bg-blue-500 rounded-full" />
              Uses physics knowledge (admittance)
            </li>
            <li className="flex items-center gap-2">
              <span className="w-2 h-2 bg-blue-500 rounded-full" />
              Learns meaningful representations
            </li>
          </ul>
        </motion.div>

        {/* Results */}
        <motion.div
          className="card border-l-4 border-green-500"
          variants={fadeInUp}
        >
          <div className="flex items-center gap-3 mb-4">
            <TrendingUp className="w-6 h-6 text-green-500" />
            <h3 className="font-semibold text-lg">Key Results</h3>
          </div>
          <p className="text-gray-600 dark:text-gray-300">
            {audienceLevel === 'beginner' ? (
              "Our method achieves better accuracy with 10x fewer labels compared to training from scratch."
            ) : audienceLevel === 'engineer' ? (
              "+35.5% F1 improvement on IEEE-24 at 10% labels. Robust to load stress, noise, and class imbalance."
            ) : (
              "Significant gains in low-data regime. Physics-SSL outperforms GraphMAE baseline without requiring explicit class weighting."
            )}
          </p>
          <ul className="mt-4 space-y-2 text-sm text-gray-500 dark:text-gray-400">
            <li className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-500" />
              +35.5% F1 on IEEE-24 @ 10% labels
            </li>
            <li className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-500" />
              Robust to class imbalance
            </li>
            <li className="flex items-center gap-2">
              <CheckCircle className="w-4 h-4 text-green-500" />
              54% reduced variance
            </li>
          </ul>
        </motion.div>
      </motion.section>

      {/* Quick Navigation Cards */}
      <section className="grid md:grid-cols-2 lg:grid-cols-4 gap-4">
        <Link to="/architecture" className="card hover:shadow-lg transition-shadow group">
          <h4 className="font-medium mb-2 group-hover:text-blue-600">Model Architecture</h4>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Explore the PhysicsGuidedEncoder and SSL pretraining pipeline
          </p>
        </Link>
        <Link to="/results" className="card hover:shadow-lg transition-shadow group">
          <h4 className="font-medium mb-2 group-hover:text-blue-600">Experimental Results</h4>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Interactive charts comparing SSL vs Scratch across tasks
          </p>
        </Link>
        <Link to="/dataset" className="card hover:shadow-lg transition-shadow group">
          <h4 className="font-medium mb-2 group-hover:text-blue-600">Dataset Explorer</h4>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Visualize IEEE 24-bus and 118-bus power grid topologies
          </p>
        </Link>
        <Link to="/inference" className="card hover:shadow-lg transition-shadow group">
          <h4 className="font-medium mb-2 group-hover:text-blue-600">Live Inference</h4>
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Run cascade prediction on real grid samples
          </p>
        </Link>
      </section>

      {/* Paper Citation */}
      <section className="card bg-gray-100 dark:bg-gray-800">
        <h3 className="font-semibold mb-3">Citation</h3>
        <code className="block bg-white dark:bg-gray-900 p-4 rounded-lg text-sm font-mono text-gray-700 dark:text-gray-300 overflow-x-auto">
          @article&#123;physics_ssl_power_2025,<br />
          &nbsp;&nbsp;title=&#123;Physics-Guided Self-Supervised Learning for Power Grid Analysis&#125;,<br />
          &nbsp;&nbsp;journal=&#123;IEEE Transactions on Power Systems&#125;,<br />
          &nbsp;&nbsp;year=&#123;2025&#125;<br />
          &#125;
        </code>
      </section>
    </div>
  )
}
