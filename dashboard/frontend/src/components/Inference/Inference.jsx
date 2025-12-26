import { useState } from 'react'
import { motion } from 'framer-motion'
import { Play, RefreshCw, AlertCircle, CheckCircle, Cpu, Zap } from 'lucide-react'

export default function Inference() {
  const [selectedGrid, setSelectedGrid] = useState('ieee24')
  const [selectedTask, setSelectedTask] = useState('cascade')
  const [isLoading, setIsLoading] = useState(false)
  const [result, setResult] = useState(null)

  // Simulated inference
  const runInference = async () => {
    setIsLoading(true)
    setResult(null)

    // Simulate API call
    await new Promise(resolve => setTimeout(resolve, 1500))

    // Mock result
    const mockResults = {
      cascade: {
        prediction: Math.random() > 0.9 ? 1 : 0,
        probability: Math.random() * 0.3 + (Math.random() > 0.9 ? 0.7 : 0),
        latency: Math.floor(Math.random() * 20 + 10),
      },
      powerflow: {
        predictions: Array.from({ length: 24 }, () => ({
          v: 0.95 + Math.random() * 0.1,
          theta: (Math.random() - 0.5) * 0.2,
        })),
        mse: Math.random() * 0.01,
        latency: Math.floor(Math.random() * 30 + 15),
      },
      lineflow: {
        predictions: Array.from({ length: 38 }, () => ({
          p: Math.random() * 100 - 50,
          q: Math.random() * 50 - 25,
        })),
        mse: Math.random() * 0.05,
        latency: Math.floor(Math.random() * 25 + 12),
      },
    }

    setResult(mockResults[selectedTask])
    setIsLoading(false)
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-2xl font-bold mb-2">Live Inference Demo</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Run the trained Physics-SSL model on sample data
        </p>
      </motion.div>

      {/* Configuration */}
      <motion.section
        className="card"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.1 }}
      >
        <h2 className="font-semibold text-lg mb-4">Configuration</h2>

        <div className="grid md:grid-cols-3 gap-6">
          {/* Grid Selection */}
          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-2">
              Power Grid
            </label>
            <div className="flex gap-2">
              {['ieee24', 'ieee118'].map(grid => (
                <button
                  key={grid}
                  onClick={() => setSelectedGrid(grid)}
                  className={`flex-1 py-2 px-4 rounded-lg font-medium transition-all ${
                    selectedGrid === grid
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300'
                  }`}
                >
                  {grid === 'ieee24' ? 'IEEE 24-bus' : 'IEEE 118-bus'}
                </button>
              ))}
            </div>
          </div>

          {/* Task Selection */}
          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-2">
              Prediction Task
            </label>
            <select
              value={selectedTask}
              onChange={(e) => setSelectedTask(e.target.value)}
              className="w-full bg-gray-100 dark:bg-gray-700 rounded-lg px-4 py-2"
            >
              <option value="cascade">Cascade Prediction</option>
              <option value="powerflow">Power Flow</option>
              <option value="lineflow">Line Flow</option>
            </select>
          </div>

          {/* Run Button */}
          <div className="flex items-end">
            <button
              onClick={runInference}
              disabled={isLoading}
              className="w-full btn-primary flex items-center justify-center gap-2 py-3"
            >
              {isLoading ? (
                <>
                  <RefreshCw className="w-5 h-5 animate-spin" />
                  Running...
                </>
              ) : (
                <>
                  <Play className="w-5 h-5" />
                  Run Inference
                </>
              )}
            </button>
          </div>
        </div>
      </motion.section>

      {/* Results */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Input Visualization */}
        <motion.section
          className="card"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <h2 className="font-semibold text-lg mb-4">Input Sample</h2>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6 h-64 flex items-center justify-center">
            <div className="text-center text-gray-500">
              <Cpu className="w-12 h-12 mx-auto mb-3 opacity-50" />
              <p className="text-sm">
                Random sample from {selectedGrid === 'ieee24' ? 'IEEE 24-bus' : 'IEEE 118-bus'} test set
              </p>
              <p className="text-xs mt-2 text-gray-400">
                {selectedGrid === 'ieee24' ? '24 nodes × 6 features' : '118 nodes × 6 features'}
              </p>
            </div>
          </div>
        </motion.section>

        {/* Output Visualization */}
        <motion.section
          className="card"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <h2 className="font-semibold text-lg mb-4">Model Output</h2>
          <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-6 h-64">
            {!result && !isLoading && (
              <div className="h-full flex items-center justify-center text-gray-500">
                <div className="text-center">
                  <Zap className="w-12 h-12 mx-auto mb-3 opacity-50" />
                  <p className="text-sm">Click "Run Inference" to see results</p>
                </div>
              </div>
            )}

            {isLoading && (
              <div className="h-full flex items-center justify-center">
                <div className="text-center">
                  <RefreshCw className="w-12 h-12 mx-auto mb-3 text-blue-500 animate-spin" />
                  <p className="text-sm text-gray-500">Processing...</p>
                </div>
              </div>
            )}

            {result && selectedTask === 'cascade' && (
              <div className="h-full flex flex-col justify-center">
                <div className="text-center">
                  {result.prediction === 1 ? (
                    <AlertCircle className="w-16 h-16 mx-auto mb-4 text-red-500" />
                  ) : (
                    <CheckCircle className="w-16 h-16 mx-auto mb-4 text-green-500" />
                  )}
                  <h3 className="text-2xl font-bold mb-2">
                    {result.prediction === 1 ? 'Cascade Detected' : 'No Cascade'}
                  </h3>
                  <p className="text-gray-500">
                    Confidence: {(result.probability * 100).toFixed(1)}%
                  </p>
                  <p className="text-sm text-gray-400 mt-4">
                    Latency: {result.latency}ms
                  </p>
                </div>
              </div>
            )}

            {result && selectedTask === 'powerflow' && (
              <div className="h-full overflow-y-auto">
                <div className="grid grid-cols-4 gap-2 text-xs">
                  {result.predictions.slice(0, 12).map((pred, i) => (
                    <div key={i} className="bg-white dark:bg-gray-800 p-2 rounded text-center">
                      <p className="font-medium">Bus {i + 1}</p>
                      <p className="text-blue-600">V: {pred.v.toFixed(3)}</p>
                      <p className="text-purple-600">θ: {pred.theta.toFixed(3)}</p>
                    </div>
                  ))}
                </div>
                <p className="text-sm text-gray-500 mt-4 text-center">
                  MSE: {result.mse.toFixed(4)} | Latency: {result.latency}ms
                </p>
              </div>
            )}

            {result && selectedTask === 'lineflow' && (
              <div className="h-full overflow-y-auto">
                <div className="grid grid-cols-3 gap-2 text-xs">
                  {result.predictions.slice(0, 9).map((pred, i) => (
                    <div key={i} className="bg-white dark:bg-gray-800 p-2 rounded text-center">
                      <p className="font-medium">Line {i + 1}</p>
                      <p className="text-green-600">P: {pred.p.toFixed(1)} MW</p>
                      <p className="text-orange-600">Q: {pred.q.toFixed(1)} MVar</p>
                    </div>
                  ))}
                </div>
                <p className="text-sm text-gray-500 mt-4 text-center">
                  MSE: {result.mse.toFixed(4)} | Latency: {result.latency}ms
                </p>
              </div>
            )}
          </div>
        </motion.section>
      </div>

      {/* Model Info */}
      <motion.section
        className="card bg-blue-50 dark:bg-blue-900/20"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
      >
        <h3 className="font-semibold mb-3 text-blue-700 dark:text-blue-300">
          About This Demo
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          This demo simulates running the Physics-SSL pretrained model. In a full deployment,
          the backend would load the trained PyTorch model and perform real inference on the
          selected power grid sample. The model uses a PhysicsGuidedEncoder with 4 layers,
          128 hidden dimensions, and admittance-weighted message passing.
        </p>
        <div className="mt-4 flex gap-4 text-sm">
          <span className="text-gray-500">
            <strong>Model:</strong> PhysicsGuidedEncoder
          </span>
          <span className="text-gray-500">
            <strong>Pretraining:</strong> PF + LF reconstruction
          </span>
          <span className="text-gray-500">
            <strong>Framework:</strong> PyTorch + PyG
          </span>
        </div>
      </motion.section>
    </div>
  )
}
