import { useState } from 'react'
import { motion } from 'framer-motion'
import { Database, Grid3X3, PieChart, Info, CheckCircle, ZoomIn, Move } from 'lucide-react'
import NetworkGraph from './NetworkGraph'

const datasetStats = {
  ieee24: {
    name: 'IEEE 24-bus RTS',
    shortName: 'IEEE 24-bus',
    nodes: 24,
    edges: 38,
    samples: { train: 8000, val: 1000, test: 1000 },
    cascadeRate: 0.067,
    features: { node: 6, edge: 3 },
    description: 'Reliability Test System - small grid for validation',
  },
  ieee118: {
    name: 'IEEE 118-bus',
    shortName: 'IEEE 118-bus',
    nodes: 118,
    edges: 186,
    samples: { train: 8000, val: 1000, test: 1000 },
    cascadeRate: 0.057,
    features: { node: 6, edge: 3 },
    description: 'Standard benchmark - medium-scale realistic grid',
  },
}

const featureDetails = {
  node: [
    { name: 'P', description: 'Active power injection (MW)', unit: 'MW' },
    { name: 'Q', description: 'Reactive power injection (MVar)', unit: 'MVar' },
    { name: 'V', description: 'Voltage magnitude (p.u.)', unit: 'p.u.' },
    { name: 'θ', description: 'Voltage angle (radians)', unit: 'rad' },
    { name: 'Bus Type', description: 'PQ (1), PV (2), or Slack (3)', unit: 'category' },
    { name: 'Zone', description: 'Control zone identifier', unit: 'integer' },
  ],
  edge: [
    { name: 'X', description: 'Line reactance (p.u.)', unit: 'p.u.' },
    { name: 'Rating', description: 'Thermal limit (MVA)', unit: 'MVA' },
    { name: 'Status', description: 'Line status (1=closed, 0=open)', unit: 'binary' },
  ],
}

export default function Dataset() {
  const [selectedGrid, setSelectedGrid] = useState('ieee24')
  const stats = datasetStats[selectedGrid]

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-2xl font-bold mb-2">Dataset Explorer</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Visualize and explore the IEEE power grid datasets used for training and evaluation
        </p>
      </motion.div>

      {/* Grid Selector with clear selection state */}
      <motion.div
        className="flex flex-col sm:flex-row gap-4"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.1 }}
      >
        {Object.entries(datasetStats).map(([key, grid]) => (
          <button
            key={key}
            onClick={() => setSelectedGrid(key)}
            className={`relative flex items-center gap-3 px-6 py-4 rounded-xl font-medium transition-all border-2 ${
              selectedGrid === key
                ? 'bg-blue-600 text-white border-blue-600 shadow-lg shadow-blue-500/25'
                : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-600'
            }`}
          >
            {/* Selection indicator */}
            {selectedGrid === key && (
              <CheckCircle className="w-5 h-5 text-white" />
            )}
            <div className="text-left">
              <div className="font-semibold">{grid.shortName}</div>
              <div className={`text-xs ${selectedGrid === key ? 'text-blue-100' : 'text-gray-500'}`}>
                {grid.nodes} nodes, {grid.edges} edges
              </div>
            </div>
          </button>
        ))}
      </motion.div>

      {/* Main Content Grid */}
      <div className="grid lg:grid-cols-3 gap-6">
        {/* Network Graph */}
        <motion.div
          className="lg:col-span-2 card"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.2 }}
        >
          <div className="flex items-center justify-between mb-4">
            <h2 className="font-semibold text-lg">
              Grid Topology: <span className="text-blue-600">{stats.name}</span>
            </h2>
            <span className="text-xs text-gray-500 bg-gray-100 dark:bg-gray-700 px-2 py-1 rounded">
              {stats.nodes} buses • {stats.edges} branches
            </span>
          </div>

          <div className="h-96 bg-gray-50 dark:bg-gray-700 rounded-lg overflow-hidden border border-gray-200 dark:border-gray-600">
            <NetworkGraph grid={selectedGrid} key={selectedGrid} />
          </div>

          {/* Interaction hints */}
          <div className="flex items-center gap-6 mt-3 text-xs text-gray-500">
            <span className="flex items-center gap-1">
              <Move className="w-3.5 h-3.5" /> Drag to pan
            </span>
            <span className="flex items-center gap-1">
              <ZoomIn className="w-3.5 h-3.5" /> Scroll to zoom
            </span>
            <span>Double-click to reset view</span>
            <span>Hover nodes for details</span>
          </div>
        </motion.div>

        {/* Statistics Panel */}
        <motion.div
          className="space-y-4"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          {/* Overview Stats */}
          <div className="card">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Database className="w-5 h-5 text-blue-500" />
              Dataset Overview
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Nodes (Buses)</span>
                <span className="font-mono font-medium">{stats.nodes}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Edges (Branches)</span>
                <span className="font-mono font-medium">{stats.edges}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Node Features</span>
                <span className="font-mono font-medium">{stats.features.node}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-gray-600 dark:text-gray-400">Edge Features</span>
                <span className="font-mono font-medium">{stats.features.edge}</span>
              </div>
            </div>
          </div>

          {/* Split Info */}
          <div className="card">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <Grid3X3 className="w-5 h-5 text-green-500" />
              Data Splits
            </h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-600 dark:text-gray-400">Training</span>
                <span className="font-mono font-medium">{stats.samples.train.toLocaleString()}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600 dark:text-gray-400">Validation</span>
                <span className="font-mono font-medium">{stats.samples.val.toLocaleString()}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-600 dark:text-gray-400">Test</span>
                <span className="font-mono font-medium">{stats.samples.test.toLocaleString()}</span>
              </div>
            </div>
            <div className="mt-3 pt-3 border-t border-gray-100 dark:border-gray-700">
              <div className="flex justify-between items-center text-sm">
                <span className="text-gray-500">Total Samples</span>
                <span className="font-mono font-semibold text-blue-600">
                  {(stats.samples.train + stats.samples.val + stats.samples.test).toLocaleString()}
                </span>
              </div>
            </div>
          </div>

          {/* Class Distribution */}
          <div className="card">
            <h3 className="font-semibold mb-4 flex items-center gap-2">
              <PieChart className="w-5 h-5 text-purple-500" />
              Class Distribution
            </h3>
            <div className="space-y-3">
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 bg-green-500 rounded-full" />
                    Non-Cascade
                  </span>
                  <span className="font-mono">{((1 - stats.cascadeRate) * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2.5 bg-gray-200 dark:bg-gray-600 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-green-500 transition-all duration-500"
                    style={{ width: `${(1 - stats.cascadeRate) * 100}%` }}
                  />
                </div>
              </div>
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="flex items-center gap-2">
                    <span className="w-2.5 h-2.5 bg-red-500 rounded-full" />
                    Cascade
                  </span>
                  <span className="font-mono">{(stats.cascadeRate * 100).toFixed(1)}%</span>
                </div>
                <div className="h-2.5 bg-gray-200 dark:bg-gray-600 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-red-500 transition-all duration-500"
                    style={{ width: `${stats.cascadeRate * 100}%` }}
                  />
                </div>
              </div>
            </div>
            <div className="mt-3 p-2 bg-amber-50 dark:bg-amber-900/20 rounded text-xs text-amber-700 dark:text-amber-300">
              Highly imbalanced: only ~{(stats.cascadeRate * 100).toFixed(0)}% positive class
            </div>
          </div>

          {/* Feature Description - Expanded */}
          <div className="card bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800">
            <h3 className="font-semibold mb-3 flex items-center gap-2 text-blue-700 dark:text-blue-300">
              <Info className="w-5 h-5" />
              Feature Descriptions
            </h3>

            {/* Node Features */}
            <div className="mb-4">
              <h4 className="text-sm font-medium text-blue-600 dark:text-blue-400 mb-2">
                Node Features (6)
              </h4>
              <div className="space-y-1.5">
                {featureDetails.node.map((f) => (
                  <div key={f.name} className="flex items-start gap-2 text-xs">
                    <span className="font-mono font-medium text-gray-700 dark:text-gray-300 w-16 shrink-0">
                      {f.name}
                    </span>
                    <span className="text-gray-600 dark:text-gray-400">
                      {f.description}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            {/* Edge Features */}
            <div>
              <h4 className="text-sm font-medium text-blue-600 dark:text-blue-400 mb-2">
                Edge Features (3)
              </h4>
              <div className="space-y-1.5">
                {featureDetails.edge.map((f) => (
                  <div key={f.name} className="flex items-start gap-2 text-xs">
                    <span className="font-mono font-medium text-gray-700 dark:text-gray-300 w-16 shrink-0">
                      {f.name}
                    </span>
                    <span className="text-gray-600 dark:text-gray-400">
                      {f.description}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  )
}
