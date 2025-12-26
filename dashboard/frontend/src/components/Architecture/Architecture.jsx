import { motion } from 'framer-motion'
import { Boxes, ArrowRight, Cpu, GitBranch, ArrowDown, Layers, Target } from 'lucide-react'

export default function Architecture() {
  return (
    <div className="space-y-8">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <h1 className="text-2xl font-bold mb-2">Model Architecture</h1>
        <p className="text-gray-600 dark:text-gray-400">
          Explore the PhysicsGuidedEncoder and SSL pretraining pipeline
        </p>
      </motion.div>

      {/* Pipeline Overview */}
      <motion.section
        className="card"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.2 }}
      >
        <h2 className="font-semibold text-lg mb-6">SSL Pipeline Overview</h2>

        <div className="flex items-center justify-between flex-wrap gap-4">
          {/* Step 1: Input */}
          <div className="flex-1 min-w-[200px] p-4 bg-gray-50 dark:bg-gray-700 rounded-lg text-center">
            <div className="w-12 h-12 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center mx-auto mb-3">
              <GitBranch className="w-6 h-6 text-blue-600" />
            </div>
            <h3 className="font-medium">1. Power Grid Graph</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Buses as nodes, branches as edges
            </p>
          </div>

          <ArrowRight className="w-6 h-6 text-gray-400 hidden md:block" />

          {/* Step 2: Masking */}
          <div className="flex-1 min-w-[200px] p-4 bg-gray-50 dark:bg-gray-700 rounded-lg text-center">
            <div className="w-12 h-12 bg-orange-100 dark:bg-orange-900 rounded-full flex items-center justify-center mx-auto mb-3">
              <span className="text-xl">🎭</span>
            </div>
            <h3 className="font-medium">2. Random Masking</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Mask 15% of node features
            </p>
          </div>

          <ArrowRight className="w-6 h-6 text-gray-400 hidden md:block" />

          {/* Step 3: Encoder */}
          <div className="flex-1 min-w-[200px] p-4 bg-gray-50 dark:bg-gray-700 rounded-lg text-center">
            <div className="w-12 h-12 bg-purple-100 dark:bg-purple-900 rounded-full flex items-center justify-center mx-auto mb-3">
              <Cpu className="w-6 h-6 text-purple-600" />
            </div>
            <h3 className="font-medium">3. PhysicsGuidedEncoder</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              4 layers with admittance weighting
            </p>
          </div>

          <ArrowRight className="w-6 h-6 text-gray-400 hidden md:block" />

          {/* Step 4: Reconstruction */}
          <div className="flex-1 min-w-[200px] p-4 bg-gray-50 dark:bg-gray-700 rounded-lg text-center">
            <div className="w-12 h-12 bg-green-100 dark:bg-green-900 rounded-full flex items-center justify-center mx-auto mb-3">
              <Boxes className="w-6 h-6 text-green-600" />
            </div>
            <h3 className="font-medium">4. PF/LF Reconstruction</h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              Predict masked voltages & flows
            </p>
          </div>
        </div>
      </motion.section>

      {/* Encoder Details */}
      <div className="grid md:grid-cols-2 gap-6">
        <motion.div
          className="card"
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3 }}
        >
          <h2 className="font-semibold text-lg mb-4">PhysicsGuidedEncoder</h2>
          <div className="space-y-3">
            <div className="flex justify-between py-2 border-b border-gray-100 dark:border-gray-700">
              <span className="text-gray-600 dark:text-gray-400">Input Dim</span>
              <span className="font-mono">6 (node) / 3 (edge)</span>
            </div>
            <div className="flex justify-between py-2 border-b border-gray-100 dark:border-gray-700">
              <span className="text-gray-600 dark:text-gray-400">Hidden Dim</span>
              <span className="font-mono">128</span>
            </div>
            <div className="flex justify-between py-2 border-b border-gray-100 dark:border-gray-700">
              <span className="text-gray-600 dark:text-gray-400">Num Layers</span>
              <span className="font-mono">4</span>
            </div>
            <div className="flex justify-between py-2 border-b border-gray-100 dark:border-gray-700">
              <span className="text-gray-600 dark:text-gray-400">Conv Type</span>
              <span className="font-mono">PhysicsGuidedConv</span>
            </div>
            <div className="flex justify-between py-2">
              <span className="text-gray-600 dark:text-gray-400">Edge Weighting</span>
              <span className="font-mono">1/|Z| (admittance)</span>
            </div>
          </div>
        </motion.div>

        <motion.div
          className="card"
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.4 }}
        >
          <h2 className="font-semibold text-lg mb-4">SSL Pretext Tasks</h2>
          <div className="space-y-4">
            <div className="p-4 bg-blue-50 dark:bg-blue-900/30 rounded-lg">
              <h3 className="font-medium text-blue-700 dark:text-blue-300">Power Flow (PF)</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Reconstruct bus voltages (V, θ) from masked inputs
              </p>
              <p className="text-xs text-gray-500 mt-2">
                Loss: MSE on voltage magnitude and angle
              </p>
            </div>
            <div className="p-4 bg-green-50 dark:bg-green-900/30 rounded-lg">
              <h3 className="font-medium text-green-700 dark:text-green-300">Line Flow (LF)</h3>
              <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                Reconstruct branch power flows (P, Q) on edges
              </p>
              <p className="text-xs text-gray-500 mt-2">
                Loss: MSE on active and reactive power
              </p>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Detailed Model Forward Pass - ML Audience */}
      <motion.section
        className="card"
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
      >
        <div className="flex items-center gap-2 mb-6">
          <Layers className="w-5 h-5 text-purple-600" />
          <h2 className="font-semibold text-lg">Model Forward Pass</h2>
          <span className="ml-2 text-xs bg-purple-100 dark:bg-purple-900/50 text-purple-700 dark:text-purple-300 px-2 py-0.5 rounded">
            ML Practitioner View
          </span>
        </div>

        {/* Data Flow Diagram */}
        <div className="space-y-4">
          {/* Input Layer */}
          <div className="border border-blue-200 dark:border-blue-800 rounded-lg p-4 bg-blue-50/50 dark:bg-blue-900/20">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="font-medium text-blue-700 dark:text-blue-300 flex items-center gap-2">
                  <span className="w-6 h-6 bg-blue-600 text-white rounded text-xs flex items-center justify-center">1</span>
                  Input Representation
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  Power grid as heterogeneous graph <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">G = (V, E)</code>
                </p>
              </div>
              <div className="text-right text-xs font-mono text-gray-500 space-y-1">
                <div>x ∈ ℝ<sup>N×6</sup></div>
                <div>edge_attr ∈ ℝ<sup>E×3</sup></div>
                <div>edge_index ∈ ℤ<sup>2×E</sup></div>
              </div>
            </div>
            <div className="mt-3 grid grid-cols-2 gap-4 text-xs">
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <span className="font-medium">Node features (6):</span>
                <span className="text-gray-500 ml-1">[P, Q, V, θ, bus_type, zone]</span>
              </div>
              <div className="bg-white dark:bg-gray-800 p-2 rounded">
                <span className="font-medium">Edge features (3):</span>
                <span className="text-gray-500 ml-1">[X, rating, status]</span>
              </div>
            </div>
          </div>

          <div className="flex justify-center">
            <ArrowDown className="w-5 h-5 text-gray-400" />
          </div>

          {/* Node Embedding */}
          <div className="border border-orange-200 dark:border-orange-800 rounded-lg p-4 bg-orange-50/50 dark:bg-orange-900/20">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="font-medium text-orange-700 dark:text-orange-300 flex items-center gap-2">
                  <span className="w-6 h-6 bg-orange-600 text-white rounded text-xs flex items-center justify-center">2</span>
                  Initial Embedding
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  Linear projection to hidden dimension
                </p>
              </div>
              <div className="text-right text-xs font-mono text-gray-500">
                h<sub>0</sub> ∈ ℝ<sup>N×128</sup>
              </div>
            </div>
            <div className="mt-3 bg-white dark:bg-gray-800 p-3 rounded font-mono text-sm">
              <span className="text-purple-600">h₀</span> = <span className="text-blue-600">ReLU</span>(
              <span className="text-green-600">W<sub>embed</sub></span> · x + b)
              <span className="text-gray-400 ml-4">// W ∈ ℝ<sup>6×128</sup></span>
            </div>
          </div>

          <div className="flex justify-center">
            <ArrowDown className="w-5 h-5 text-gray-400" />
          </div>

          {/* Message Passing Layers */}
          <div className="border border-purple-200 dark:border-purple-800 rounded-lg p-4 bg-purple-50/50 dark:bg-purple-900/20">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="font-medium text-purple-700 dark:text-purple-300 flex items-center gap-2">
                  <span className="w-6 h-6 bg-purple-600 text-white rounded text-xs flex items-center justify-center">3</span>
                  PhysicsGuidedConv Layers (×4)
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  Message passing with physics-informed edge weighting
                </p>
              </div>
              <div className="text-right text-xs font-mono text-gray-500">
                h<sub>l</sub> ∈ ℝ<sup>N×128</sup>
              </div>
            </div>

            {/* Message Passing Formula */}
            <div className="mt-3 space-y-3">
              <div className="bg-white dark:bg-gray-800 p-3 rounded font-mono text-sm overflow-x-auto">
                <div className="text-gray-500 text-xs mb-2">// Message aggregation with admittance weighting</div>
                <div>
                  <span className="text-purple-600">m<sub>i</sub></span> =
                  <span className="text-gray-600"> Σ</span><sub>j∈N(i)</sub>
                  <span className="text-orange-600"> w<sub>ij</sub></span> ·
                  <span className="text-blue-600">MLP</span>([h<sub>j</sub> ∥ e<sub>ij</sub>])
                </div>
                <div className="mt-2 text-gray-500 text-xs">
                  where w<sub>ij</sub> = 1/|Z<sub>ij</sub>| (line admittance = inverse impedance)
                </div>
              </div>

              <div className="bg-white dark:bg-gray-800 p-3 rounded font-mono text-sm">
                <div className="text-gray-500 text-xs mb-2">// Node update with residual connection</div>
                <div>
                  <span className="text-purple-600">h<sub>l+1</sub></span> =
                  <span className="text-blue-600"> LayerNorm</span>(
                  <span className="text-green-600">ReLU</span>(
                  <span className="text-purple-600">h<sub>l</sub></span> +
                  <span className="text-purple-600">m<sub>i</sub></span>))
                </div>
              </div>
            </div>

            {/* Layer visualization */}
            <div className="mt-4 flex items-center justify-center gap-2">
              {[1, 2, 3, 4].map((layer) => (
                <div key={layer} className="flex items-center">
                  <div className="w-12 h-12 bg-purple-100 dark:bg-purple-800 rounded-lg flex items-center justify-center text-purple-700 dark:text-purple-200 text-sm font-medium">
                    L{layer}
                  </div>
                  {layer < 4 && <ArrowRight className="w-4 h-4 text-gray-400 mx-1" />}
                </div>
              ))}
            </div>
          </div>

          <div className="flex justify-center">
            <ArrowDown className="w-5 h-5 text-gray-400" />
          </div>

          {/* Node Embeddings Output */}
          <div className="border border-green-200 dark:border-green-800 rounded-lg p-4 bg-green-50/50 dark:bg-green-900/20">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="font-medium text-green-700 dark:text-green-300 flex items-center gap-2">
                  <span className="w-6 h-6 bg-green-600 text-white rounded text-xs flex items-center justify-center">4</span>
                  Node Embeddings
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  Learned representations encoding grid structure + physics
                </p>
              </div>
              <div className="text-right text-xs font-mono text-gray-500">
                Z ∈ ℝ<sup>N×128</sup>
              </div>
            </div>
            <div className="mt-3 bg-white dark:bg-gray-800 p-3 rounded font-mono text-sm">
              <span className="text-green-600">Z</span> = h<sub>4</sub>
              <span className="text-gray-400 ml-4">// Final layer output = node embeddings</span>
            </div>
          </div>

          <div className="flex justify-center">
            <ArrowDown className="w-5 h-5 text-gray-400" />
          </div>

          {/* Task Heads */}
          <div className="border border-red-200 dark:border-red-800 rounded-lg p-4 bg-red-50/50 dark:bg-red-900/20">
            <div className="flex items-start justify-between">
              <div>
                <h3 className="font-medium text-red-700 dark:text-red-300 flex items-center gap-2">
                  <span className="w-6 h-6 bg-red-600 text-white rounded text-xs flex items-center justify-center">5</span>
                  Task-Specific Heads
                </h3>
                <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                  Different heads for SSL pretraining vs downstream tasks
                </p>
              </div>
            </div>

            <div className="mt-4 grid md:grid-cols-3 gap-3">
              {/* Cascade Head */}
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="flex items-center gap-2 mb-2">
                  <Target className="w-4 h-4 text-red-500" />
                  <span className="font-medium text-sm">Cascade Prediction</span>
                </div>
                <div className="font-mono text-xs space-y-1 text-gray-600 dark:text-gray-400">
                  <div>z<sub>graph</sub> = mean(Z)</div>
                  <div>ŷ = σ(MLP(z<sub>graph</sub>))</div>
                  <div className="text-gray-400">→ ℝ<sup>1</sup> (binary)</div>
                </div>
              </div>

              {/* PF Head */}
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="flex items-center gap-2 mb-2">
                  <Cpu className="w-4 h-4 text-blue-500" />
                  <span className="font-medium text-sm">Power Flow (SSL)</span>
                </div>
                <div className="font-mono text-xs space-y-1 text-gray-600 dark:text-gray-400">
                  <div>V̂, θ̂ = MLP(Z)</div>
                  <div>ℒ = MSE(V̂, V) + MSE(θ̂, θ)</div>
                  <div className="text-gray-400">→ ℝ<sup>N×2</sup></div>
                </div>
              </div>

              {/* LF Head */}
              <div className="bg-white dark:bg-gray-800 p-3 rounded">
                <div className="flex items-center gap-2 mb-2">
                  <Boxes className="w-4 h-4 text-green-500" />
                  <span className="font-medium text-sm">Line Flow (SSL)</span>
                </div>
                <div className="font-mono text-xs space-y-1 text-gray-600 dark:text-gray-400">
                  <div>e<sub>ij</sub> = Z<sub>i</sub> ∥ Z<sub>j</sub></div>
                  <div>P̂, Q̂ = MLP(e<sub>ij</sub>)</div>
                  <div className="text-gray-400">→ ℝ<sup>E×2</sup></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Training Procedure Summary */}
        <div className="mt-6 p-4 bg-gray-100 dark:bg-gray-700 rounded-lg">
          <h4 className="font-medium mb-3">Training Procedure</h4>
          <div className="grid md:grid-cols-2 gap-4 text-sm">
            <div>
              <span className="font-medium text-purple-600">Phase 1: SSL Pretraining</span>
              <ul className="mt-1 text-gray-600 dark:text-gray-400 space-y-1 text-xs">
                <li>• Mask 15% of node features randomly</li>
                <li>• Train encoder + PF/LF heads jointly</li>
                <li>• Loss: MSE on masked reconstructions</li>
                <li>• 100 epochs, Adam (lr=1e-3)</li>
              </ul>
            </div>
            <div>
              <span className="font-medium text-green-600">Phase 2: Fine-tuning</span>
              <ul className="mt-1 text-gray-600 dark:text-gray-400 space-y-1 text-xs">
                <li>• Freeze or unfreeze encoder (both work)</li>
                <li>• Train task head (e.g., cascade classifier)</li>
                <li>• Loss: BCE with pos_weight for imbalance</li>
                <li>• 100 epochs, Adam (lr=1e-4)</li>
              </ul>
            </div>
          </div>
        </div>
      </motion.section>

      {/* Key Insight Box */}
      <motion.div
        className="card bg-gradient-to-r from-purple-50 to-blue-50 dark:from-purple-900/20 dark:to-blue-900/20 border border-purple-200 dark:border-purple-800"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.6 }}
      >
        <h3 className="font-semibold text-purple-700 dark:text-purple-300 mb-2">
          Key Insight: Physics-Informed Edge Weighting
        </h3>
        <p className="text-sm text-gray-600 dark:text-gray-400">
          The core innovation is using <strong>admittance (1/|Z|)</strong> as edge weights during message passing.
          In power systems, buses connected by low-impedance lines have stronger electrical coupling and should
          influence each other more during aggregation. This physics prior improves learning efficiency vs. uniform weighting.
        </p>
        <div className="mt-3 font-mono text-sm bg-white dark:bg-gray-800 p-3 rounded">
          w<sub>ij</sub> = 1 / |Z<sub>ij</sub>| = 1 / √(R<sub>ij</sub>² + X<sub>ij</sub>²)
          <span className="text-gray-400 ml-3">// Higher admittance → stronger message weight</span>
        </div>
      </motion.div>
    </div>
  )
}
