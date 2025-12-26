import { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'

// Sample IEEE 24-bus topology with better layout
const ieee24Nodes = [
  { id: 1, type: 'load', x: 80, y: 80 },
  { id: 2, type: 'load', x: 160, y: 60 },
  { id: 3, type: 'load', x: 240, y: 80 },
  { id: 4, type: 'load', x: 320, y: 100 },
  { id: 5, type: 'load', x: 120, y: 150 },
  { id: 6, type: 'load', x: 200, y: 130 },
  { id: 7, type: 'gen', x: 280, y: 150 },
  { id: 8, type: 'load', x: 360, y: 170 },
  { id: 9, type: 'load', x: 100, y: 220 },
  { id: 10, type: 'load', x: 180, y: 200 },
  { id: 11, type: 'load', x: 260, y: 220 },
  { id: 12, type: 'load', x: 340, y: 240 },
  { id: 13, type: 'gen', x: 80, y: 290 },
  { id: 14, type: 'gen', x: 160, y: 270 },
  { id: 15, type: 'gen', x: 240, y: 290 },
  { id: 16, type: 'gen', x: 320, y: 310 },
  { id: 17, type: 'load', x: 400, y: 290 },
  { id: 18, type: 'gen', x: 120, y: 360 },
  { id: 19, type: 'load', x: 200, y: 340 },
  { id: 20, type: 'load', x: 280, y: 360 },
  { id: 21, type: 'gen', x: 360, y: 340 },
  { id: 22, type: 'gen', x: 160, y: 410 },
  { id: 23, type: 'gen', x: 280, y: 430 },
  { id: 24, type: 'load', x: 400, y: 410 },
]

const ieee24Edges = [
  [1, 2], [1, 3], [1, 5], [2, 4], [2, 6], [3, 9], [3, 24], [4, 9], [5, 10],
  [6, 10], [7, 8], [8, 9], [8, 10], [9, 11], [9, 12], [10, 11], [10, 12],
  [11, 13], [11, 14], [12, 13], [12, 23], [13, 23], [14, 16], [15, 16],
  [15, 21], [15, 24], [16, 17], [16, 19], [17, 18], [17, 22], [18, 21],
  [19, 20], [20, 23], [21, 22],
]

// Generate IEEE-118 nodes procedurally with fixed seed for consistency
const generateIEEE118 = () => {
  const nodes = []
  const rows = 10
  const cols = 12

  // Use deterministic positions
  for (let i = 0; i < 118; i++) {
    const row = Math.floor(i / cols)
    const col = i % cols
    // Deterministic "random" offset based on index
    const offsetX = ((i * 7) % 20) - 10
    const offsetY = ((i * 11) % 14) - 7
    nodes.push({
      id: i + 1,
      type: i % 4 === 0 ? 'gen' : 'load', // Deterministic generator placement
      x: 40 + col * 38 + offsetX,
      y: 30 + row * 40 + offsetY,
    })
  }

  // Generate deterministic edges
  const edges = []
  for (let i = 0; i < 118; i++) {
    if (i + 1 < 118 && (i % 3 !== 0)) edges.push([i + 1, i + 2])
    if (i + cols < 118 && (i % 2 === 0)) edges.push([i + 1, i + cols + 1])
  }

  return { nodes, edges: edges.slice(0, 186) }
}

const ieee118Data = generateIEEE118()

export default function NetworkGraph({ grid }) {
  const svgRef = useRef()
  const containerRef = useRef()
  const [tooltip, setTooltip] = useState({ show: false, x: 0, y: 0, content: '' })

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const width = container.clientWidth
    const height = container.clientHeight

    // Clear previous
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)

    const g = svg.append('g')

    // Get data based on grid
    const nodeData = grid === 'ieee24' ? ieee24Nodes : ieee118Data.nodes
    const edgeData = grid === 'ieee24' ? ieee24Edges : ieee118Data.edges

    // Calculate data bounds
    const xExtent = d3.extent(nodeData, d => d.x)
    const yExtent = d3.extent(nodeData, d => d.y)

    // Add padding for nodes at edges
    const padding = grid === 'ieee24' ? 40 : 25

    // Scale to fit with proper margins
    const scaleX = d3.scaleLinear()
      .domain([xExtent[0] - 20, xExtent[1] + 20])
      .range([padding, width - padding])

    const scaleY = d3.scaleLinear()
      .domain([yExtent[0] - 20, yExtent[1] + 20])
      .range([padding, height - padding])

    // Draw edges first (below nodes)
    g.selectAll('.edge')
      .data(edgeData)
      .enter()
      .append('line')
      .attr('class', 'network-edge')
      .attr('x1', d => {
        const node = nodeData.find(n => n.id === d[0])
        return node ? scaleX(node.x) : 0
      })
      .attr('y1', d => {
        const node = nodeData.find(n => n.id === d[0])
        return node ? scaleY(node.y) : 0
      })
      .attr('x2', d => {
        const node = nodeData.find(n => n.id === d[1])
        return node ? scaleX(node.x) : 0
      })
      .attr('y2', d => {
        const node = nodeData.find(n => n.id === d[1])
        return node ? scaleY(node.y) : 0
      })
      .attr('stroke', '#94a3b8')
      .attr('stroke-width', grid === 'ieee24' ? 2 : 1.5)
      .style('opacity', 0.6)

    // Draw nodes
    const nodeRadius = grid === 'ieee24' ? 14 : 8
    const nodes = g.selectAll('.node')
      .data(nodeData)
      .enter()
      .append('g')
      .attr('class', 'network-node')
      .attr('transform', d => `translate(${scaleX(d.x)},${scaleY(d.y)})`)
      .style('cursor', 'pointer')

    // Node circles
    nodes.append('circle')
      .attr('r', nodeRadius)
      .attr('fill', d => d.type === 'gen' ? '#22c55e' : '#3b82f6')
      .attr('stroke', 'white')
      .attr('stroke-width', grid === 'ieee24' ? 2.5 : 1.5)
      .on('mouseover', function(event, d) {
        d3.select(this).attr('r', nodeRadius * 1.3)
        setTooltip({
          show: true,
          x: event.pageX,
          y: event.pageY,
          content: `Bus ${d.id} (${d.type === 'gen' ? 'Generator' : 'Load'})`
        })
      })
      .on('mouseout', function() {
        d3.select(this).attr('r', nodeRadius)
        setTooltip({ show: false, x: 0, y: 0, content: '' })
      })

    // Labels for IEEE-24 only
    if (grid === 'ieee24') {
      nodes.append('text')
        .attr('y', 4)
        .attr('text-anchor', 'middle')
        .attr('fill', 'white')
        .attr('font-size', '9px')
        .attr('font-weight', 'bold')
        .attr('pointer-events', 'none')
        .text(d => d.id)
    }

    // Add legend with better positioning and background
    const legendWidth = 85
    const legendHeight = 60
    const legend = svg.append('g')
      .attr('transform', `translate(${width - legendWidth - 10}, 10)`)

    // Legend background
    legend.append('rect')
      .attr('x', -5)
      .attr('y', -5)
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .attr('fill', 'white')
      .attr('rx', 6)
      .attr('opacity', 0.9)
      .attr('stroke', '#e5e7eb')

    // Generator legend item
    legend.append('circle')
      .attr('cx', 10)
      .attr('cy', 15)
      .attr('r', 7)
      .attr('fill', '#22c55e')

    legend.append('text')
      .attr('x', 24)
      .attr('y', 19)
      .attr('font-size', '11px')
      .attr('fill', '#374151')
      .text('Generator')

    // Load legend item
    legend.append('circle')
      .attr('cx', 10)
      .attr('cy', 38)
      .attr('r', 7)
      .attr('fill', '#3b82f6')

    legend.append('text')
      .attr('x', 24)
      .attr('y', 42)
      .attr('font-size', '11px')
      .attr('fill', '#374151')
      .text('Load')

    // Zoom behavior
    const zoom = d3.zoom()
      .scaleExtent([0.5, 4])
      .on('zoom', (event) => {
        g.attr('transform', event.transform)
      })

    svg.call(zoom)

    // Add zoom reset button behavior (double-click)
    svg.on('dblclick.zoom', () => {
      svg.transition().duration(500).call(zoom.transform, d3.zoomIdentity)
    })

  }, [grid])

  return (
    <div ref={containerRef} className="w-full h-full relative">
      <svg ref={svgRef} className="w-full h-full cursor-grab active:cursor-grabbing" />

      {/* Tooltip */}
      {tooltip.show && (
        <div
          className="fixed bg-gray-900 text-white text-sm px-3 py-2 rounded-lg shadow-lg pointer-events-none z-50"
          style={{
            left: tooltip.x + 10,
            top: tooltip.y - 30,
          }}
        >
          {tooltip.content}
        </div>
      )}
    </div>
  )
}
