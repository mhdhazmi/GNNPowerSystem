import { useEffect, useRef } from 'react'
import * as d3 from 'd3'

const robustnessData = {
  loadStress: {
    factors: [1.0, 1.1, 1.2, 1.3],
    ssl: [0.994, 0.994, 0.990, 0.985],
    scratch: [0.987, 0.987, 0.976, 0.958],
  },
  noise: {
    factors: [0.0, 0.01, 0.05, 0.1],
    ssl: [0.994, 0.994, 0.992, 0.989],
    scratch: [0.987, 0.987, 0.983, 0.969],
  },
  topology: {
    factors: [0.0, 0.05, 0.1, 0.15],
    ssl: [0.994, 0.266, 0.160, 0.121],
    scratch: [0.987, 0.318, 0.180, 0.137],
  },
}

export default function RobustnessChart() {
  const containerRef = useRef()

  useEffect(() => {
    const container = containerRef.current
    if (!container) return

    const totalWidth = container.clientWidth
    const chartWidth = Math.floor((totalWidth - 60) / 3)
    const height = 250
    const margin = { top: 30, right: 20, bottom: 50, left: 50 }

    // Clear previous
    d3.select(container).selectAll('*').remove()

    const charts = [
      { key: 'loadStress', title: 'Load Stress', xLabel: 'Load Factor' },
      { key: 'noise', title: 'Measurement Noise', xLabel: 'Noise σ' },
      { key: 'topology', title: 'Topology Dropout', xLabel: 'Dropout Rate' },
    ]

    charts.forEach((chart, idx) => {
      const data = robustnessData[chart.key]
      const svg = d3.select(container)
        .append('svg')
        .attr('width', chartWidth)
        .attr('height', height)
        .style('display', 'inline-block')

      const g = svg.append('g')
        .attr('transform', `translate(${margin.left},${margin.top})`)

      const innerWidth = chartWidth - margin.left - margin.right
      const innerHeight = height - margin.top - margin.bottom

      // Scales
      const x = d3.scaleLinear()
        .domain(d3.extent(data.factors))
        .range([0, innerWidth])

      const yMin = chart.key === 'topology' ? 0 : 0.9
      const y = d3.scaleLinear()
        .domain([yMin, 1.02])
        .range([innerHeight, 0])

      // Grid lines
      g.append('g')
        .attr('class', 'grid')
        .call(d3.axisLeft(y).ticks(5).tickSize(-innerWidth).tickFormat(''))
        .selectAll('line')
        .style('stroke', '#e5e5e5')
        .style('stroke-dasharray', '2,2')

      // Axes
      g.append('g')
        .attr('transform', `translate(0,${innerHeight})`)
        .call(d3.axisBottom(x).ticks(4))
        .selectAll('text')
        .style('font-size', '10px')

      g.append('g')
        .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.2f')))
        .selectAll('text')
        .style('font-size', '10px')

      // Title
      svg.append('text')
        .attr('x', chartWidth / 2)
        .attr('y', 18)
        .attr('text-anchor', 'middle')
        .style('font-size', '13px')
        .style('font-weight', '600')
        .style('fill', '#374151')
        .text(chart.title)

      // X-axis label
      g.append('text')
        .attr('x', innerWidth / 2)
        .attr('y', innerHeight + 40)
        .attr('text-anchor', 'middle')
        .style('font-size', '11px')
        .style('fill', '#6b7280')
        .text(chart.xLabel)

      // Line generators
      const line = d3.line()
        .x((d, i) => x(data.factors[i]))
        .y(d => y(d))
        .curve(d3.curveMonotoneX)

      // SSL line
      g.append('path')
        .datum(data.ssl)
        .attr('fill', 'none')
        .attr('stroke', '#3498DB')
        .attr('stroke-width', 2.5)
        .attr('d', line)

      // Scratch line
      g.append('path')
        .datum(data.scratch)
        .attr('fill', 'none')
        .attr('stroke', '#E74C3C')
        .attr('stroke-width', 2.5)
        .attr('stroke-dasharray', '5,5')
        .attr('d', line)

      // SSL dots
      g.selectAll('.ssl-dot')
        .data(data.ssl)
        .enter()
        .append('circle')
        .attr('cx', (d, i) => x(data.factors[i]))
        .attr('cy', d => y(d))
        .attr('r', 5)
        .attr('fill', '#3498DB')
        .attr('stroke', 'white')
        .attr('stroke-width', 2)

      // Scratch dots
      g.selectAll('.scratch-dot')
        .data(data.scratch)
        .enter()
        .append('circle')
        .attr('cx', (d, i) => x(data.factors[i]))
        .attr('cy', d => y(d))
        .attr('r', 5)
        .attr('fill', '#E74C3C')
        .attr('stroke', 'white')
        .attr('stroke-width', 2)

      // Class rate line for topology dropout
      if (chart.key === 'topology') {
        g.append('line')
          .attr('x1', 0)
          .attr('x2', innerWidth)
          .attr('y1', y(0.057))
          .attr('y2', y(0.057))
          .attr('stroke', '#9ca3af')
          .attr('stroke-dasharray', '3,3')

        g.append('text')
          .attr('x', innerWidth - 5)
          .attr('y', y(0.057) - 5)
          .attr('text-anchor', 'end')
          .style('font-size', '9px')
          .style('fill', '#9ca3af')
          .text('class rate')
      }

      // Legend (only on first chart)
      if (idx === 0) {
        const legend = g.append('g')
          .attr('transform', `translate(${innerWidth - 70}, 0)`)

        legend.append('line')
          .attr('x1', 0).attr('x2', 20)
          .attr('y1', 5).attr('y2', 5)
          .attr('stroke', '#3498DB')
          .attr('stroke-width', 2.5)

        legend.append('text')
          .attr('x', 25).attr('y', 9)
          .style('font-size', '10px')
          .text('SSL')

        legend.append('line')
          .attr('x1', 0).attr('x2', 20)
          .attr('y1', 22).attr('y2', 22)
          .attr('stroke', '#E74C3C')
          .attr('stroke-width', 2.5)
          .attr('stroke-dasharray', '5,5')

        legend.append('text')
          .attr('x', 25).attr('y', 26)
          .style('font-size', '10px')
          .text('Scratch')
      }
    })

  }, [])

  return (
    <div ref={containerRef} className="w-full" style={{ minHeight: '250px' }} />
  )
}
