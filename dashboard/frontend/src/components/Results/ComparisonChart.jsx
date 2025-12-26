import { useEffect, useRef } from 'react'
import * as d3 from 'd3'

export default function ComparisonChart({ data, grid, task }) {
  const svgRef = useRef()
  const containerRef = useRef()

  useEffect(() => {
    if (!data || Object.keys(data).length === 0) return

    const container = containerRef.current
    const width = container.clientWidth
    const height = container.clientHeight
    const margin = { top: 40, right: 30, bottom: 60, left: 60 }
    const innerWidth = width - margin.left - margin.right
    const innerHeight = height - margin.top - margin.bottom

    // Clear previous content
    d3.select(svgRef.current).selectAll('*').remove()

    const svg = d3.select(svgRef.current)
      .attr('width', width)
      .attr('height', height)

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`)

    // Data preparation
    const labels = Object.keys(data)
    const datasets = [
      { key: 'scratch', label: 'Scratch', color: '#E74C3C' },
      { key: 'ssl', label: 'Physics-SSL', color: '#3498DB' },
    ]

    // Scales
    const x0 = d3.scaleBand()
      .domain(labels)
      .rangeRound([0, innerWidth])
      .paddingInner(0.2)

    const x1 = d3.scaleBand()
      .domain(datasets.map(d => d.key))
      .rangeRound([0, x0.bandwidth()])
      .padding(0.1)

    const y = d3.scaleLinear()
      .domain([0, 1.05])
      .range([innerHeight, 0])

    // Axes
    g.append('g')
      .attr('class', 'axis x-axis')
      .attr('transform', `translate(0,${innerHeight})`)
      .call(d3.axisBottom(x0))
      .selectAll('text')
      .style('font-size', '12px')

    g.append('g')
      .attr('class', 'axis y-axis')
      .call(d3.axisLeft(y).ticks(5).tickFormat(d3.format('.2f')))
      .selectAll('text')
      .style('font-size', '12px')

    // Y-axis label
    g.append('text')
      .attr('transform', 'rotate(-90)')
      .attr('y', -45)
      .attr('x', -innerHeight / 2)
      .attr('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('fill', '#666')
      .text('F1-Score')

    // X-axis label
    g.append('text')
      .attr('x', innerWidth / 2)
      .attr('y', innerHeight + 45)
      .attr('text-anchor', 'middle')
      .style('font-size', '14px')
      .style('fill', '#666')
      .text('Label Fraction')

    // Grid lines
    g.append('g')
      .attr('class', 'grid')
      .call(d3.axisLeft(y).ticks(5).tickSize(-innerWidth).tickFormat(''))
      .selectAll('line')
      .style('stroke', '#e5e5e5')
      .style('stroke-dasharray', '3,3')

    // Bars
    const labelGroups = g.selectAll('.label-group')
      .data(labels)
      .enter()
      .append('g')
      .attr('class', 'label-group')
      .attr('transform', d => `translate(${x0(d)},0)`)

    labelGroups.selectAll('rect')
      .data(label => datasets.map(d => ({
        key: d.key,
        label: d.label,
        color: d.color,
        value: data[label]?.[d.key] || 0,
      })))
      .enter()
      .append('rect')
      .attr('x', d => x1(d.key))
      .attr('y', d => y(d.value))
      .attr('width', x1.bandwidth())
      .attr('height', d => innerHeight - y(d.value))
      .attr('fill', d => d.color)
      .attr('rx', 4)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this).style('opacity', 0.8)
        // Tooltip
        const tooltip = g.append('g')
          .attr('class', 'tooltip-group')
          .attr('transform', `translate(${x0(labels[0]) + x1(d.key) + x1.bandwidth() / 2}, ${y(d.value) - 10})`)

        tooltip.append('rect')
          .attr('x', -40)
          .attr('y', -25)
          .attr('width', 80)
          .attr('height', 25)
          .attr('fill', '#1f2937')
          .attr('rx', 4)

        tooltip.append('text')
          .attr('text-anchor', 'middle')
          .attr('fill', 'white')
          .attr('font-size', '12px')
          .attr('y', -8)
          .text(`${d.label}: ${d.value.toFixed(3)}`)
      })
      .on('mouseout', function() {
        d3.select(this).style('opacity', 1)
        g.selectAll('.tooltip-group').remove()
      })

    // Value labels on bars
    labelGroups.selectAll('.value-label')
      .data(label => datasets.map(d => ({
        key: d.key,
        value: data[label]?.[d.key] || 0,
      })))
      .enter()
      .append('text')
      .attr('class', 'value-label')
      .attr('x', d => x1(d.key) + x1.bandwidth() / 2)
      .attr('y', d => y(d.value) - 5)
      .attr('text-anchor', 'middle')
      .style('font-size', '11px')
      .style('font-weight', '500')
      .style('fill', '#374151')
      .text(d => d.value.toFixed(3))

    // Legend
    const legend = svg.append('g')
      .attr('transform', `translate(${width - 150}, 15)`)

    datasets.forEach((d, i) => {
      const legendItem = legend.append('g')
        .attr('transform', `translate(0, ${i * 22})`)

      legendItem.append('rect')
        .attr('width', 16)
        .attr('height', 16)
        .attr('fill', d.color)
        .attr('rx', 3)

      legendItem.append('text')
        .attr('x', 22)
        .attr('y', 12)
        .style('font-size', '12px')
        .style('fill', '#374151')
        .text(d.label)
    })

  }, [data, grid, task])

  return (
    <div ref={containerRef} className="w-full h-full">
      <svg ref={svgRef} className="w-full h-full" />
    </div>
  )
}
