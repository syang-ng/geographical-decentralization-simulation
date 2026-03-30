import type { Block } from '../types/blocks'

function renderBlock(block: Block): string {
  switch (block.type) {
    case 'stat':
      return `- ${block.label}: ${block.value}${block.sublabel ? ` (${block.sublabel})` : ''}${block.delta ? ` [${block.delta}]` : ''}`
    case 'insight':
      return `${block.title ? `### ${block.title}\n\n` : ''}${block.text}`
    case 'chart':
      return [
        `### ${block.title}`,
        ...block.data.map(datum => `- ${datum.label}: ${datum.value}${block.unit ?? ''}${datum.category ? ` (${datum.category})` : ''}`),
      ].join('\n')
    case 'comparison':
      return [
        `### ${block.title}`,
        `**${block.left.label}**`,
        ...block.left.items.map(item => `- ${item.key}: ${item.value}`),
        '',
        `**${block.right.label}**`,
        ...block.right.items.map(item => `- ${item.key}: ${item.value}`),
        ...(block.verdict ? ['', `Verdict: ${block.verdict}`] : []),
      ].join('\n')
    case 'table':
      return [
        `### ${block.title}`,
        `| ${block.headers.join(' | ')} |`,
        `| ${block.headers.map(() => '---').join(' | ')} |`,
        ...block.rows.map(row => `| ${row.join(' | ')} |`),
      ].join('\n')
    case 'caveat':
      return `> Caveat: ${block.text}`
    case 'source':
      return [
        '### Sources',
        ...block.refs.map(ref => `- ${ref.label}${ref.section ? ` (${ref.section})` : ''}${ref.url ? ` - ${ref.url}` : ''}`),
      ].join('\n')
    case 'map':
      return [
        `### ${block.title}`,
        ...block.regions.map(region => `- ${region.label ?? region.name}: ${region.value}`),
      ].join('\n')
    case 'timeseries':
      return [
        `### ${block.title}`,
        ...block.series.map(series => {
          const latest = series.data[series.data.length - 1]
          return `- ${series.label}${latest ? `: latest ${latest.y} at x=${latest.x}` : ''}`
        }),
      ].join('\n')
    default:
      return ''
  }
}

export function blocksToMarkdown(title: string, summary: string, blocks: readonly Block[]): string {
  return [
    `# ${title}`,
    '',
    summary,
    '',
    ...blocks.map(renderBlock).flatMap(section => [section, '']),
  ].join('\n').trim()
}
