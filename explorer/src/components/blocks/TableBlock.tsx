import { cn } from '../../lib/cn'
import type { TableBlock as TableBlockType } from '../../types/blocks'

interface TableBlockProps {
  block: TableBlockType
}

export function TableBlock({ block }: TableBlockProps) {
  const highlightSet = new Set(block.highlight ?? [])

  return (
    <div className="bg-surface border border-border-subtle rounded-xl p-5">
      <h3 className="text-sm font-medium text-text-primary mb-4">
        {block.title}
      </h3>

      <div className="overflow-x-auto -mx-5 px-5">
        <table className="w-full text-xs">
          <thead>
            <tr className="bg-[#0a0a0a]">
              {block.headers.map((header, i) => (
                <th
                  key={i}
                  className="text-left uppercase tracking-wider text-muted font-medium px-3 py-2 first:rounded-tl-md last:rounded-tr-md"
                >
                  {header}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {block.rows.map((row, rowIdx) => (
              <tr
                key={rowIdx}
                className={cn(
                  'border-t border-border-subtle transition-colors hover:bg-white/[0.02]',
                  highlightSet.has(rowIdx) && 'bg-accent-warm/5',
                )}
              >
                {row.map((cell, cellIdx) => (
                  <td
                    key={cellIdx}
                    className="px-3 py-2 text-text-primary/80 tabular-nums"
                  >
                    {cell}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
