import { ExternalLink } from 'lucide-react'

export function Header() {
  return (
    <header className="border-b border-border-subtle">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-4">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 shrink-0 rounded-full bg-accent" />
              <h1 className="text-sm font-semibold text-text-primary truncate">
                Geographical Decentralization in Ethereum
              </h1>
            </div>
            <p className="text-[11px] text-muted mt-0.5 ml-4">
              Yang, Oz, Wu, Zhang (2025) · Interactive Research Explorer
            </p>
          </div>

          <div className="flex items-center gap-2 shrink-0">
            <a
              href="https://arxiv.org/abs/2509.21475"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-muted rounded-md bg-white/5 hover:bg-white/10 transition-colors"
            >
              arXiv
              <ExternalLink className="w-3 h-3" />
            </a>
            <a
              href="https://github.com/syang-ng/geographical-decentralization-simulation"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 px-3 py-1.5 text-xs text-muted rounded-md bg-white/5 hover:bg-white/10 transition-colors"
            >
              GitHub
              <ExternalLink className="w-3 h-3" />
            </a>
          </div>
        </div>
      </div>
    </header>
  )
}
