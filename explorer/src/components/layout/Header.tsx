import { GlobeWireframe } from '../decorative/GlobeWireframe'

export function Header() {
  return (
    <header className="border-b border-border-subtle overflow-hidden geo-accent-bar">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-4 relative">
        {/* Decorative globe wireframe — right side watermark */}
        <GlobeWireframe className="absolute right-0 top-1/2 -translate-y-1/2 opacity-60 pointer-events-none hidden sm:block" />

        <div className="flex items-start justify-between gap-4 relative z-10">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 shrink-0 rounded-full bg-accent dot-pulse" />
              <h1 className="text-base sm:text-lg font-semibold text-text-primary font-serif leading-tight">
                Geographical Decentralization in Ethereum
              </h1>
            </div>
            <p className="text-xs text-muted mt-1 ml-4">
              Yang, Oz, Wu, Zhang (2025) · Interactive Research Explorer
            </p>
            <p className="text-[10px] text-text-faint mt-0.5 ml-4 font-mono tracking-wide hidden sm:block">
              40 GCP regions · 7 macro-regions · SSP vs MSP
            </p>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <a
              href="https://arxiv.org/abs/2509.21475"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-muted hover:text-accent transition-colors"
            >
              arXiv
            </a>
            <a
              href="https://github.com/syang-ng/geographical-decentralization-simulation"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-muted hover:text-accent transition-colors"
            >
              GitHub
            </a>
          </div>
        </div>
      </div>
    </header>
  )
}
