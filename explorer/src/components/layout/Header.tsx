export function Header() {
  return (
    <header className="border-b border-border-subtle">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-4">
        <div className="flex items-start justify-between gap-4">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <div className="w-2 h-2 shrink-0 rounded-full bg-accent" />
              <h1 className="text-base font-semibold text-text-primary font-serif truncate">
                Geographical Decentralization in Ethereum
              </h1>
            </div>
            <p className="text-xs text-muted mt-0.5 ml-4">
              Yang, Oz, Wu, Zhang (2025) · Interactive Research Explorer
            </p>
          </div>

          <div className="flex items-center gap-3 shrink-0">
            <a
              href="https://arxiv.org/abs/2509.21475"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-muted hover:text-text-primary transition-colors"
            >
              arXiv
            </a>
            <a
              href="https://github.com/syang-ng/geographical-decentralization-simulation"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-muted hover:text-text-primary transition-colors"
            >
              GitHub
            </a>
          </div>
        </div>
      </div>
    </header>
  )
}
