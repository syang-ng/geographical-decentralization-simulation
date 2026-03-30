export function Footer() {
  return (
    <footer className="border-t border-rule mt-12">
      <div className="max-w-5xl mx-auto px-4 sm:px-6 py-6">
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
          <p className="text-xs text-muted">
            Yang, Oz, Wu, Zhang (2025) · arXiv:2509.21475 · MIT License
          </p>
          <div className="flex gap-4">
            <a
              href="https://arxiv.org/abs/2509.21475"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-muted hover:text-text-primary transition-colors"
            >
              Read Paper
            </a>
            <a
              href="https://github.com/syang-ng/geographical-decentralization-simulation"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-muted hover:text-text-primary transition-colors"
            >
              View Source
            </a>
            <a
              href="https://geo-decentralization.github.io/"
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-muted hover:text-text-primary transition-colors"
            >
              3D Viewer
            </a>
          </div>
        </div>
      </div>
    </footer>
  )
}
