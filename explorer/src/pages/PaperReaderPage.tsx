import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { ArrowUpRight, Eye, EyeOff, Link2, Quote } from 'lucide-react'
import { BlockCanvas } from '../components/explore/BlockCanvas'
import { cn } from '../lib/cn'
import { SPRING, SPRING_SOFT } from '../lib/theme'
import { PAPER_METADATA, PAPER_SECTIONS } from '../data/paper-sections'

interface PaperNarrative {
  readonly lede: string
  readonly paragraphs: readonly string[]
  readonly pullQuote: string
  readonly figureCaption: string
}

const PAPER_NARRATIVE: Record<string, PaperNarrative> = {
  'system-model': {
    lede: 'The paper starts from a simple but consequential premise: geography is part of the protocol once latency affects value capture and consensus.',
    paragraphs: [
      'SSP and MSP expose different latency-critical paths, but both transform regional network position into economic advantage. In SSP, a proposer wants fast access to the best relay while also keeping relay-to-attester propagation tight enough to satisfy the attestation threshold. In MSP, the proposer wants to sit where value from many sources accumulates while still remaining close enough to attesters to finalize in time.',
      'That turns validator placement into a geographic game. The paper frames this as a tension between value capture and quorum reachability, and that framing matters because it explains why the same infrastructure change can help one paradigm and hurt the other.',
    ],
    pullQuote: 'The mechanism differs, but the pressure is the same: latency becomes an allocation rule for where validators want to live.',
    figureCaption: 'The core comparison is the latency path itself: SSP optimizes a best relay path, while MSP optimizes over many direct information inputs.',
  },
  'simulation-design': {
    lede: 'The simulation is deliberately simplified, but it is simplified in a way that makes the causal story easy to inspect.',
    paragraphs: [
      'Validators are agents that repeatedly compare expected rewards across measured cloud regions, then migrate if the gain exceeds switching cost. That design keeps the paper close to a geographic equilibrium story rather than a one-off optimization snapshot.',
      'The costs of that clarity are explicit. MEV is modeled as deterministic and linear in latency, migration cost is fixed, and information is complete. Those assumptions make the engine more interpretable, but the paper is careful to treat them as modeling limits rather than claims about production Ethereum.',
    ],
    pullQuote: 'This is a paper about structural pressure, not about reproducing every empirical detail of block production.',
    figureCaption: 'The simulation design is intentionally legible: 40 measured regions, 1,000 validators, and paper-facing runs commonly reported over 10,000 slots under bounded modeling assumptions.',
  },
  'baseline-results': {
    lede: 'Under the homogeneous starting baseline, both paradigms centralize. The interesting part is how differently they get there.',
    paragraphs: [
      'MSP moves faster and ends more concentrated in the baseline runs. The paper attributes that to the additive nature of local block building: value can accumulate from many distributed sources, so the optimization landscape rewards locations that sit at the overlap between source proximity and attester reachability.',
      'SSP still centralizes, but the locus is shaped by relay geography and the proposer-relay-attester chain. That makes the final map look different even when the underlying force is still latency-driven concentration.',
    ],
    pullQuote: 'Baseline results matter here because they show centralization without needing exotic assumptions.',
    figureCaption: 'The baseline comparison sets the tone for the rest of the paper: MSP is more aggressive in the default geography, SSP is more path-dependent.',
  },
  'se1-source-placement': {
    lede: 'Infrastructure placement is not a neutral background condition. It changes the shape of the optimization problem itself.',
    paragraphs: [
      'The striking result in SE1 is not just that source placement matters, but that aligned and misaligned placements invert the severity of centralization depending on the paradigm. MSP benefits from aligned source placement because value capture and consensus pressure pull in the same direction.',
      'SSP behaves differently because badly placed relays create a stronger co-location premium. When the relay path is the bottleneck, shaving proposer-relay latency becomes disproportionately valuable, so misalignment can make concentration worse instead of better.',
    ],
    pullQuote: 'The same geography can be stabilizing in one paradigm and destabilizing in the other.',
    figureCaption: 'SE1 is the cleanest demonstration that the paper is not merely comparing two labels; it is comparing two different latency geometries.',
  },
  'se2-distribution': {
    lede: 'The paper then asks a harder question: what if the system is already geographically unequal before agents start moving?',
    paragraphs: [
      'Using a more realistic validator distribution shifts the interpretation of the results. Once the starting state is already concentrated in the US and Europe, both paradigms converge quickly because the system begins near the eventual attractor.',
      'That result is important for the website because it keeps the narrative honest. Paradigm choice matters, but initial conditions can dominate. The model is not claiming a single mechanism explains all observed concentration on its own.',
    ],
    pullQuote: 'If the system starts centralized, the paradigm mostly changes how the imbalance amplifies, not whether it exists.',
    figureCaption: 'SE2 reframes the story from "which paradigm centralizes more?" to "how much of the outcome was already baked into the starting distribution?"',
  },
  'se3-joint': {
    lede: 'Joint heterogeneity is where the paper briefly finds something that looks like relief, then carefully refuses to overclaim it.',
    paragraphs: [
      'In the combined heterogeneous case, the temporary dip in concentration appears when SSP starts from today’s concentrated validator geography and relay placement is poorly connected to that start. That makes the trajectory visually unusual because it is one of the only times the model briefly moves away from concentration rather than further into it.',
      'But the paper treats that as a temporary artifact of competing geographic pulls, not a recipe for decentralization. That caution is a good editorial anchor for the whole reader experience: the goal is to diagnose pressures, not to manufacture optimistic takeaways.',
    ],
    pullQuote: 'A temporary dip in Gini is not the same thing as a decentralization mechanism.',
    figureCaption: 'SE3 is best read as a warning against overinterpreting transient trajectories as stable system improvements.',
  },
  'se4a-attestation': {
    lede: 'SE4a is the paper\'s signature result because it shows the same protocol parameter producing opposite geographic effects across paradigms.',
    paragraphs: [
      'Raising the attestation threshold makes SSP centralize more because the relay path becomes more timing-sensitive. The proposer gains more by clustering tightly around the relay geography that minimizes end-to-end delay.',
      'In MSP, a higher threshold forces a harder compromise between being close to attesters and being close to information sources. Those geographic objectives do not perfectly coincide, so stronger timing pressure can actually disperse the equilibrium rather than compress it.',
    ],
    pullQuote: 'The most surprising result in the paper is also the most revealing: timing rules are not paradigm-neutral.',
    figureCaption: 'Attestation threshold is where the paper most clearly shows that "faster consensus" and "more centralization" do not move identically in SSP and MSP.',
  },
  'se4b-slots': {
    lede: 'Shorter slots do less to change where validators end up than to change how unevenly rewards are distributed on the way there.',
    paragraphs: [
      'The paper finds that moving to 6-second slots leaves the broad geographic equilibrium largely intact. The same regions remain attractive, and the same concentration tendencies persist.',
      'What changes is reward variance. When the slot is shorter, a fixed latency advantage consumes a bigger fraction of the available timing budget. That raises the penalty for being outside the favored corridors even if the final map does not change dramatically.',
    ],
    pullQuote: 'Shorter slots amplify inequality faster than they rewrite the geography.',
    figureCaption: 'The slot-time experiment is a reminder that not every protocol change moves the concentration map, but many still change who gets paid.',
  },
  discussion: {
    lede: 'The discussion section is diagnostic rather than prescriptive, and that is the right tone to preserve in the UI.',
    paragraphs: [
      'The paper sketches mitigation directions such as rewarding underrepresented regions, decentralizing relays and sources, or compensating for latency at the protocol layer. But none of these are presented as settled policy recommendations.',
      'That restraint matters. The contribution is to show that geographic concentration is endogenous to the timing structure of the system, not to claim the model has already solved how to counteract it.',
    ],
    pullQuote: 'The strongest claim here is about diagnosis: the protocol and infrastructure together create concentration pressure.',
    figureCaption: 'Mitigation ideas are included as design directions, not as recommendations validated by this model.',
  },
  limitations: {
    lede: 'The limitations section is one of the most important parts of the paper because it defines where confidence should stop.',
    paragraphs: [
      'Every simplification in the model trades realism for tractability: cloud-only latency, deterministic MEV, full information, fixed migration cost, and no strategic coalition behavior. Those assumptions make the simulations readable and comparable, but they also bound what can be claimed.',
      'For the website, this section should remain close to the end of the reading flow rather than hidden behind a footnote. It keeps the project aligned with the researchers\' intent: truth first, then interpretation.',
    ],
    pullQuote: 'A good research interface should make the caveats feel structural, not optional.',
    figureCaption: 'The limitations list is part of the paper\'s core meaning, not an appendix to ignore.',
  },
}

export function PaperReaderPage() {
  const [readerMode, setReaderMode] = useState<'editorial' | 'focus'>(() => {
    const stored = window.localStorage.getItem('paper-reader-mode')
    return stored === 'focus' ? 'focus' : 'editorial'
  })
  const [activeSectionId, setActiveSectionId] = useState<string>(() => {
    const initialHash = window.location.hash.replace('#', '')
    return PAPER_SECTIONS.some(section => section.id === initialHash)
      ? initialHash
      : PAPER_SECTIONS[0].id
  })
  const [copiedSectionId, setCopiedSectionId] = useState<string | null>(null)
  const focusMode = readerMode === 'focus'

  useEffect(() => {
    window.localStorage.setItem('paper-reader-mode', readerMode)
  }, [readerMode])

  useEffect(() => {
    const initialHash = window.location.hash.replace('#', '')
    if (!initialHash) return

    const target = document.getElementById(initialHash)
    if (!target) return

    const raf = window.requestAnimationFrame(() => {
      target.scrollIntoView({ block: 'start', behavior: 'smooth' })
    })

    return () => window.cancelAnimationFrame(raf)
  }, [])

  useEffect(() => {
    const sections = PAPER_SECTIONS
      .map(section => document.getElementById(section.id))
      .filter((element): element is HTMLElement => element instanceof HTMLElement)

    if (sections.length === 0) return

    const observer = new IntersectionObserver(
      entries => {
        const visible = entries
          .filter(entry => entry.isIntersecting)
          .sort((left, right) => right.intersectionRatio - left.intersectionRatio)

        if (visible[0]?.target.id) {
          setActiveSectionId(visible[0].target.id)
        }
      },
      {
        rootMargin: '-22% 0px -55% 0px',
        threshold: [0.15, 0.35, 0.6],
      },
    )

    sections.forEach(section => observer.observe(section))
    return () => observer.disconnect()
  }, [])

  useEffect(() => {
    if (!activeSectionId) return
    const url = new URL(window.location.href)
    url.hash = activeSectionId
    window.history.replaceState({}, '', url.toString())
  }, [activeSectionId])

  const activeSectionIndex = Math.max(
    0,
    PAPER_SECTIONS.findIndex(section => section.id === activeSectionId),
  )
  const progressPercent = ((activeSectionIndex + 1) / PAPER_SECTIONS.length) * 100
  const activeSection = PAPER_SECTIONS.find(section => section.id === activeSectionId) ?? PAPER_SECTIONS[0]

  const handleCopySectionLink = async (sectionId: string) => {
    const url = new URL(window.location.href)
    url.hash = sectionId
    try {
      await navigator.clipboard.writeText(url.toString())
      setCopiedSectionId(sectionId)
      window.setTimeout(() => {
        setCopiedSectionId(current => (current === sectionId ? null : current))
      }, 1600)
    } catch {
      // Ignore clipboard failures
    }
  }

  return (
    <div className="space-y-12">
      {/* Paper title hero */}
      <motion.section
        initial={{ opacity: 0, y: 18 }}
        animate={{ opacity: 1, y: 0 }}
        transition={SPRING_SOFT}
      >
        <div className="grid gap-8 xl:grid-cols-[minmax(0,1fr)_300px]">
          <div className="max-w-4xl">
            <h1 className="text-3xl font-medium leading-tight text-text-primary font-serif sm:text-4xl">
              {PAPER_METADATA.title}
            </h1>
            <p className="mt-4 max-w-3xl text-base leading-relaxed text-muted font-serif">
              {PAPER_METADATA.abstract}
            </p>

            <div className="mt-6 flex flex-wrap gap-2">
              {PAPER_METADATA.keyClaims.map(claim => (
                <span
                  key={claim}
                  className="rounded-md border border-border-subtle px-3 py-1.5 text-xs text-text-primary"
                >
                  {claim}
                </span>
              ))}
            </div>
          </div>

          <div className="border border-border-subtle rounded-lg p-5">
            <div className="flex items-center justify-between gap-3">
              <span className="text-xs text-muted">Reading guide</span>
              <button
                onClick={() => setReaderMode(current => (current === 'editorial' ? 'focus' : 'editorial'))}
                className="inline-flex items-center gap-1.5 text-xs text-muted transition-colors hover:text-text-primary"
              >
                {focusMode ? <EyeOff className="h-3 w-3" /> : <Eye className="h-3 w-3" />}
                {focusMode ? 'Editorial mode' : 'Focus mode'}
              </button>
            </div>
            <div className="mt-3 text-sm text-text-primary">{PAPER_METADATA.citation}</div>

            <div className="mt-5 space-y-3 border-t border-border-subtle pt-4">
              <div>
                <div className="text-xs text-muted">Start with</div>
                <div className="mt-1 text-sm font-medium text-text-primary">SE4a attestation threshold</div>
                <div className="mt-1 text-xs text-muted">The same protocol lever pushes SSP and MSP in opposite geographic directions.</div>
              </div>
              <div>
                <div className="text-xs text-muted">Then test</div>
                <div className="mt-1 text-sm font-medium text-text-primary">SE2 starting geography</div>
                <div className="mt-1 text-xs text-muted">How much of the result is already baked into the real Ethereum map?</div>
              </div>
              <div>
                <div className="text-xs text-muted">Truth boundary</div>
                <div className="mt-1 text-sm font-medium text-text-primary">
                  {focusMode ? 'Focused, paper-backed' : 'Editorial, paper-backed'}
                </div>
              </div>
            </div>

            <div className="mt-5 space-y-2 border-t border-border-subtle pt-4">
              {PAPER_METADATA.references.map(reference => (
                <a
                  key={reference.label}
                  href={reference.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="flex items-center justify-between rounded-md border border-border-subtle px-3 py-2 text-sm text-text-primary transition-colors hover:border-border-hover"
                >
                  <span>{reference.label}</span>
                  <ArrowUpRight className="h-3.5 w-3.5 text-muted" />
                </a>
              ))}
            </div>
          </div>
        </div>
      </motion.section>

      <hr className="border-rule" />

      <div className={cn('grid gap-8', focusMode ? 'xl:grid-cols-[minmax(0,1fr)]' : 'xl:grid-cols-[220px_minmax(0,1fr)]')}>
        {/* TOC sidebar */}
        {!focusMode && (
          <aside className="xl:sticky xl:top-28 xl:self-start">
          <div className="border border-border-subtle rounded-lg p-4">
            <span className="text-xs text-muted">Contents</span>
            <div className="mt-4 border border-border-subtle rounded-md p-3">
              <div className="flex items-center justify-between gap-3 text-xs text-muted">
                <span>Reading progress</span>
                <span>{activeSectionIndex + 1} / {PAPER_SECTIONS.length}</span>
              </div>
              <div className="mt-2 h-1 overflow-hidden rounded-full bg-[#E8E8E6]">
                <motion.div
                  className="h-full rounded-full bg-accent"
                  animate={{ width: `${progressPercent}%` }}
                  transition={SPRING_SOFT}
                />
              </div>
            </div>
            <div className="mt-4 border border-border-subtle rounded-md p-3">
              <div className="text-xs text-muted">Best entries</div>
              <div className="mt-2 space-y-1">
                <a
                  href="#se4a-attestation"
                  onClick={() => setActiveSectionId('se4a-attestation')}
                  className="block text-sm text-text-primary transition-colors hover:text-accent"
                >
                  Paradox: gamma flips by paradigm
                </a>
                <a
                  href="#se2-distribution"
                  onClick={() => setActiveSectionId('se2-distribution')}
                  className="block text-sm text-text-primary transition-colors hover:text-accent"
                >
                  Starting state can dominate
                </a>
                <a
                  href="#limitations"
                  onClick={() => setActiveSectionId('limitations')}
                  className="block text-sm text-text-primary transition-colors hover:text-accent"
                >
                  Confidence boundary
                </a>
              </div>
            </div>
            <nav className="mt-4 space-y-1">
              {PAPER_SECTIONS.map(section => (
                <a
                  key={section.id}
                  href={`#${section.id}`}
                  onClick={() => setActiveSectionId(section.id)}
                  className={cn(
                    'block rounded-md px-3 py-2 text-sm transition-colors',
                    activeSectionId === section.id
                      ? 'bg-surface-active text-text-primary'
                      : 'text-muted hover:bg-surface-active hover:text-text-primary',
                  )}
                >
                  <div className={cn(
                    'text-xs',
                    activeSectionId === section.id ? 'text-accent' : 'text-muted',
                  )}>
                    {section.number}
                  </div>
                  <div className="mt-0.5 leading-snug">{section.title}</div>
                </a>
              ))}
            </nav>
          </div>
          </aside>
        )}

        <div className="space-y-12">
          {/* Focus mode sticky bar */}
          {focusMode && (
            <div className="sticky top-28 z-10 rounded-lg border border-border-subtle bg-white px-4 py-3">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <div className="text-xs text-muted">Now reading</div>
                  <div className="mt-1 text-sm text-text-primary">
                    {activeSection.number} {activeSection.title}
                  </div>
                </div>
                <div className="flex items-center gap-3">
                  <div className="h-1 w-32 overflow-hidden rounded-full bg-[#E8E8E6]">
                    <motion.div
                      className="h-full rounded-full bg-accent"
                      animate={{ width: `${progressPercent}%` }}
                      transition={SPRING_SOFT}
                    />
                  </div>
                  <button
                    onClick={() => handleCopySectionLink(activeSection.id)}
                    className="inline-flex items-center gap-1.5 text-xs text-muted transition-colors hover:text-text-primary"
                  >
                    <Link2 className="h-3 w-3" />
                    {copiedSectionId === activeSection.id ? 'Copied' : 'Copy link'}
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* Paper sections */}
          {PAPER_SECTIONS.map((section, index) => {
            const narrative = PAPER_NARRATIVE[section.id]
            const figuresFirst = index % 2 === 1
            const previousSection = PAPER_SECTIONS[index - 1]
            const nextSection = PAPER_SECTIONS[index + 1]

            return (
              <motion.section
                key={section.id}
                id={section.id}
                initial={{ opacity: 0, y: 18 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, amount: 0.2 }}
                transition={SPRING}
                className={cn(
                  'scroll-mt-28 border border-border-subtle rounded-lg bg-white p-5 sm:p-6',
                  focusMode && 'mx-auto max-w-5xl',
                )}
              >
                {/* Section header */}
                <div className="mb-6 flex flex-wrap items-end justify-between gap-4 border-b border-border-subtle pb-5">
                  <div className={cn('max-w-2xl', focusMode && 'max-w-3xl')}>
                    <div className="text-xs text-accent">
                      {section.number}
                    </div>
                    <h2 className="mt-2 text-2xl font-medium text-text-primary font-serif sm:text-3xl">
                      {section.title}
                    </h2>
                    <p className="mt-3 text-base leading-relaxed text-muted">
                      {section.description}
                    </p>
                  </div>

                  <div className="border border-border-subtle rounded-md px-4 py-3 text-sm text-muted">
                    <div className="text-xs text-muted">Reading note</div>
                    <div className="mt-1 max-w-xs text-text-primary">
                      {narrative.figureCaption}
                    </div>
                    <button
                      onClick={() => handleCopySectionLink(section.id)}
                      className="mt-3 inline-flex items-center gap-1.5 text-xs text-muted transition-colors hover:text-text-primary"
                    >
                      <Link2 className="h-3 w-3" />
                      {copiedSectionId === section.id ? 'Copied' : 'Copy section link'}
                    </button>
                  </div>
                </div>

                {/* Content grid */}
                <div className={cn('grid gap-6', focusMode ? 'xl:grid-cols-[minmax(0,1fr)]' : 'xl:grid-cols-12')}>
                  <div className={cn(focusMode ? 'space-y-5' : 'xl:col-span-7 space-y-5', figuresFirst && 'xl:order-2')}>
                    <p className={cn('text-xl leading-relaxed text-text-primary font-serif', focusMode ? 'max-w-3xl text-[1.65rem]' : 'max-w-2xl')}>
                      {narrative.lede}
                    </p>

                    <div className={cn('space-y-4 text-[15px] text-text-body font-serif', focusMode ? 'max-w-3xl text-base leading-9' : 'leading-8')}>
                      {narrative.paragraphs.map(paragraph => (
                        <p key={paragraph} className={cn(focusMode ? 'max-w-3xl' : 'max-w-2xl')}>
                          {paragraph}
                        </p>
                      ))}
                    </div>

                    {/* Pull quote */}
                    <div className="border-l-[3px] border-l-accent pl-5 py-2">
                      <div className="flex items-center gap-1.5 text-xs text-muted mb-2">
                        <Quote className="h-3 w-3" />
                        Pull quote
                      </div>
                      <p className={cn('leading-relaxed text-text-primary font-serif italic', focusMode ? 'max-w-3xl text-xl' : 'max-w-2xl text-lg')}>
                        {narrative.pullQuote}
                      </p>
                    </div>
                  </div>

                  <div className={cn(focusMode ? 'space-y-4' : 'xl:col-span-5 space-y-4', figuresFirst && 'xl:order-1')}>
                    <div className="border border-border-subtle rounded-md p-4 bg-[#FAFAF8]">
                      <BlockCanvas blocks={section.blocks} showExport={false} />
                    </div>
                    <p className="px-1 text-xs leading-6 text-muted">
                      {narrative.figureCaption}
                    </p>
                  </div>
                </div>

                {/* Section navigation */}
                <div className="mt-8 flex flex-wrap items-center justify-between gap-3 border-t border-border-subtle pt-5">
                  {previousSection ? (
                    <a
                      href={`#${previousSection.id}`}
                      onClick={() => setActiveSectionId(previousSection.id)}
                      className="text-xs text-muted transition-colors hover:text-text-primary"
                    >
                      {'<-'} {previousSection.number} {previousSection.title}
                    </a>
                  ) : (
                    <span className="text-xs text-text-faint">Beginning of paper</span>
                  )}

                  {nextSection ? (
                    <a
                      href={`#${nextSection.id}`}
                      onClick={() => setActiveSectionId(nextSection.id)}
                      className="text-xs text-accent transition-colors hover:text-text-primary"
                    >
                      Continue to {nextSection.number} {'->'}
                    </a>
                  ) : (
                    <span className="text-xs text-text-faint">End of paper</span>
                  )}
                </div>
              </motion.section>
            )
          })}

          {/* References footer */}
          <section className="border border-border-subtle rounded-lg bg-white p-5 sm:p-6">
            <span className="text-xs text-muted">References and intent</span>
            <div className="mt-4 grid gap-6 lg:grid-cols-[minmax(0,1fr)_280px]">
              <div className="space-y-4 text-[15px] leading-8 text-text-body font-serif">
                <p>
                  This reader view is meant to make the paper easier to absorb, not to replace the canonical study. The underlying findings, blocks, and simulation-backed claims remain the same as the rest of the explorer.
                </p>
                <p>
                  The design goal is editorial clarity: more like a structured research feature than a dashboard. The best first stops are the gamma paradox, the starting-geography section, and the limitations, because those three sections define the paper’s surprise, its realism, and its confidence boundary.
                </p>
              </div>

              <div className="space-y-2">
                {PAPER_METADATA.references.map(reference => (
                  <a
                    key={`footer-${reference.label}`}
                    href={reference.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="flex items-center justify-between rounded-md border border-border-subtle px-3 py-2 text-sm text-text-primary transition-colors hover:border-border-hover"
                  >
                    <span>{reference.label}</span>
                    <ArrowUpRight className="h-3.5 w-3.5 text-muted" />
                  </a>
                ))}
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  )
}
