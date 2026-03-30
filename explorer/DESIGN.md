# Design System: "Printed Paper, Digital Margins"

A well-typeset research paper that happens to be interactive. Not a dashboard. Not an AI product. A reading surface with life in its seams.

## Philosophy

This explorer is a companion to Yang et al. (2025). Its job is to make the paper's findings legible, trustworthy, and explorable. Every design choice serves that job. If a visual element wouldn't survive in a printed paper or on an e-ink screen, it needs to justify its existence.

### Three reference poles

| Reference | What we take | What we leave |
|-----------|-------------|---------------|
| **E-ink / academic paper** | High contrast, generous whitespace, serif body text, section numbering, no decoration that wouldn't survive print | The gray, the slowness |
| **Stripe docs** | Perfect type scale, color dots as status indicators, one accent used surgically, 4px rhythm, components as furniture not decoration | The marketing energy |
| **Anthropic brand** | Warmth through restraint, off-white canvas, one typeface family, no gradients, borders so subtle they almost disappear | The beige |

## Core Principles

### 1. Light mode

Research papers are light. E-ink is light. Dark mode says "hacker tool." Light mode says "I trust this enough to print it." The canvas is warm off-white (`#FAFAF8`), not clinical pure white.

### 2. Color through absence

The surface is near-monochrome. Color appears in exactly two places:

- **Status dots**: Small saturated circles (8-10px) that carry semantic meaning. These are the Stripe device — a tiny punch of color on a quiet surface makes them powerful.
- **Data visualization**: Charts, maps, and comparison highlights use the palette. Color earns its place by encoding information.

Everything else is black, near-black, or gray. No `bg-accent/10` tinted backgrounds. No `radial-gradient` decorative washes. No glass morphism. No glows.

### 3. Status dot vocabulary

| Dot | Hex | Meaning |
|-----|-----|---------|
| Blue | `#3B82F6` | Interactive / active / SSP paradigm |
| Warm | `#D97757` | Accent data / MSP paradigm |
| Green | `#22C55E` | Verified / curated / positive |
| Amber | `#F59E0B` | Caution / caveat / history-reused |
| Red | `#EF4444` | Surprising result / danger / negative |
| Gray | `#9CA3AF` | Inactive / muted / neutral |

Dots appear: next to tab labels (active tab), next to provenance labels, next to section headers in Deep Dive, next to experiment tags. They replace colored badge backgrounds.

### 4. Typography is the hierarchy

One sans-serif family (Inter) for UI. Serif (`Georgia`, `Charter`, system serif) for prose — insight blocks, paper reader narrative, section descriptions.

**No uppercase tracking-wider micro-labels.** This is the single strongest marker of "AI dashboard" aesthetic. Replace every instance with normal-case text at a slightly smaller size and muted color.

Type scale (based on 4px grid):
- `xs`: 12px / 16px — metadata, sublabels
- `sm`: 14px / 20px — body text, table cells
- `base`: 16px / 24px — section descriptions, insight prose
- `lg`: 20px / 28px — section titles
- `xl`: 24px / 32px — page headings
- `2xl`: 32px / 40px — paper title (serif)

Weight: Regular (400) for body, Medium (500) for labels, Semibold (600) for headings. No bold (700) except stat values.

### 5. Space is the structure

Whitespace replaces decoration as the primary grouping mechanism.

- Between major sections: `48px` (`py-12`)
- Between cards/blocks: `16px` (`gap-4`)
- Inside cards: `20-24px` padding
- Page margins: `max-w-3xl` for reading content, `max-w-5xl` for data-heavy pages

Horizontal rules (`<hr>`) are `1px solid #E5E5E5` — used sparingly between major content shifts, never as decoration.

### 6. Borders, not shadows

One border weight: `1px solid #E8E8E6`. Cards are flat rectangles with this border and no shadow. On hover, border transitions to `#D4D4D2`. No box-shadow except:
- Focus rings on interactive elements (`ring-2 ring-blue-500/20`)
- Elevated overlays (dropdowns, tooltips): `shadow-sm` only

### 7. Animation is life, not decoration

The surface is quiet but alive. Every interaction has a micro-response. Motion principles:

**Physics-based**: All transitions use spring physics (`stiffness: 300, damping: 30`). No ease, no linear, no cubic-bezier.

**Purposeful**: Animation communicates state change, not personality. Rules:
- **Enter**: Fade in + slight upward shift (8px). Staggered for lists.
- **Exit**: Fade out. No downward shift (exits should feel instantaneous).
- **Hover**: Subtle lift (`y: -2px`) on interactive cards. Border color transition on all bordered elements.
- **Click**: Scale down (`0.98`) on press, spring back on release.
- **Expand/collapse**: Height animation with spring. Content fades in slightly delayed.
- **Tab switch**: Cross-fade with horizontal offset (entering from direction of tab).
- **Loading**: Gentle pulse on skeleton shapes. No spinners except for the query bar.

**Restraint**: Never animate color. Never animate border-radius. Never use bounce. Never delay more than 100ms.

### 8. Interactive feedback hierarchy

| Interaction | Response | Timing |
|-------------|----------|--------|
| Hover on card | Border lightens, `y: -2px` lift | Spring, immediate |
| Hover on button | Background fill appears (gray-100) | 150ms transition |
| Click button | Scale 0.98 → spring back | Spring |
| Expand accordion | Height animates, content fades in at 50ms delay | Spring |
| Tab change | Underline slides (layoutId), content cross-fades | Spring |
| Submit query | Input border pulses once, loading state | Immediate |
| Block canvas enter | Staggered fade-in, 60ms between blocks | Spring |
| Stat counter | Number counts up from 0 (if first render) | 400ms, spring |

## Component Patterns

### Header
```
[● dot] Paper title in serif (2xl)
        Authors · Interactive Research Explorer (xs, muted)
                                                    [arXiv] [GitHub]
```
No background. No border-bottom gradient. Just text and space. The dot is blue (active/live indicator).

### Tab nav
```
[● Findings]  [History]  [Paper]  [Deep Dive]  [Simulation]
─────────────────────────────────────────────────────────────
```
Plain text. Active tab has a blue dot before its label and an underline that slides between tabs (layoutId animation). Inactive tabs are muted gray. No icons — the words are clear enough. Sticky with white background + subtle bottom border.

### Stat block
```
┌─────────────────────────┐
│ 40                      │
│ GCP Regions             │
│ Real inter-region       │
│ latency measurements    │
└─────────────────────────┘
```
Large number in semibold (3xl). Label in medium (sm). Sublabel in muted (xs). White background, single border. No colored accent, no tinted background.

### Insight block
```
│ Two-layer geographic game
│
│ Validators choose GCP regions to minimize
│ latency on two critical paths...
```
Left border in blue (3px). No background tint. Title in semibold sans. Body in serif, relaxed line-height. Key-finding emphasis: warm dot before title. Surprising emphasis: red dot before title.

### Comparison block
```
SSP (External)              │  MSP (Local)
● Blue dot                  │  ● Warm dot
                            │
Gini_g        ~0.40         │  Gini_g        ~0.55
HHI_g         ~0.06         │  HHI_g         ~0.10
LC_g          ~8 regions    │  LC_g          ~4 regions
                            │
─────────────────────────────────────────────
Verdict: MSP centralizes 37-80% more...
```
Two columns separated by a thin vertical line. Paradigm label with its color dot. Clean key-value pairs. Verdict below a horizontal rule in italic.

### Table block
Clean table with subtle header row (gray-50 background). Row hover highlights. Highlighted rows have a warm left-border accent, not a background tint.

### Caveat block
```
⚠ This transient decentralization effect is
  fragile and parameter-dependent...
```
Amber dot (not icon) before text. Muted text color. Indented. No border, no background — the dot and indentation are enough.

### Chart block
White background. Thin gray gridlines. Data colors from the palette. No decorative gradients on bars. Clean axis labels. Legend uses color dots.

### Map block
Light gray land masses (or just the dot plot on white). No dark background. Dots use the density color scale. Clean legend with dot + label pairs.

### Query bar
```
┌─────────────────────────────────────────────┐
│ ✦  Ask anything about the paper...      [→] │
└─────────────────────────────────────────────┘
  How does SSP compare?  ·  Attestation threshold  ·  ...
```
Single border input. Sparkle icon in muted gray. On focus: blue ring. Example chips below as plain text with dot separators, not pill buttons.

### Provenance indicator
```
● Curated overview
  Editorial overview assembled from the paper's main findings.
```
A colored dot + plain text label. No pill, no badge, no colored background. The dot does all the work.

### Empty state
```
        No explorations yet.

   Ask a question on the Findings tab.
   Every response is saved here.

        [Start exploring →]
```
Centered. Generous vertical space. One text button. No icon above the heading.

## Color Palette

### Surface
| Token | Value | Use |
|-------|-------|-----|
| `canvas` | `#FAFAF8` | Page background |
| `surface` | `#FFFFFF` | Card/block background |
| `surface-hover` | `#F5F5F3` | Hovered card background |
| `border` | `#E8E8E6` | Card borders, dividers |
| `border-hover` | `#D4D4D2` | Hovered borders |
| `rule` | `#E5E5E5` | Horizontal rules |

### Text
| Token | Value | Use |
|-------|-------|-----|
| `text-primary` | `#1A1A1A` | Headings, values |
| `text-body` | `#374151` | Body text, table cells |
| `text-muted` | `#6B7280` | Sublabels, metadata |
| `text-faint` | `#9CA3AF` | Placeholder, inactive |

### Accent (used sparingly)
| Token | Value | Use |
|-------|-------|-----|
| `blue` | `#3B82F6` | Active states, SSP, links, focus rings |
| `warm` | `#D97757` | MSP paradigm, secondary data |
| `green` | `#22C55E` | Verified, curated, positive |
| `amber` | `#F59E0B` | Caveat, history, caution |
| `red` | `#EF4444` | Surprising, danger |

### Data (charts only)
```
#3B82F6, #D97757, #2DD4BF, #FBBF24, #F43F5E
```
Same five colors as current `BLOCK_COLORS`, used only inside chart/map visualizations.

## Spacing Scale

Based on 4px grid. Use Tailwind values:
```
4px   = 1    (gap-1, p-1)
8px   = 2    (gap-2, p-2)
12px  = 3    (gap-3, p-3)
16px  = 4    (gap-4, p-4, standard card gap)
20px  = 5    (p-5, card padding)
24px  = 6    (p-6, generous card padding)
32px  = 8    (py-8, section spacing)
48px  = 12   (py-12, major section breaks)
```

## What This Replaces

| Current pattern | New pattern |
|----------------|-------------|
| `text-[10px] uppercase tracking-[0.18em]` micro-labels | `text-xs text-muted` normal-case |
| `bg-accent/10 text-accent` tinted badges | `● dot` + plain text |
| `bg-[radial-gradient(...)]` decorative headers | White/canvas background + space |
| `shadow-[0_24px_80px_rgba(0,0,0,0.18)]` | `border border-[#E8E8E6]` |
| `glass-1`, `glass-2` morphism | `bg-white border` |
| Dark canvas `#050505` | Light canvas `#FAFAF8` |
| Colored provenance pills | Dot + label text |
| Icons in tab nav | Text-only with dot indicator |
| `UPPERCASE TRACKING` section headers | Normal-case section headers |
| Decorative shimmer/glow animations | Physics-based interaction responses |

## Implementation Order

1. **CSS variables + Tailwind config**: Swap dark palette → light palette
2. **index.css**: Remove glass, shimmer, dark-specific utilities
3. **Header + TabNav + Footer**: Simplify to text + dots
4. **Block components** (stat, insight, comparison, table, caveat): Apply flat white cards
5. **Chart + Map + TimeSeries**: Light backgrounds, thin gridlines
6. **Page headers** (Findings, Deep Dive, Paper, Sim Lab): Remove gradients, apply space
7. **QueryBar + provenance**: Simplify to border input + dot labels
8. **Animation pass**: Ensure all interactions have spring responses
9. **Verify**: Every tab, every block type, mobile + desktop
