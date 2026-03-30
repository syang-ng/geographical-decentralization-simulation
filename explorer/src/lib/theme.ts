/** Shared animation + color constants. Import these instead of re-declaring per file. */

export const SPRING = { type: 'spring' as const, stiffness: 220, damping: 28, mass: 0.92 }
export const SPRING_SOFT = { type: 'spring' as const, stiffness: 170, damping: 24, mass: 0.96 }
export const SPRING_SNAPPY = { type: 'spring' as const, stiffness: 360, damping: 30, mass: 0.88 }

/** Hover lift preset — subtle card elevation on hover */
export const HOVER_LIFT = {
  whileHover: { y: -1, scale: 1.002, boxShadow: '0 12px 28px rgba(26,26,26,0.08)' },
  whileTap: { scale: 0.985 },
  transition: SPRING_SOFT,
} as const

/** Stagger children preset for scroll-triggered reveals */
export const STAGGER_CONTAINER = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.04, delayChildren: 0.015 } },
} as const

export const STAGGER_ITEM = {
  hidden: { opacity: 0, y: 10, scale: 0.995 },
  visible: { opacity: 1, y: 0, scale: 1, transition: SPRING_SOFT },
} as const

/** Block visualization palette — SSP ocean, MSP earth, plus supporting colors */
export const BLOCK_COLORS = [
  '#2563EB', // ocean blue (SSP)
  '#C2553A', // terracotta (MSP)
  '#16A34A', // meridian green
  '#D97706', // amber
  '#DC2626', // signal red
] as const

/** GCP macro-region coordinate labels — decorative micro-detail */
export const REGION_COORDS = [
  { region: 'North America', coord: '37.4°N 122.1°W', short: 'NA' },
  { region: 'Europe', coord: '50.1°N 8.7°E', short: 'EU' },
  { region: 'Asia Pacific', coord: '35.7°N 139.8°E', short: 'AP' },
  { region: 'South America', coord: '23.5°S 46.6°W', short: 'SA' },
  { region: 'Middle East', coord: '25.3°N 55.3°E', short: 'ME' },
  { region: 'Africa', coord: '33.9°S 18.4°E', short: 'AF' },
  { region: 'Oceania', coord: '33.9°S 151.2°E', short: 'OC' },
] as const
