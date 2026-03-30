/** Shared animation + color constants. Import these instead of re-declaring per file. */

export const SPRING = { type: 'spring' as const, stiffness: 240, damping: 30 }
export const SPRING_SOFT = { type: 'spring' as const, stiffness: 180, damping: 24 }
export const SPRING_SNAPPY = { type: 'spring' as const, stiffness: 400, damping: 35 }

/** Hover lift preset — subtle card elevation on hover */
export const HOVER_LIFT = {
  whileHover: { y: -2, boxShadow: '0 4px 12px rgba(0,0,0,0.06)' },
  whileTap: { scale: 0.985 },
  transition: SPRING,
} as const

/** Stagger children preset for scroll-triggered reveals */
export const STAGGER_CONTAINER = {
  hidden: {},
  visible: { transition: { staggerChildren: 0.06, delayChildren: 0.02 } },
} as const

export const STAGGER_ITEM = {
  hidden: { opacity: 0, y: 16, filter: 'blur(4px)' },
  visible: { opacity: 1, y: 0, filter: 'blur(0px)', transition: SPRING },
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
