/** Shared animation + color constants. Import these instead of re-declaring per file. */

export const SPRING = { type: 'spring' as const, stiffness: 240, damping: 30 }
export const SPRING_SOFT = { type: 'spring' as const, stiffness: 180, damping: 24 }

/** Block visualization palette — SSP blue, MSP warm, plus supporting colors */
export const BLOCK_COLORS = [
  '#3B82F6', // accent (SSP blue)
  '#d97757', // warm (MSP terracotta)
  '#2dd4bf', // success / teal
  '#fbbf24', // warning / amber
  '#f43f5e', // danger / rose
] as const
