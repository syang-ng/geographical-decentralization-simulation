import { motion } from 'framer-motion'
import { SPRING_SOFT } from '../../lib/theme'

/** Faint meridian wireframe globe — decorative header background element */
export function GlobeWireframe({ className = '' }: { readonly className?: string }) {
  return (
    <motion.svg
      initial={{ opacity: 0, rotate: -8 }}
      animate={{ opacity: 1, rotate: 0 }}
      transition={{ ...SPRING_SOFT, delay: 0.3 }}
      className={className}
      width="180"
      height="180"
      viewBox="0 0 180 180"
      fill="none"
      aria-hidden="true"
    >
      {/* Outer sphere */}
      <circle cx="90" cy="90" r="80" stroke="var(--color-meridian)" strokeWidth="0.75" opacity="0.4" />

      {/* Longitude meridians */}
      <ellipse cx="90" cy="90" rx="55" ry="80" stroke="var(--color-meridian)" strokeWidth="0.5" opacity="0.3" />
      <ellipse cx="90" cy="90" rx="28" ry="80" stroke="var(--color-meridian)" strokeWidth="0.5" opacity="0.25" />

      {/* Latitude parallels */}
      <ellipse cx="90" cy="90" rx="80" ry="55" stroke="var(--color-meridian)" strokeWidth="0.5" opacity="0.2" />
      <line x1="10" y1="90" x2="170" y2="90" stroke="var(--color-meridian)" strokeWidth="0.5" opacity="0.25" />
      <ellipse cx="90" cy="55" rx="70" ry="8" stroke="var(--color-meridian)" strokeWidth="0.4" opacity="0.15" />
      <ellipse cx="90" cy="125" rx="70" ry="8" stroke="var(--color-meridian)" strokeWidth="0.4" opacity="0.15" />

      {/* Validator node dots — spread across the globe */}
      <circle cx="55" cy="65" r="2" fill="var(--color-accent)" opacity="0.5" />
      <circle cx="110" cy="58" r="2.5" fill="var(--color-accent)" opacity="0.6" />
      <circle cx="130" cy="85" r="1.5" fill="var(--color-accent-warm)" opacity="0.4" />
      <circle cx="70" cy="100" r="2" fill="var(--color-accent)" opacity="0.35" />
      <circle cx="95" cy="75" r="1.5" fill="var(--color-success)" opacity="0.4" />
      <circle cx="45" cy="110" r="1.5" fill="var(--color-accent-warm)" opacity="0.3" />
      <circle cx="120" cy="115" r="2" fill="var(--color-accent)" opacity="0.3" />

      {/* Faint network lines between nodes */}
      <line x1="55" y1="65" x2="110" y2="58" stroke="var(--color-accent)" strokeWidth="0.3" opacity="0.2" />
      <line x1="110" y1="58" x2="130" y2="85" stroke="var(--color-meridian)" strokeWidth="0.3" opacity="0.15" />
      <line x1="70" y1="100" x2="95" y2="75" stroke="var(--color-meridian)" strokeWidth="0.3" opacity="0.15" />
      <line x1="55" y1="65" x2="70" y2="100" stroke="var(--color-meridian)" strokeWidth="0.3" opacity="0.12" />
    </motion.svg>
  )
}
