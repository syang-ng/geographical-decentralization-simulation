export interface GcpRegion {
  readonly id: string
  readonly name: string
  readonly city: string
  readonly lat: number
  readonly lon: number
  readonly macroRegion: MacroRegion
}

export type MacroRegion =
  | 'North America'
  | 'Europe'
  | 'Asia Pacific'
  | 'Middle East'
  | 'South America'
  | 'Africa'
  | 'Oceania'

function macro(id: string): MacroRegion {
  if (id.startsWith('us-') || id.startsWith('northamerica-')) return 'North America'
  if (id.startsWith('europe-')) return 'Europe'
  if (id.startsWith('asia-')) return 'Asia Pacific'
  if (id.startsWith('me-')) return 'Middle East'
  if (id.startsWith('southamerica-')) return 'South America'
  if (id.startsWith('africa-')) return 'Africa'
  if (id.startsWith('australia-')) return 'Oceania'
  return 'Asia Pacific'
}

// Derived from data/gcp_regions.csv in the upstream repo
export const GCP_REGIONS: readonly GcpRegion[] = [
  { id: 'africa-south1', name: 'africa-south1', city: 'Johannesburg, South Africa', lat: -26.2041, lon: 28.0473, macroRegion: macro('africa-south1') },
  { id: 'asia-east1', name: 'asia-east1', city: 'Changhua County, Taiwan', lat: 23.9575, lon: 120.536, macroRegion: macro('asia-east1') },
  { id: 'asia-east2', name: 'asia-east2', city: 'Hong Kong, China', lat: 22.3964, lon: 114.1095, macroRegion: macro('asia-east2') },
  { id: 'asia-northeast1', name: 'asia-northeast1', city: 'Tokyo, Japan', lat: 35.6895, lon: 139.6917, macroRegion: macro('asia-northeast1') },
  { id: 'asia-northeast2', name: 'asia-northeast2', city: 'Osaka, Japan', lat: 34.6937, lon: 135.5023, macroRegion: macro('asia-northeast2') },
  { id: 'asia-northeast3', name: 'asia-northeast3', city: 'Seoul, South Korea', lat: 37.5665, lon: 126.978, macroRegion: macro('asia-northeast3') },
  { id: 'asia-south1', name: 'asia-south1', city: 'Mumbai, India', lat: 19.076, lon: 72.8777, macroRegion: macro('asia-south1') },
  { id: 'asia-south2', name: 'asia-south2', city: 'Delhi, India', lat: 28.7041, lon: 77.1025, macroRegion: macro('asia-south2') },
  { id: 'asia-southeast1', name: 'asia-southeast1', city: 'Singapore', lat: 1.3521, lon: 103.8198, macroRegion: macro('asia-southeast1') },
  { id: 'asia-southeast2', name: 'asia-southeast2', city: 'Jakarta, Indonesia', lat: -6.2088, lon: 106.8456, macroRegion: macro('asia-southeast2') },
  { id: 'australia-southeast1', name: 'australia-southeast1', city: 'Sydney, Australia', lat: -33.8688, lon: 151.2093, macroRegion: macro('australia-southeast1') },
  { id: 'australia-southeast2', name: 'australia-southeast2', city: 'Melbourne, Australia', lat: -37.8136, lon: 144.9631, macroRegion: macro('australia-southeast2') },
  { id: 'europe-central2', name: 'europe-central2', city: 'Warsaw, Poland', lat: 52.2297, lon: 21.0122, macroRegion: macro('europe-central2') },
  { id: 'europe-north1', name: 'europe-north1', city: 'Hamina, Finland', lat: 60.1719, lon: 24.9414, macroRegion: macro('europe-north1') },
  { id: 'europe-north2', name: 'europe-north2', city: 'Stockholm, Sweden', lat: 59.3293, lon: 18.0686, macroRegion: macro('europe-north2') },
  { id: 'europe-southwest1', name: 'europe-southwest1', city: 'Madrid, Spain', lat: 40.4168, lon: -3.7038, macroRegion: macro('europe-southwest1') },
  { id: 'europe-west1', name: 'europe-west1', city: 'St. Ghislain, Belgium', lat: 50.5039, lon: 4.4699, macroRegion: macro('europe-west1') },
  { id: 'europe-west2', name: 'europe-west2', city: 'London, UK', lat: 51.5074, lon: -0.1278, macroRegion: macro('europe-west2') },
  { id: 'europe-west3', name: 'europe-west3', city: 'Frankfurt, Germany', lat: 50.1109, lon: 8.6821, macroRegion: macro('europe-west3') },
  { id: 'europe-west4', name: 'europe-west4', city: 'Eemshaven, Netherlands', lat: 53.3487, lon: 6.2603, macroRegion: macro('europe-west4') },
  { id: 'europe-west6', name: 'europe-west6', city: 'Zurich, Switzerland', lat: 47.3769, lon: 8.5417, macroRegion: macro('europe-west6') },
  { id: 'europe-west8', name: 'europe-west8', city: 'Milan, Italy', lat: 45.4642, lon: 9.19, macroRegion: macro('europe-west8') },
  { id: 'europe-west9', name: 'europe-west9', city: 'Paris, France', lat: 48.8566, lon: 2.3522, macroRegion: macro('europe-west9') },
  { id: 'europe-west10', name: 'europe-west10', city: 'Berlin, Germany', lat: 52.52, lon: 13.405, macroRegion: macro('europe-west10') },
  { id: 'europe-west12', name: 'europe-west12', city: 'Turin, Italy', lat: 45.0703, lon: 7.6869, macroRegion: macro('europe-west12') },
  { id: 'me-central1', name: 'me-central1', city: 'Doha, Qatar', lat: 25.2854, lon: 51.531, macroRegion: macro('me-central1') },
  { id: 'me-west1', name: 'me-west1', city: 'Tel Aviv, Israel', lat: 32.0853, lon: 34.7818, macroRegion: macro('me-west1') },
  { id: 'northamerica-northeast1', name: 'northamerica-northeast1', city: 'Montreal, Canada', lat: 45.5017, lon: -73.5673, macroRegion: macro('northamerica-northeast1') },
  { id: 'northamerica-northeast2', name: 'northamerica-northeast2', city: 'Toronto, Canada', lat: 43.6511, lon: -79.347, macroRegion: macro('northamerica-northeast2') },
  { id: 'southamerica-east1', name: 'southamerica-east1', city: 'Sao Paulo, Brazil', lat: -23.5505, lon: -46.6333, macroRegion: macro('southamerica-east1') },
  { id: 'southamerica-west1', name: 'southamerica-west1', city: 'Santiago, Chile', lat: -33.4489, lon: -70.6693, macroRegion: macro('southamerica-west1') },
  { id: 'us-central1', name: 'us-central1', city: 'Council Bluffs, Iowa', lat: 41.2565, lon: -95.9345, macroRegion: macro('us-central1') },
  { id: 'us-east1', name: 'us-east1', city: 'Moncks Corner, SC', lat: 32.7767, lon: -79.9311, macroRegion: macro('us-east1') },
  { id: 'us-east4', name: 'us-east4', city: 'Ashburn, Virginia', lat: 38.8951, lon: -77.0364, macroRegion: macro('us-east4') },
  { id: 'us-east5', name: 'us-east5', city: 'Columbus, Ohio', lat: 39.9612, lon: -82.9988, macroRegion: macro('us-east5') },
  { id: 'us-south1', name: 'us-south1', city: 'Dallas, Texas', lat: 32.7767, lon: -96.797, macroRegion: macro('us-south1') },
  { id: 'us-west1', name: 'us-west1', city: 'The Dalles, Oregon', lat: 45.6066, lon: -121.1794, macroRegion: macro('us-west1') },
  { id: 'us-west2', name: 'us-west2', city: 'Los Angeles, California', lat: 34.0522, lon: -118.2437, macroRegion: macro('us-west2') },
  { id: 'us-west3', name: 'us-west3', city: 'Salt Lake City, Utah', lat: 40.7608, lon: -111.891, macroRegion: macro('us-west3') },
  { id: 'us-west4', name: 'us-west4', city: 'Las Vegas, Nevada', lat: 36.1699, lon: -115.1398, macroRegion: macro('us-west4') },
] as const

export const MACRO_REGION_COUNTS: Record<MacroRegion, number> = {
  'North America': GCP_REGIONS.filter(r => r.macroRegion === 'North America').length,
  'Europe': GCP_REGIONS.filter(r => r.macroRegion === 'Europe').length,
  'Asia Pacific': GCP_REGIONS.filter(r => r.macroRegion === 'Asia Pacific').length,
  'Middle East': GCP_REGIONS.filter(r => r.macroRegion === 'Middle East').length,
  'South America': GCP_REGIONS.filter(r => r.macroRegion === 'South America').length,
  'Africa': GCP_REGIONS.filter(r => r.macroRegion === 'Africa').length,
  'Oceania': GCP_REGIONS.filter(r => r.macroRegion === 'Oceania').length,
}
