/**
 * AutoAlpha localStorage keys. Migrates from legacy quantaalpha_* on first read.
 */
export const LS_CONFIG = 'autoalpha_config';
export const LS_FACTORS = 'autoalpha_factors';
export const LS_ACTIVE_LIBRARY = 'autoalpha_active_library';

const LEGACY_MAP: Record<string, string> = {
  [LS_CONFIG]: 'quantaalpha_config',
  [LS_FACTORS]: 'quantaalpha_factors',
  [LS_ACTIVE_LIBRARY]: 'quantaalpha_active_library',
};

function migrateGet(key: string): string | null {
  const v = localStorage.getItem(key);
  if (v !== null) return v;
  const legacyKey = LEGACY_MAP[key];
  if (!legacyKey) return null;
  const old = localStorage.getItem(legacyKey);
  if (old !== null) {
    localStorage.setItem(key, old);
    return old;
  }
  return null;
}

export function getAutoAlphaConfigRaw(): string | null {
  return migrateGet(LS_CONFIG);
}

export function setAutoAlphaConfigRaw(json: string): void {
  localStorage.setItem(LS_CONFIG, json);
}

export function getAutoAlphaFactorsRaw(): string | null {
  return migrateGet(LS_FACTORS);
}

export function setAutoAlphaFactorsRaw(json: string): void {
  localStorage.setItem(LS_FACTORS, json);
}

export function getActiveLibraryName(): string | null {
  return migrateGet(LS_ACTIVE_LIBRARY);
}

export function setActiveLibraryName(name: string): void {
  localStorage.setItem(LS_ACTIVE_LIBRARY, name);
}
