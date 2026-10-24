export const ACCEPTED_TYPES = ["image/png", "image/jpeg"] as const;

/** Client-side pre-check. The backend re-validates everything. */
export function validateFile(file: { type: string; size: number }, maxMb: number): string | null {
  if (!(ACCEPTED_TYPES as readonly string[]).includes(file.type)) {
    return "Please choose a PNG or JPEG image. DICOM files are not supported yet.";
  }
  if (file.size === 0) return "This file is empty.";
  if (file.size > maxMb * 1024 * 1024) {
    return `This file is ${(file.size / 1024 / 1024).toFixed(1)} MB. The limit is ${maxMb} MB.`;
  }
  return null;
}

export function formatBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(0)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}
