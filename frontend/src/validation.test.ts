import { describe, expect, it } from "vitest";
import { formatBytes, validateFile } from "./validation";

describe("validateFile", () => {
  it("accepts PNG and JPEG within the limit", () => {
    expect(validateFile({ type: "image/png", size: 1000 }, 10)).toBeNull();
    expect(validateFile({ type: "image/jpeg", size: 1000 }, 10)).toBeNull();
  });
  it("rejects other types", () => {
    expect(validateFile({ type: "application/dicom", size: 1000 }, 10)).toMatch(/PNG or JPEG/);
    expect(validateFile({ type: "image/gif", size: 1000 }, 10)).toMatch(/PNG or JPEG/);
  });
  it("rejects empty and oversized files", () => {
    expect(validateFile({ type: "image/png", size: 0 }, 10)).toMatch(/empty/);
    expect(validateFile({ type: "image/png", size: 11 * 1024 * 1024 }, 10)).toMatch(/limit is 10 MB/);
  });
});

describe("formatBytes", () => {
  it("formats sizes", () => {
    expect(formatBytes(500)).toBe("500 B");
    expect(formatBytes(2048)).toBe("2 KB");
    expect(formatBytes(3 * 1024 * 1024)).toBe("3.0 MB");
  });
});
