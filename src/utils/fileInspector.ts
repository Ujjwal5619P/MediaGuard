import type { ClientFileInspection, ExifSummary } from '../types/forensics';

const EMPTY_EXIF: ExifSummary = {
  camera_make: null,
  camera_model: null,
  software: null,
  datetime_original: null,
  has_exif: false,
};

export function detectMagicByteMime(bytes: Uint8Array): string | null {
  if (bytes.length < 12) return null;

  // JPEG: FF D8 FF
  if (bytes[0] === 0xff && bytes[1] === 0xd8 && bytes[2] === 0xff) {
    return 'image/jpeg';
  }

  // PNG: 89 50 4E 47 0D 0A 1A 0A
  if (
    bytes[0] === 0x89 &&
    bytes[1] === 0x50 &&
    bytes[2] === 0x4e &&
    bytes[3] === 0x47 &&
    bytes[4] === 0x0d &&
    bytes[5] === 0x0a &&
    bytes[6] === 0x1a &&
    bytes[7] === 0x0a
  ) {
    return 'image/png';
  }

  // WebP: RIFF....WEBP
  if (
    bytes[0] === 0x52 &&
    bytes[1] === 0x49 &&
    bytes[2] === 0x46 &&
    bytes[3] === 0x46 &&
    bytes[8] === 0x57 &&
    bytes[9] === 0x45 &&
    bytes[10] === 0x42 &&
    bytes[11] === 0x50
  ) {
    return 'image/webp';
  }

  return null;
}

function readAscii(view: DataView, offset: number, length: number): string | null {
  if (offset < 0 || offset + length > view.byteLength || length <= 0) {
    return null;
  }
  const chars: string[] = [];
  for (let i = 0; i < length; i++) {
    const code = view.getUint8(offset + i);
    if (code === 0) break;
    if (code >= 32 && code <= 126) {
      chars.push(String.fromCharCode(code));
    }
  }
  const cleaned = chars.join('').trim();
  return cleaned.length > 0 ? cleaned : null;
}

function parseJpegExif(buffer: ArrayBuffer): ExifSummary {
  try {
    const view = new DataView(buffer);
    if (view.byteLength < 12 || view.getUint16(0, false) !== 0xffd8) {
      return EMPTY_EXIF;
    }

    let offset = 2;
    while (offset + 4 < view.byteLength) {
      if (view.getUint8(offset) !== 0xff) break;
      const marker = view.getUint8(offset + 1);
      if (marker === 0xda || marker === 0xd9) break; // SOS or EOI

      const segmentLength = view.getUint16(offset + 2, false);
      if (segmentLength < 2 || offset + 2 + segmentLength > view.byteLength) {
        break;
      }

      // APP1 Exif\0\0
      if (marker === 0xe1 && segmentLength >= 10) {
        const header = readAscii(view, offset + 4, 4);
        if (header === 'Exif' && view.getUint16(offset + 8, false) === 0x0000) {
          const tiffStart = offset + 10;
          if (tiffStart + 8 > view.byteLength) return EMPTY_EXIF;

          const endianMarker = view.getUint16(tiffStart, false);
          const littleEndian = endianMarker === 0x4949;
          if (!littleEndian && endianMarker !== 0x4d4d) return EMPTY_EXIF;

          if (view.getUint16(tiffStart + 2, littleEndian) !== 0x002a) {
            return EMPTY_EXIF;
          }

          const ifd0Offset = view.getUint32(tiffStart + 4, littleEndian);
          const ifd0Abs = tiffStart + ifd0Offset;
          if (ifd0Abs + 2 > view.byteLength) return EMPTY_EXIF;

          const result: ExifSummary = {
            camera_make: null,
            camera_model: null,
            software: null,
            datetime_original: null,
            has_exif: true,
          };

          const readIfdString = (entryOffset: number, count: number): string | null => {
            if (count <= 0 || count > 256) return null;
            if (count <= 4) {
              return readAscii(view, entryOffset + 8, count);
            }
            const valueRelOffset = view.getUint32(entryOffset + 8, littleEndian);
            return readAscii(view, tiffStart + valueRelOffset, count);
          };

          let exifIfdRelOffset: number | null = null;
          const entries = Math.min(view.getUint16(ifd0Abs, littleEndian), 128);
          for (let i = 0; i < entries; i++) {
            const entryOffset = ifd0Abs + 2 + i * 12;
            if (entryOffset + 12 > view.byteLength) break;
            const tag = view.getUint16(entryOffset, littleEndian);
            const type = view.getUint16(entryOffset + 2, littleEndian);
            const count = view.getUint32(entryOffset + 4, littleEndian);

            if (type === 2) {
              if (tag === 0x010f) result.camera_make = readIfdString(entryOffset, count);
              if (tag === 0x0110) result.camera_model = readIfdString(entryOffset, count);
              if (tag === 0x0131) result.software = readIfdString(entryOffset, count);
              if (tag === 0x0132 && !result.datetime_original) {
                result.datetime_original = readIfdString(entryOffset, count);
              }
            } else if (tag === 0x8769) {
              exifIfdRelOffset = view.getUint32(entryOffset + 8, littleEndian);
            }
          }

          if (exifIfdRelOffset !== null) {
            const exifAbs = tiffStart + exifIfdRelOffset;
            if (exifAbs + 2 <= view.byteLength) {
              const subEntries = Math.min(view.getUint16(exifAbs, littleEndian), 128);
              for (let i = 0; i < subEntries; i++) {
                const entryOffset = exifAbs + 2 + i * 12;
                if (entryOffset + 12 > view.byteLength) break;
                const tag = view.getUint16(entryOffset, littleEndian);
                const type = view.getUint16(entryOffset + 2, littleEndian);
                const count = view.getUint32(entryOffset + 4, littleEndian);
                if (type === 2 && tag === 0x9003) {
                  result.datetime_original = readIfdString(entryOffset, count);
                }
              }
            }
          }

          return result;
        }
      }

      offset += 2 + segmentLength;
    }
    return EMPTY_EXIF;
  } catch {
    return EMPTY_EXIF;
  }
}

async function computeSha256Hex(buffer: ArrayBuffer): Promise<string> {
  const hashBuffer = await crypto.subtle.digest('SHA-256', buffer);
  const hashArray = Array.from(new Uint8Array(hashBuffer));
  return hashArray.map((b) => b.toString(16).padStart(2, '0')).join('');
}

function readImageDimensions(url: string): Promise<{ width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      resolve({ width: img.naturalWidth, height: img.naturalHeight });
    };
    img.onerror = () => {
      reject(new Error('Unable to decode image dimensions. File may be corrupted or unsupported.'));
    };
    img.src = url;
  });
}

export async function inspectFileClientSide(file: File): Promise<ClientFileInspection> {
  const buffer = await file.arrayBuffer();
  const bytes = new Uint8Array(buffer);
  const detectedSignatureMime = detectMagicByteMime(bytes);
  const sha256 = await computeSha256Hex(buffer);
  const previewUrl = URL.createObjectURL(file);

  let width = 0;
  let height = 0;
  if (detectedSignatureMime) {
    try {
      const dims = await readImageDimensions(previewUrl);
      width = dims.width;
      height = dims.height;
    } catch {
      width = 0;
      height = 0;
    }
  }

  const exif =
    detectedSignatureMime === 'image/jpeg' ? parseJpegExif(buffer) : EMPTY_EXIF;

  return {
    filename: file.name,
    declaredMime: file.type || 'unknown',
    detectedSignatureMime,
    signatureValid: detectedSignatureMime !== null && width > 0 && height > 0,
    byteSize: file.size,
    width,
    height,
    sha256,
    exif,
    previewUrl,
  };
}

export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}
