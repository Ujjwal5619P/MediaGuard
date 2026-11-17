import { useRef, useState, type DragEvent, type MouseEvent } from 'react';
import {
  Eye,
  Hand,
  Layers,
  Maximize2,
  RotateCcw,
  Upload,
  ZoomIn,
  ZoomOut,
} from 'lucide-react';
import type { ExperimentalAttribution, FaceBox } from '../types/forensics';

type OverlayLayer = 'original' | 'face_gradcam' | 'dinov2_attention';

interface ForensicCanvasProps {
  imageUrl: string | null;
  filename: string | null;
  width: number;
  height: number;
  attribution: ExperimentalAttribution | null;
  faceBoxes: FaceBox[] | null;
  apiBaseUrl: string;
  onDropFile?: (file: File) => void;
  onRequestFileSelect?: () => void;
}

export function ForensicCanvas({
  imageUrl,
  filename,
  width,
  height,
  attribution,
  faceBoxes,
  apiBaseUrl,
  onDropFile,
  onRequestFileSelect,
}: ForensicCanvasProps) {
  const [zoom, setZoom] = useState<'fit' | 1 | 1.5 | 2>('fit');
  const [layer, setLayer] = useState<OverlayLayer>('original');
  const [splitPercent, setSplitPercent] = useState<number>(65);
  const [opacity, setOpacity] = useState<number>(85);
  const [coords, setCoords] = useState<{ x: number; y: number } | null>(null);
  const [pan, setPan] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [isPanning, setIsPanning] = useState(false);
  const [isDroppingOnCanvas, setIsDroppingOnCanvas] = useState(false);
  const panOriginRef = useRef<{ startX: number; startY: number; origX: number; origY: number } | null>(
    null
  );

  const handleImageMouseMove = (e: MouseEvent<HTMLImageElement>) => {
    if (!width || !height) return;
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) return;
    const relX = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
    const relY = Math.max(0, Math.min(1, (e.clientY - rect.top) / rect.height));
    setCoords({
      x: Math.round(relX * width),
      y: Math.round(relY * height),
    });
  };

  const handleStageMouseDown = (e: MouseEvent<HTMLDivElement>) => {
    if (zoom === 'fit' || !primaryImageSrc) return;
    if (e.button !== 0) return;
    setIsPanning(true);
    panOriginRef.current = {
      startX: e.clientX,
      startY: e.clientY,
      origX: pan.x,
      origY: pan.y,
    };
  };

  const handleStageMouseMove = (e: MouseEvent<HTMLDivElement>) => {
    if (!isPanning || !panOriginRef.current) return;
    setPan({
      x: panOriginRef.current.origX + (e.clientX - panOriginRef.current.startX),
      y: panOriginRef.current.origY + (e.clientY - panOriginRef.current.startY),
    });
  };

  const handleStageMouseUp = () => {
    setIsPanning(false);
    panOriginRef.current = null;
  };

  const handleZoomChange = (next: 'fit' | 1 | 1.5 | 2) => {
    setZoom(next);
    if (next === 'fit') {
      setPan({ x: 0, y: 0 });
    }
  };

  const resetView = () => {
    setZoom('fit');
    setPan({ x: 0, y: 0 });
    setLayer('original');
    setSplitPercent(65);
    setOpacity(85);
  };

  const handleCanvasDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDroppingOnCanvas(false);
    const file = e.dataTransfer.files?.[0];
    if (file && onDropFile) {
      onDropFile(file);
    }
  };

  const resolveUrl = (raw: string | null | undefined): string | null => {
    if (!raw) return null;
    return raw.startsWith('http') ? raw : `${apiBaseUrl}${raw}`;
  };

  const gradcamUrl = attribution?.gradcam_available
    ? resolveUrl(attribution.gradcam_url)
    : null;
  const patchAttentionUrl = attribution?.patch_attention_available
    ? resolveUrl(attribution.patch_attention_url)
    : null;

  const activeOverlayUrl =
    layer === 'face_gradcam'
      ? gradcamUrl
      : layer === 'dinov2_attention'
      ? patchAttentionUrl
      : null;

  const activeLayerAvailable =
    layer === 'original' ||
    (layer === 'face_gradcam' && Boolean(gradcamUrl)) ||
    (layer === 'dinov2_attention' && Boolean(patchAttentionUrl));

  const primaryImageSrc = imageUrl ?? activeOverlayUrl;

  return (
    <section
      aria-label="Forensic Inspection Viewport"
      className="flex flex-col rounded-xl border border-slate-800 bg-[#0B0E17]"
    >
      {/* Toolbar */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 px-4 py-3">
        <div className="flex flex-wrap items-center gap-1 rounded-lg border border-slate-800 bg-[#07090E] p-1">
          <button
            type="button"
            onClick={() => setLayer('original')}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              layer === 'original'
                ? 'bg-[#111827] text-slate-100'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Eye className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Original</span>
          </button>
          <button
            type="button"
            onClick={() => setLayer('face_gradcam')}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              layer === 'face_gradcam'
                ? 'bg-[#111827] text-cyan-300'
                : gradcamUrl
                ? 'text-cyan-400 hover:text-cyan-300'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Face Grad-CAM</span>
          </button>
          <button
            type="button"
            onClick={() => setLayer('dinov2_attention')}
            className={`flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-medium transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
              layer === 'dinov2_attention'
                ? 'bg-[#111827] text-cyan-300'
                : patchAttentionUrl
                ? 'text-cyan-400 hover:text-cyan-300'
                : 'text-slate-400 hover:text-slate-200'
            }`}
          >
            <Layers className="h-3.5 w-3.5" aria-hidden="true" />
            <span>DINOv2 Attribution</span>
          </button>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1 rounded-lg border border-slate-800 bg-[#07090E] p-1">
            <button
              type="button"
              onClick={() => handleZoomChange('fit')}
              className={`rounded px-2.5 py-1 text-xs font-mono tabular-nums transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                zoom === 'fit'
                  ? 'bg-[#111827] text-cyan-400'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              Fit
            </button>
            <button
              type="button"
              onClick={() => handleZoomChange(1)}
              className={`flex items-center gap-1 rounded px-2.5 py-1 text-xs font-mono tabular-nums transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                zoom === 1
                  ? 'bg-[#111827] text-cyan-400'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <ZoomOut className="h-3 w-3" aria-hidden="true" />
              <span>100%</span>
            </button>
            <button
              type="button"
              onClick={() => handleZoomChange(1.5)}
              className={`rounded px-2.5 py-1 text-xs font-mono tabular-nums transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                zoom === 1.5
                  ? 'bg-[#111827] text-cyan-400'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              150%
            </button>
            <button
              type="button"
              onClick={() => handleZoomChange(2)}
              className={`flex items-center gap-1 rounded px-2.5 py-1 text-xs font-mono tabular-nums transition-colors whitespace-nowrap shrink-0 cursor-pointer ${
                zoom === 2
                  ? 'bg-[#111827] text-cyan-400'
                  : 'text-slate-400 hover:text-slate-200'
              }`}
            >
              <ZoomIn className="h-3 w-3" aria-hidden="true" />
              <span>200%</span>
            </button>
          </div>

          <button
            type="button"
            onClick={resetView}
            title="Reset zoom, pan, and overlay view"
            className="flex items-center gap-1.5 rounded-lg border border-slate-800 bg-[#07090E] px-3 py-1.5 text-xs text-slate-300 hover:border-slate-700 hover:text-white transition-colors whitespace-nowrap shrink-0 cursor-pointer"
          >
            <RotateCcw className="h-3.5 w-3.5" aria-hidden="true" />
            <span>Reset</span>
          </button>
        </div>
      </div>

      {/* Overlay Banner */}
      {layer !== 'original' && !activeLayerAvailable && (
        <div className="border-b border-slate-800 bg-amber-950/20 px-4 py-2.5 text-xs text-amber-200">
          <span className="font-semibold">Overlay Unavailable:</span>{' '}
          {layer === 'face_gradcam'
            ? 'Grad-CAM requires at least one detected face crop and EfficientNet-B0 attribution enabled.'
            : attribution?.reason ||
              'DINOv2 attribution overlay was not generated for this investigation.'}
        </div>
      )}

      {layer !== 'original' && activeLayerAvailable && (
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-800 bg-[#07090E] px-4 py-2.5 text-xs text-slate-300">
          <span>
            {layer === 'face_gradcam'
              ? 'Experimental EfficientNet-B0 Grad-CAM saliency across detected face regions (diagnostic aid, not definitive proof).'
              : 'Experimental DINOv2 ViT-S/14 CLS-to-patch self-attention overlay (semantic salience aid, not manipulation proof).'}
          </span>
          {imageUrl && (
            <div className="flex flex-wrap items-center gap-4">
              <label className="flex items-center gap-2 font-mono tabular-nums">
                <span className="text-slate-400">Split</span>
                <input
                  type="range"
                  min={0}
                  max={100}
                  value={splitPercent}
                  onChange={(e) => setSplitPercent(Number(e.target.value))}
                  className="accent-cyan-400"
                />
                <span>{splitPercent}%</span>
              </label>
              <label className="flex items-center gap-2 font-mono tabular-nums">
                <span className="text-slate-400">Opacity</span>
                <input
                  type="range"
                  min={10}
                  max={100}
                  value={opacity}
                  onChange={(e) => setOpacity(Number(e.target.value))}
                  className="accent-cyan-400"
                />
                <span>{opacity}%</span>
              </label>
            </div>
          )}
        </div>
      )}

      {/* Large Central Viewport */}
      <div
        onMouseDown={handleStageMouseDown}
        onMouseMove={handleStageMouseMove}
        onMouseUp={handleStageMouseUp}
        onMouseLeave={handleStageMouseUp}
        onDragOver={(e) => {
          if (!onDropFile) return;
          e.preventDefault();
          setIsDroppingOnCanvas(true);
        }}
        onDragLeave={() => setIsDroppingOnCanvas(false)}
        onDrop={handleCanvasDrop}
        className={`forensic-grid relative flex min-h-[460px] xl:min-h-[560px] max-h-[680px] items-center justify-center overflow-hidden p-6 transition-colors ${
          zoom !== 'fit' && primaryImageSrc ? (isPanning ? 'cursor-grabbing' : 'cursor-grab') : ''
        } ${isDroppingOnCanvas ? 'ring-2 ring-inset ring-cyan-400 bg-cyan-950/15' : ''}`}
      >
        {primaryImageSrc ? (
          <div
            style={{
              transform: `translate(${pan.x}px, ${pan.y}px)`,
            }}
            className="relative inline-flex items-center justify-center transition-transform duration-75"
          >
            <img
              src={primaryImageSrc}
              alt={filename ? `Inspection media: ${filename}` : 'Inspection media preview'}
              referrerPolicy="no-referrer"
              draggable={false}
              onMouseMove={handleImageMouseMove}
              onMouseLeave={() => setCoords(null)}
              style={
                zoom === 'fit'
                  ? { maxHeight: '520px', width: 'auto', maxWidth: '100%' }
                  : {
                      width: width ? `${width * zoom}px` : 'auto',
                      height: height ? `${height * zoom}px` : 'auto',
                      maxWidth: 'none',
                    }
              }
              className="rounded border border-slate-800 object-contain select-none"
            />
            {imageUrl && layer !== 'original' && activeOverlayUrl && (
              <img
                src={activeOverlayUrl}
                alt="Experimental attribution overlay"
                referrerPolicy="no-referrer"
                draggable={false}
                style={{
                  opacity: opacity / 100,
                  clipPath: `inset(0 ${100 - splitPercent}% 0 0)`,
                }}
                className="pointer-events-none absolute inset-0 h-full w-full rounded object-contain"
              />
            )}
            {imageUrl &&
              width > 0 &&
              height > 0 &&
              faceBoxes &&
              faceBoxes.map((box, idx) => (
                <div
                  key={idx}
                  aria-label={`Detected face region ${idx + 1}`}
                  style={{
                    left: `${(box.x / width) * 100}%`,
                    top: `${(box.y / height) * 100}%`,
                    width: `${(box.width / width) * 100}%`,
                    height: `${(box.height / height) * 100}%`,
                  }}
                  className="pointer-events-none absolute border border-cyan-400/90"
                />
              ))}
          </div>
        ) : (
          <div className="flex max-w-md flex-col items-center text-center">
            <div className="mb-4 flex h-12 w-12 items-center justify-center rounded-xl border border-slate-800 bg-[#111827] text-cyan-400">
              {onDropFile ? (
                <Upload className="h-5 w-5" aria-hidden="true" />
              ) : (
                <Maximize2 className="h-5 w-5" aria-hidden="true" />
              )}
            </div>
            <h2 className="text-base font-semibold text-slate-100">
              {filename
                ? 'Original Upload Deleted After Analysis'
                : 'Drop an Image Here to Begin Forensic Inspection'}
            </h2>
            <p className="mt-2 text-xs leading-relaxed text-slate-400">
              {filename
                ? 'Original uploaded images are permanently deleted immediately after inference. Select Face Grad-CAM or DINOv2 Attribution above if an artifact was preserved.'
                : 'Supports PNG, JPEG, and WebP up to 15 MB. Magic bytes, dimensions, and SHA-256 checksums are inspected before real model inference.'}
            </p>
            {!filename && onRequestFileSelect && (
              <button
                type="button"
                onClick={onRequestFileSelect}
                className="mt-5 rounded-lg bg-cyan-500 px-4 py-2 text-xs font-semibold text-slate-950 hover:bg-cyan-400 transition-colors cursor-pointer"
              >
                Choose Image File
              </button>
            )}
          </div>
        )}
      </div>

      {/* Bottom Telemetry Strip */}
      <div className="flex flex-wrap items-center justify-between gap-4 border-t border-slate-800 px-4 py-2.5 text-xs font-mono tabular-nums text-slate-400">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-slate-200">{filename || 'No active file'}</span>
          {width > 0 && height > 0 && (
            <>
              <span aria-hidden="true" className="text-slate-700">·</span>
              <span>{width} × {height} px</span>
            </>
          )}
          {faceBoxes && faceBoxes.length > 0 && (
            <>
              <span aria-hidden="true" className="text-slate-700">·</span>
              <span className="text-cyan-400">
                {faceBoxes.length} face{faceBoxes.length > 1 ? 's' : ''} detected
              </span>
            </>
          )}
        </div>
        <div className="flex items-center gap-3">
          {zoom !== 'fit' && primaryImageSrc && (
            <span className="inline-flex items-center gap-1 text-slate-400">
              <Hand className="h-3 w-3" aria-hidden="true" />
              <span>Drag image to pan</span>
            </span>
          )}
          <span>
            {coords
              ? `Pixel (${coords.x}, ${coords.y})`
              : 'Hover image for pixel coordinates'}
          </span>
        </div>
      </div>
    </section>
  );
}
