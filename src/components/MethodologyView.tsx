import { ExternalLink } from 'lucide-react';

export function MethodologyView() {
  return (
    <div className="mx-auto max-w-[1160px] px-6 py-8 space-y-10">
      <section className="space-y-3 border-b border-slate-800 pb-8">
        <h2 className="text-2xl font-semibold text-slate-100">
          Detection Methodology, Pretrained Models &amp; Scientific Limitations
        </h2>
        <p className="max-w-3xl text-sm leading-relaxed text-slate-300">
          MediaGuard — Verify Before You Trust — executes independent pretrained vision pipelines for AI-generated image detection, face manipulation analysis, and self-supervised visual feature attribution. Primary synthetic origin scoring uses <span className="font-mono text-slate-100">Imagers/Imager-1-tiny</span>, paired with a separate UnivFD comparison detector and task-isolated face forensics.
        </p>
      </section>

      <section className="grid grid-cols-1 gap-6 md:grid-cols-3 border-b border-slate-800 pb-8">
        <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-6 space-y-3">
          <div className="flex items-start justify-between gap-2">
            <h3 className="text-base font-semibold text-slate-100">
              01. Primary AI Detector — Imager-1-Tiny
            </h3>
            <span className="font-mono text-[11px] text-cyan-400 shrink-0">Apache-2.0</span>
          </div>
          <p className="text-xs leading-relaxed text-slate-300">
            Official{' '}
            <a
              href="https://huggingface.co/Imagers/Imager-1-tiny"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 font-mono text-cyan-400 hover:text-cyan-300"
            >
              <span>Imagers/Imager-1-tiny</span>
              <ExternalLink className="h-3 w-3" aria-hidden="true" />
            </a>{' '}
            (<span className="font-mono text-slate-100">ResNet-50</span>, Apache-2.0 weights).
          </p>
          <p className="text-xs leading-relaxed text-slate-400">
            Uses a single <span className="font-mono text-slate-200">224×224</span> center crop, ImageNet normalization, and a two-class head. The verified checkpoint metadata maps class 0 to real and class 1 to AI; MediaGuard reports the class-1 softmax score and applies the checkpoint's <span className="font-mono">0.385</span> threshold on CPU.
          </p>
        </div>

        <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-6 space-y-3">
          <div className="flex items-start justify-between gap-2">
            <h3 className="text-base font-semibold text-slate-100">
              02. Secondary Comparison — UnivFD
            </h3>
            <span className="font-mono text-[11px] text-cyan-400 shrink-0">MIT License</span>
          </div>
          <p className="text-xs leading-relaxed text-slate-300">
            Retains <span className="font-mono text-slate-100">UniversalFakeDetect (UnivFD)</span> from{' '}
            <a
              href="https://github.com/Arjunkalliyadath/ai-image-detector"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-1 font-mono text-cyan-400 hover:text-cyan-300"
            >
              <span>Arjunkalliyadath/ai-image-detector</span>
              <ExternalLink className="h-3 w-3" aria-hidden="true" />
            </a>{' '}
            strictly as a non-primary comparison reference.
          </p>
          <p className="text-xs leading-relaxed text-slate-400">
            Bicubic resize + center-crop to <span className="font-mono text-slate-200">224×224</span>, OpenAI CLIP <span className="font-mono text-slate-200">ViT-L/14</span> (304M vision params) L2-normalized 768-d embeddings, scored via <span className="font-mono text-slate-200">fc_weights.pth</span>. Reported independently alongside the primary detector without ensemble blending.
          </p>
        </div>

        <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-6 space-y-3">
          <div className="flex items-start justify-between gap-2">
            <h3 className="text-base font-semibold text-slate-100">
              03. Face Manipulation — EffNet-B0
            </h3>
            <span className="font-mono text-[11px] text-amber-400 shrink-0">MIT / FF++</span>
          </div>
          <p className="text-xs leading-relaxed text-slate-300">
            Detects frontal faces first via OpenCV Haar cascades. Images without detectable faces report <span className="font-mono text-slate-100">no_face_detected</span> explicitly rather than guessing on background pixels.
          </p>
          <p className="text-xs leading-relaxed text-slate-400">
            Detected face crops (+25% margin, resized to <span className="font-mono text-slate-200">224×224</span>) are scored by <span className="font-mono text-slate-200">Xicor9/efficientnet-b0-ffpp-c23</span> (FaceForensics++ C23: DeepFake, FaceSwap, Face2Face, NeuralTextures) with real CNN <span className="font-mono">Grad-CAM</span> saliency on <span className="font-mono">features[-1]</span>.
          </p>
        </div>
      </section>

      <section className="space-y-4 border-b border-slate-800 pb-8">
        <h3 className="text-lg font-semibold text-slate-100">
          Self-Supervised Features &amp; Experimental Explainability vs. Proof
        </h3>
        <div className="grid grid-cols-1 gap-6 md:grid-cols-2 text-xs leading-relaxed text-slate-300">
          <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5">
            <h4 className="font-semibold text-slate-100">DINOv2 ViT-S/14 Feature Extractor</h4>
            <p className="mt-2 text-slate-400">
              <span className="font-mono text-slate-200">facebook/dinov2-small</span> (Apache-2.0) extracts real 384-dimensional CLS feature vectors and averages last-block CLS-to-patch self-attention heads. Used strictly for visual representation and experimental salience—never presented as a standalone deepfake classifier.
            </p>
          </div>
          <div className="rounded-xl border border-slate-800 bg-[#0B0E17] p-5">
            <h4 className="font-semibold text-slate-100">Saliency Overlays Are Not Proof</h4>
            <p className="mt-2 text-slate-400">
              Both EfficientNet-B0 Grad-CAM (on detected face crops) and DINOv2 patch attention highlight regions influencing neural representations. Neither proves pixel-level tampering nor establishes chain-of-custody provenance on its own.
            </p>
          </div>
        </div>
      </section>

      <section className="space-y-4">
        <h3 className="text-lg font-semibold text-slate-100">
          Dataset Evaluation Governance, Hardware Constraints &amp; Security
        </h3>
        <ul className="space-y-2.5 text-xs leading-relaxed text-slate-300 list-disc pl-5">
          <li>
            <span className="font-medium text-slate-100">Local Dataset Evaluation &amp; Split Separation:</span> Evaluates local folders (<span className="font-mono">dataset/ai_generated/</span> and <span className="font-mono">dataset/real/</span>) with full confusion matrix (TP, FP, TN, FN), Accuracy, Precision, Recall, F1, FPR, and FNR. Never trains, fine-tunes, or auto-tunes thresholds on evaluation images; threshold selection belongs strictly on <span className="font-mono">validation</span> splits while keeping <span className="font-mono">final_test</span> untouched.
          </li>
          <li>
            <span className="font-medium text-slate-100">Uncalibrated Outputs:</span> Imager-1-Tiny's class-1 softmax score, UnivFD's sigmoid score, and EfficientNet softmax probabilities are statistical signals, not certainty. Imager-1-Tiny uses the author-reported threshold <span className="font-mono">0.385</span>; other tasks use <span className="font-mono">0.50</span> by default.
          </li>
          <li>
            <span className="font-medium text-slate-100">CPU Detector:</span> The primary detector always runs FP32 on CPU, including on machines that also have CUDA. Optional local comparison, face, and feature models keep their existing GPU-aware execution when lightweight deployment mode is off.
          </li>
          <li>
            <span className="font-medium text-slate-100">Privacy &amp; Bounded Validation:</span> Individual scan uploads are bounded to 15 MB, 8192 px max dimension, and 36 MP, validated via magic-byte headers, and originals are deleted immediately after inference.
          </li>
        </ul>
      </section>
    </div>
  );
}
