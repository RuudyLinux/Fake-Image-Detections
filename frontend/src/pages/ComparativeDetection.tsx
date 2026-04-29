import { useState } from 'react';
import { GitCompare, Loader2, AlertCircle } from 'lucide-react';
import UploadZone from '../components/UploadZone';
import ComparativeResultPanel from '../components/ComparativeResultPanel';
import { detectComparative } from '../api/detection';
import type { ComparativeResult, DetectionStatus } from '../types';

export default function ComparativeDetection() {
  const [original, setOriginal] = useState<File | null>(null);
  const [suspected, setSuspected] = useState<File | null>(null);
  const [status, setStatus] = useState<DetectionStatus>('idle');
  const [result, setResult] = useState<ComparativeResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const canAnalyze = !!original && !!suspected && status !== 'analyzing';

  const handleAnalyze = async () => {
    if (!original || !suspected) return;
    setStatus('analyzing');
    setError(null);
    setResult(null);
    try {
      const data = await detectComparative(original, suspected);
      setResult(data);
      setStatus('done');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Comparison failed. Is the backend running?';
      setError(message);
      setStatus('error');
    }
  };

  const handleReset = () => {
    setOriginal(null);
    setSuspected(null);
    setResult(null);
    setError(null);
    setStatus('idle');
  };

  return (
    <div className="min-h-screen pt-24 pb-16 px-6">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-2 text-blue-400 text-sm font-medium mb-2">
            <GitCompare size={14} /> Comparative Detection
          </div>
          <h1 className="text-3xl font-bold text-slate-100">Compare With Original</h1>
          <p className="text-slate-400 mt-1">
            Upload the original and suspected fake — SSIM and pixel analysis pinpoint altered regions.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Upload panel */}
          <div className="lg:col-span-2 space-y-4">
            <div className="card space-y-5">
              <UploadZone label="Original Image" file={original} onFileChange={setOriginal} />
              <div className="border-t border-dark-300" />
              <UploadZone label="Suspected Fake Image" file={suspected} onFileChange={setSuspected} />

              <button
                onClick={handleAnalyze}
                disabled={!canAnalyze}
                className="btn-primary w-full justify-center"
                style={{
                  background: canAnalyze ? undefined : undefined,
                  ...(canAnalyze
                    ? { background: 'linear-gradient(135deg, #2563eb, #1d4ed8)' }
                    : {}),
                }}
              >
                {status === 'analyzing' ? (
                  <>
                    <Loader2 size={16} className="animate-spin" /> Comparing…
                  </>
                ) : (
                  <>
                    <GitCompare size={16} /> Compare Images
                  </>
                )}
              </button>

              {result && (
                <button onClick={handleReset} className="btn-secondary w-full justify-center text-sm">
                  Reset
                </button>
              )}
            </div>

            {/* Info */}
            <div className="card space-y-3">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Techniques</p>
              {[
                { name: 'SSIM Analysis', desc: 'Structural Similarity Index measuring' },
                { name: 'Pixel Difference', desc: 'Absolute per-pixel deviation map' },
                { name: 'Region Detection', desc: 'Contour-based altered area localization' },
              ].map(({ name, desc }) => (
                <div key={name} className="flex gap-3">
                  <div className="w-1.5 h-1.5 rounded-full bg-blue-500 mt-1.5 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-slate-300">{name}</p>
                    <p className="text-xs text-slate-500">{desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Results */}
          <div className="lg:col-span-3">
            {status === 'idle' && (
              <div className="card h-full min-h-64 flex flex-col items-center justify-center text-center gap-4 border-dashed">
                <div className="w-16 h-16 rounded-full bg-dark-400 flex items-center justify-center">
                  <GitCompare size={28} className="text-dark-200" />
                </div>
                <div>
                  <p className="text-slate-400 font-medium">No comparison yet</p>
                  <p className="text-slate-600 text-sm">Upload both images and click Compare</p>
                </div>
              </div>
            )}

            {status === 'analyzing' && (
              <div className="card h-full min-h-64 flex flex-col items-center justify-center gap-4">
                <Loader2 size={40} className="text-blue-500 animate-spin" />
                <div className="text-center">
                  <p className="text-slate-300 font-medium">Comparing images…</p>
                  <p className="text-slate-500 text-sm">SSIM · Pixel diff · Region detection</p>
                </div>
              </div>
            )}

            {status === 'error' && (
              <div className="card border-red-800/40 flex items-start gap-4">
                <AlertCircle className="text-red-400 flex-shrink-0 mt-0.5" size={20} />
                <div>
                  <p className="font-semibold text-red-400">Comparison Failed</p>
                  <p className="text-slate-400 text-sm mt-1">{error}</p>
                </div>
              </div>
            )}

            {status === 'done' && result && original && suspected && (
              <ComparativeResultPanel result={result} original={original} suspected={suspected} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
