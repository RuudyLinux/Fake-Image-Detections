import { useState } from 'react';
import { Search, Loader2, AlertCircle } from 'lucide-react';
import UploadZone from '../components/UploadZone';
import BlindResultPanel from '../components/BlindResultPanel';
import { detectBlind } from '../api/detection';
import type { BlindDetectionResult, DetectionStatus } from '../types';

export default function BlindDetection() {
  const [file, setFile] = useState<File | null>(null);
  const [status, setStatus] = useState<DetectionStatus>('idle');
  const [result, setResult] = useState<BlindDetectionResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleAnalyze = async () => {
    if (!file) return;
    setStatus('analyzing');
    setError(null);
    setResult(null);
    try {
      const data = await detectBlind(file);
      setResult(data);
      setStatus('done');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Analysis failed. Is the backend running?';
      setError(message);
      setStatus('error');
    }
  };

  const handleReset = () => {
    setFile(null);
    setResult(null);
    setError(null);
    setStatus('idle');
  };

  return (
    <div className="min-h-screen pt-24 pb-16 px-6">
      <div className="max-w-5xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <div className="flex items-center gap-2 text-purple-400 text-sm font-medium mb-2">
            <Search size={14} /> Blind Detection
          </div>
          <h1 className="text-3xl font-bold text-slate-100">Detect Without Original</h1>
          <p className="text-slate-400 mt-1">
            Upload any image — CNN, FFT, and GAN models analyze it for manipulation signatures.
          </p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-5 gap-6">
          {/* Upload panel */}
          <div className="lg:col-span-2 space-y-4">
            <div className="card space-y-4">
              <UploadZone label="Upload Image" file={file} onFileChange={setFile} />

              <button
                onClick={handleAnalyze}
                disabled={!file || status === 'analyzing'}
                className="btn-primary w-full justify-center"
              >
                {status === 'analyzing' ? (
                  <>
                    <Loader2 size={16} className="animate-spin" /> Analyzing…
                  </>
                ) : (
                  <>
                    <Search size={16} /> Analyze Image
                  </>
                )}
              </button>

              {result && (
                <button onClick={handleReset} className="btn-secondary w-full justify-center text-sm">
                  Reset
                </button>
              )}
            </div>

            {/* Info card */}
            <div className="card space-y-3">
              <p className="text-xs font-semibold text-slate-500 uppercase tracking-wider">Models Used</p>
              {[
                { name: 'CNN Artifact Detector', desc: 'Compression artifacts, noise patterns' },
                { name: 'FFT Frequency Analysis', desc: 'Unnatural frequency signatures' },
                { name: 'GAN Fingerprint', desc: 'Generative model traces' },
              ].map(({ name, desc }) => (
                <div key={name} className="flex gap-3">
                  <div className="w-1.5 h-1.5 rounded-full bg-purple-500 mt-1.5 flex-shrink-0" />
                  <div>
                    <p className="text-sm font-medium text-slate-300">{name}</p>
                    <p className="text-xs text-slate-500">{desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>

          {/* Results panel */}
          <div className="lg:col-span-3">
            {status === 'idle' && (
              <div className="card h-full min-h-64 flex flex-col items-center justify-center text-center gap-4 border-dashed">
                <div className="w-16 h-16 rounded-full bg-dark-400 flex items-center justify-center">
                  <Search size={28} className="text-dark-200" />
                </div>
                <div>
                  <p className="text-slate-400 font-medium">No analysis yet</p>
                  <p className="text-slate-600 text-sm">Upload an image and click Analyze</p>
                </div>
              </div>
            )}

            {status === 'analyzing' && (
              <div className="card h-full min-h-64 flex flex-col items-center justify-center gap-4">
                <Loader2 size={40} className="text-purple-500 animate-spin" />
                <div className="text-center">
                  <p className="text-slate-300 font-medium">Running models…</p>
                  <p className="text-slate-500 text-sm">CNN · FFT · GAN fingerprint</p>
                </div>
              </div>
            )}

            {status === 'error' && (
              <div className="card border-red-800/40 flex items-start gap-4">
                <AlertCircle className="text-red-400 flex-shrink-0 mt-0.5" size={20} />
                <div>
                  <p className="font-semibold text-red-400">Analysis Failed</p>
                  <p className="text-slate-400 text-sm mt-1">{error}</p>
                </div>
              </div>
            )}

            {status === 'done' && result && file && (
              <BlindResultPanel result={result} imageFile={file} />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
