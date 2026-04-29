import { GitCompare, Map } from 'lucide-react';
import type { ComparativeResult } from '../types';
import ConfidenceBar from './ConfidenceBar';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000';

interface Props {
  result: ComparativeResult;
  original: File;
  suspected: File;
}

export default function ComparativeResultPanel({ result, original, suspected }: Props) {
  const simPct = Math.round(result.similarity_score * 100);
  const origUrl = URL.createObjectURL(original);
  const suspUrl = URL.createObjectURL(suspected);

  return (
    <div className="space-y-6 animate-in fade-in duration-500">
      {/* Verdict */}
      <div
        className={`card glow-purple border ${
          result.is_manipulated ? 'border-red-700/50' : 'border-green-700/50'
        } flex items-center gap-4`}
      >
        <div
          className={`w-16 h-16 rounded-2xl flex items-center justify-center flex-shrink-0 ${
            result.is_manipulated ? 'bg-red-900/30' : 'bg-green-900/30'
          }`}
        >
          <GitCompare
            className={result.is_manipulated ? 'text-red-400' : 'text-green-400'}
            size={32}
          />
        </div>
        <div>
          <p className="text-sm text-slate-400 uppercase tracking-widest font-medium">Comparison Result</p>
          <p className={`text-3xl font-bold ${result.is_manipulated ? 'text-red-400' : 'text-green-400'}`}>
            {result.is_manipulated ? 'MANIPULATED' : 'AUTHENTIC'}
          </p>
          <p className="text-slate-400 text-sm mt-0.5">
            Similarity: <span className="text-slate-200 font-semibold">{simPct}%</span>
            {result.regions_count > 0 && (
              <> · <span className="text-red-400 font-semibold">{result.regions_count} altered region{result.regions_count > 1 ? 's' : ''}</span> detected</>
            )}
          </p>
        </div>
      </div>

      {/* Similarity bar */}
      <div className="card">
        <ConfidenceBar
          label="Structural Similarity (SSIM)"
          value={result.similarity_score}
          color={result.similarity_score >= 0.9 ? 'green' : result.similarity_score >= 0.7 ? 'yellow' : 'red'}
        />
      </div>

      {/* Image comparison grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div className="card space-y-3">
          <p className="text-sm font-medium text-slate-400">Original Image</p>
          <img src={origUrl} alt="original" className="w-full rounded-xl object-contain max-h-56" />
        </div>
        <div className="card space-y-3">
          <p className="text-sm font-medium text-slate-400">Suspected Fake</p>
          <img src={suspUrl} alt="suspected" className="w-full rounded-xl object-contain max-h-56" />
        </div>
      </div>

      {/* Diff maps */}
      {(result.diff_map_url || result.highlighted_url) && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {result.diff_map_url && (
            <div className="card space-y-3">
              <p className="text-sm font-medium text-slate-400 flex items-center gap-1.5">
                <Map size={14} className="text-purple-400" /> Difference Map
              </p>
              <img
                src={`${API_BASE}${result.diff_map_url}`}
                alt="diff map"
                className="w-full rounded-xl object-contain max-h-56"
              />
            </div>
          )}
          {result.highlighted_url && (
            <div className="card space-y-3">
              <p className="text-sm font-medium text-slate-400">Highlighted Regions</p>
              <img
                src={`${API_BASE}${result.highlighted_url}`}
                alt="highlighted"
                className="w-full rounded-xl object-contain max-h-56"
              />
            </div>
          )}
        </div>
      )}
    </div>
  );
}
