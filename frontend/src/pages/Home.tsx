import { useNavigate } from 'react-router-dom';
import { Search, GitCompare, Zap, Shield, Brain } from 'lucide-react';

export default function Home() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-6 py-24">
      {/* Hero */}
      <div className="text-center mb-16 space-y-4 max-w-2xl">
        <div className="inline-flex items-center gap-2 bg-purple-900/30 border border-purple-700/40 text-purple-300 text-sm font-medium px-4 py-1.5 rounded-full mb-2">
          <Zap size={14} /> AI-Powered Fake Image Detection
        </div>
        <h1 className="text-5xl font-bold leading-tight">
          Expose Image{' '}
          <span className="gradient-text">Manipulation</span>
          <br />
          Instantly
        </h1>
        <p className="text-slate-400 text-lg leading-relaxed">
          Deep learning models analyze images for GAN artifacts, frequency anomalies,
          and pixel-level forgeries. Get explainable results in seconds.
        </p>
      </div>

      {/* Mode cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-3xl">
        {/* Blind Detection */}
        <button
          onClick={() => navigate('/blind')}
          className="group card glow-purple text-left hover:border-purple-600 transition-all duration-300 hover:scale-[1.02] space-y-4"
        >
          <div className="w-14 h-14 rounded-2xl bg-purple-900/40 border border-purple-700/40 flex items-center justify-center group-hover:bg-purple-800/40 transition-colors">
            <Search className="text-purple-400" size={26} />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-100 mb-1">Detect Without Original</h2>
            <p className="text-slate-400 text-sm leading-relaxed">
              Upload a single image. CNN artifact detection, FFT frequency analysis, and
              GAN fingerprint detection reveal manipulations with explainable heatmaps.
            </p>
          </div>
          <ul className="space-y-1.5">
            {['CNN Artifact Detection', 'FFT Frequency Analysis', 'GAN Fingerprint', 'Explainability Heatmap'].map(
              (f) => (
                <li key={f} className="flex items-center gap-2 text-sm text-slate-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-purple-500 flex-shrink-0" />
                  {f}
                </li>
              )
            )}
          </ul>
          <div className="btn-primary w-full justify-center text-sm group-hover:bg-purple-700">
            Start Blind Detection →
          </div>
        </button>

        {/* Comparative Detection */}
        <button
          onClick={() => navigate('/compare')}
          className="group card text-left hover:border-blue-600 transition-all duration-300 hover:scale-[1.02] space-y-4"
          style={{ boxShadow: '0 0 30px rgba(59, 130, 246, 0.08)' }}
        >
          <div className="w-14 h-14 rounded-2xl bg-blue-900/30 border border-blue-700/30 flex items-center justify-center group-hover:bg-blue-800/30 transition-colors">
            <GitCompare className="text-blue-400" size={26} />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-100 mb-1">Compare With Original</h2>
            <p className="text-slate-400 text-sm leading-relaxed">
              Upload the original alongside a suspected fake. SSIM structural comparison
              and pixel-level analysis highlights manipulated regions.
            </p>
          </div>
          <ul className="space-y-1.5">
            {['SSIM Structural Similarity', 'Pixel Difference Map', 'Manipulated Region Highlight', 'Visual Diff Overlay'].map(
              (f) => (
                <li key={f} className="flex items-center gap-2 text-sm text-slate-400">
                  <span className="w-1.5 h-1.5 rounded-full bg-blue-500 flex-shrink-0" />
                  {f}
                </li>
              )
            )}
          </ul>
          <div className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold px-6 py-3 rounded-xl transition-all duration-200 flex items-center justify-center gap-2 text-sm">
            Start Comparison →
          </div>
        </button>
      </div>

      {/* Features row */}
      <div className="grid grid-cols-3 gap-6 mt-14 max-w-2xl w-full">
        {[
          { icon: Brain, label: 'Deep Learning', desc: 'PyTorch CNN trained on forgery datasets' },
          { icon: Shield, label: 'Explainable AI', desc: 'Grad-CAM heatmaps show where forgery occurs' },
          { icon: Zap, label: 'Fast Analysis', desc: 'Results in under 5 seconds' },
        ].map(({ icon: Icon, label, desc }) => (
          <div key={label} className="text-center space-y-2">
            <div className="w-10 h-10 rounded-xl bg-dark-400 border border-dark-300 flex items-center justify-center mx-auto">
              <Icon size={18} className="text-purple-400" />
            </div>
            <p className="text-sm font-semibold text-slate-200">{label}</p>
            <p className="text-xs text-slate-500 leading-relaxed">{desc}</p>
          </div>
        ))}
      </div>
    </div>
  );
}
