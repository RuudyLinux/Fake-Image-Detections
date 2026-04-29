import { useRef, useState, useCallback } from 'react';
import { Upload, X, Image } from 'lucide-react';

interface Props {
  label: string;
  file: File | null;
  onFileChange: (f: File | null) => void;
  accept?: string;
}

export default function UploadZone({ label, file, onFileChange, accept = 'image/*' }: Props) {
  const inputRef = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  const preview = file ? URL.createObjectURL(file) : null;

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragging(false);
      const dropped = e.dataTransfer.files[0];
      if (dropped && dropped.type.startsWith('image/')) onFileChange(dropped);
    },
    [onFileChange]
  );

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onFileChange(e.target.files?.[0] ?? null);
  };

  return (
    <div className="space-y-2">
      <p className="text-sm font-medium text-slate-400">{label}</p>

      {file && preview ? (
        <div className="relative rounded-xl overflow-hidden border border-dark-300 group">
          <img src={preview} alt="preview" className="w-full h-56 object-cover" />
          <div className="absolute inset-0 bg-dark-900/60 opacity-0 group-hover:opacity-100 transition-opacity flex items-center justify-center gap-3">
            <button
              onClick={() => inputRef.current?.click()}
              className="bg-dark-600 hover:bg-dark-400 text-white p-2 rounded-lg transition-colors"
              title="Change image"
            >
              <Image size={18} />
            </button>
            <button
              onClick={() => onFileChange(null)}
              className="bg-red-900/80 hover:bg-red-800 text-white p-2 rounded-lg transition-colors"
              title="Remove"
            >
              <X size={18} />
            </button>
          </div>
          <div className="absolute bottom-2 left-2 right-2 bg-dark-900/80 rounded-lg px-3 py-1.5 text-xs text-slate-300 truncate">
            {file.name} · {(file.size / 1024).toFixed(1)} KB
          </div>
        </div>
      ) : (
        <div
          onClick={() => inputRef.current?.click()}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-xl h-56 flex flex-col items-center justify-center gap-3 cursor-pointer transition-all duration-200 ${
            dragging
              ? 'border-purple-500 bg-purple-500/10'
              : 'border-dark-300 hover:border-purple-600 hover:bg-dark-500/30'
          }`}
        >
          <div className="w-14 h-14 rounded-full bg-dark-400 flex items-center justify-center">
            <Upload className="text-purple-500" size={24} />
          </div>
          <div className="text-center">
            <p className="text-slate-300 font-medium">Drop image here</p>
            <p className="text-slate-500 text-sm">or click to browse · PNG, JPG, WEBP</p>
          </div>
        </div>
      )}

      <input
        ref={inputRef}
        type="file"
        accept={accept}
        className="hidden"
        onChange={handleChange}
      />
    </div>
  );
}
