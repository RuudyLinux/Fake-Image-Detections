interface Props {
  label: string;
  value: number;
  color?: string;
}

export default function ConfidenceBar({ label, value, color = 'purple' }: Props) {
  const pct = Math.round(value * 100);
  const colorMap: Record<string, string> = {
    purple: 'bg-purple-600',
    green: 'bg-green-500',
    red: 'bg-red-500',
    yellow: 'bg-yellow-500',
    blue: 'bg-blue-500',
  };

  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-sm">
        <span className="text-slate-400">{label}</span>
        <span className="text-slate-200 font-semibold">{pct}%</span>
      </div>
      <div className="h-2 bg-dark-400 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all duration-700 ${colorMap[color] ?? colorMap.purple}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
