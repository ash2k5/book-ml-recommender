interface BookCoverProps {
  src: string;
  title: string;
  className?: string;
}

// Covers come from a single external host and some catalog rows have none, so a
// plain lazy <img> with a typographic fallback is simpler and more robust here
// than next/image (no remote allowlist, no optimization cost on flaky covers).
export default function BookCover({ src, title, className }: BookCoverProps) {
  if (!src) {
    return (
      <div
        className={`flex items-center justify-center bg-surface-container-high p-4 text-center ${className ?? ""}`}
      >
        <span className="ds-label-sm text-on-surface-variant line-clamp-4">
          {title}
        </span>
      </div>
    );
  }
  return (
    <img src={src} alt={`Cover of ${title}`} loading="lazy" className={className} />
  );
}
