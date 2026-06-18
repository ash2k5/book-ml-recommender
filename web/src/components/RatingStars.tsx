import { Star } from "lucide-react";

interface RatingStarsProps {
  rating: number | null;
  size?: number;
}

const STARS = [0, 1, 2, 3, 4];

export default function RatingStars({ rating, size = 14 }: RatingStarsProps) {
  const value = rating ?? 0;
  const fillPct = Math.max(0, Math.min(100, (value / 5) * 100));
  return (
    <span
      className="relative inline-flex"
      role="img"
      aria-label={rating == null ? "Not rated" : `Rated ${rating} out of 5`}
    >
      <span className="flex text-outline-variant" aria-hidden>
        {STARS.map((i) => (
          <Star key={i} size={size} />
        ))}
      </span>
      <span
        className="absolute left-0 top-0 flex overflow-hidden text-primary"
        style={{ width: `${fillPct}%` }}
        aria-hidden
      >
        {STARS.map((i) => (
          <Star key={i} size={size} className="shrink-0 fill-current" />
        ))}
      </span>
    </span>
  );
}
