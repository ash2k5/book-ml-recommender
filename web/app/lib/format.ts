export function formatRating(rating: number | null): string {
  return rating == null ? "—" : rating.toFixed(2);
}

export function formatCount(count: number | null): string {
  if (!count) return "";
  if (count >= 1_000_000) return `${(count / 1_000_000).toFixed(1)}M`;
  if (count >= 1_000) return `${(count / 1_000).toFixed(1)}K`;
  return String(count);
}

export function genreList(genres: string): string[] {
  return genres
    .split(",")
    .map((genre) => genre.trim())
    .filter(Boolean);
}
