"use client";

import { useRouter } from "next/navigation";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@ash2k5/ui";

interface GenreFilterProps {
  genres: string[];
  current: string;
}

const ALL = "__all__";

export default function GenreFilter({ genres, current }: GenreFilterProps) {
  const router = useRouter();
  return (
    <Select
      value={current || ALL}
      onValueChange={(value) =>
        router.push(
          value === ALL ? "/" : `/?genre=${encodeURIComponent(value)}`,
        )
      }
    >
      <SelectTrigger className="min-w-44" aria-label="Filter by genre">
        <SelectValue />
      </SelectTrigger>
      <SelectContent>
        <SelectItem value={ALL}>All books</SelectItem>
        {genres.map((genre) => (
          <SelectItem key={genre} value={genre}>
            {genre}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  );
}
