import Link from "next/link";
import type { ReactNode } from "react";
import type { Book } from "../lib/api";
import { formatRating } from "../lib/format";
import BookCover from "./BookCover";
import RatingStars from "./RatingStars";

interface BookCardProps {
  book: Book;
  badge?: ReactNode;
}

export default function BookCard({ book, badge }: BookCardProps) {
  return (
    <Link
      href={`/books/${book.id}`}
      className="group flex flex-col gap-3 outline-offset-4 focus-visible:outline-2 focus-visible:outline-on-surface"
    >
      <div className="relative aspect-[2/3] overflow-hidden border border-outline-variant bg-surface-container">
        <BookCover
          src={book.cover_url}
          title={book.title}
          className="h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.03]"
        />
        {badge && <div className="absolute left-2 top-2">{badge}</div>}
      </div>
      <div className="flex flex-col gap-1">
        <h3 className="line-clamp-2 font-display text-base leading-snug text-on-surface">
          {book.title}
        </h3>
        <p className="ds-body-sm line-clamp-1 text-on-surface-variant">
          {book.author}
        </p>
        <div className="flex items-center gap-2 pt-1">
          <RatingStars rating={book.rating} />
          <span className="ds-label-sm ds-tabular text-on-surface-variant">
            {formatRating(book.rating)}
          </span>
        </div>
      </div>
    </Link>
  );
}
