import type { Metadata } from "next";
import Link from "next/link";
import { notFound } from "next/navigation";
import { ArrowLeft } from "lucide-react";
import { Badge } from "@ash2k5/ui";
import { getBook, getRecommendations, getRelated } from "../../../lib/api";
import { formatCount, formatRating, genreList } from "../../../lib/format";
import BookCover from "../../../components/BookCover";
import BookRow from "../../../components/BookRow";
import RatingStars from "../../../components/RatingStars";

export async function generateMetadata({
  params,
}: {
  params: Promise<{ id: string }>;
}): Promise<Metadata> {
  const { id } = await params;
  const book = await getBook(id);
  return { title: book ? `${book.title} — ${book.author}` : "Book not found" };
}

export default async function BookPage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  const book = await getBook(id);
  if (!book) notFound();

  const [recommendations, related] = await Promise.all([
    getRecommendations(id, 8),
    getRelated(id),
  ]);
  const genres = genreList(book.genres);

  return (
    <div className="ds-container ds-section flex flex-col gap-16">
      <Link
        href="/"
        className="ds-label-sm inline-flex w-fit items-center gap-2 text-on-surface-variant transition-colors hover:text-on-surface"
      >
        <ArrowLeft size={14} aria-hidden /> Back to library
      </Link>

      <article className="grid gap-10 md:grid-cols-[minmax(0,18rem)_1fr] md:gap-14">
        <div className="aspect-[2/3] w-full max-w-xs overflow-hidden border border-outline-variant bg-surface-container">
          <BookCover
            src={book.cover_url}
            title={book.title}
            className="h-full w-full object-cover"
          />
        </div>

        <div className="flex flex-col gap-5">
          <div className="flex flex-wrap items-center gap-2">
            {book.genre && <Badge>{book.genre}</Badge>}
            {book.year ? <Badge>{book.year}</Badge> : null}
          </div>
          <h1 className="ds-headline-lg text-on-surface">{book.title}</h1>
          <p className="ds-body-lg text-on-surface-variant">by {book.author}</p>

          <div className="flex flex-wrap items-center gap-3">
            <RatingStars rating={book.rating} size={18} />
            <span className="ds-tabular ds-body-md text-on-surface">
              {formatRating(book.rating)}
            </span>
            {book.num_ratings ? (
              <span className="ds-label-sm text-on-surface-variant">
                {formatCount(book.num_ratings)} ratings
              </span>
            ) : null}
          </div>

          {book.description && (
            <p className="ds-body-md max-w-2xl whitespace-pre-line text-on-surface-variant">
              {book.description}
            </p>
          )}

          <dl className="flex flex-wrap gap-x-12 gap-y-4 pt-2">
            {book.publisher && (
              <div className="flex flex-col gap-1">
                <dt className="ds-label-sm text-on-surface-variant">
                  Publisher
                </dt>
                <dd className="ds-body-sm text-on-surface">{book.publisher}</dd>
              </div>
            )}
            {genres.length > 1 && (
              <div className="flex max-w-md flex-col gap-1">
                <dt className="ds-label-sm text-on-surface-variant">Genres</dt>
                <dd className="ds-body-sm text-on-surface">
                  {genres.join(", ")}
                </dd>
              </div>
            )}
          </dl>
        </div>
      </article>

      <BookRow title="You might also like" books={recommendations} />
      {book.genre && (
        <BookRow title={`More in ${book.genre}`} books={related.same_genre} />
      )}
      <BookRow title={`More by ${book.author}`} books={related.same_author} />
    </div>
  );
}
