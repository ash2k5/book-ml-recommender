import { getByGenre, getFeatured, getGenres } from "./lib/api";
import BookGrid from "./components/BookGrid";
import GenreFilter from "./components/GenreFilter";

export default async function HomePage({
  searchParams,
}: {
  searchParams: Promise<{ genre?: string }>;
}) {
  const { genre = "" } = await searchParams;
  const [books, genres] = await Promise.all([
    genre ? getByGenre(genre) : getFeatured(60),
    getGenres(),
  ]);

  return (
    <div className="ds-container">
      <section className="ds-section flex flex-col gap-5">
        <p className="ds-label-caps text-primary">
          Content-based recommendations
        </p>
        <h1 className="ds-display max-w-3xl text-on-surface">
          Find your next book.
        </h1>
        <p className="ds-body-lg max-w-xl text-on-surface-variant">
          A 1,500-title library ranked by what each book is actually about — TF-IDF
          over plot, author, and genre, not just star averages.
        </p>
      </section>

      <section className="flex flex-col gap-8 pb-24">
        <div className="flex flex-wrap items-end justify-between gap-4 border-b border-outline-variant pb-5">
          <h2 className="ds-headline-md text-on-surface">
            {genre || "Featured"}
          </h2>
          <div className="flex items-center gap-4">
            <span className="ds-label-sm text-on-surface-variant">
              {books.length} {books.length === 1 ? "book" : "books"}
            </span>
            <GenreFilter genres={genres} current={genre} />
          </div>
        </div>
        <BookGrid
          books={books}
          emptyLabel={
            genre ? `No books found in ${genre}.` : "No books to show."
          }
        />
      </section>
    </div>
  );
}
