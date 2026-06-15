import type { Metadata } from "next";
import { searchBooks } from "../lib/api";
import BookGrid from "../components/BookGrid";
import EmptyState from "../components/EmptyState";

export const metadata: Metadata = { title: "Search" };

export default async function SearchPage({
  searchParams,
}: {
  searchParams: Promise<{ q?: string }>;
}) {
  const { q = "" } = await searchParams;
  const query = q.trim();
  const books = query ? await searchBooks(query, 48) : [];

  return (
    <div className="ds-container ds-section flex flex-col gap-8">
      <header className="flex flex-col gap-2 border-b border-outline-variant pb-5">
        <p className="ds-label-caps text-primary">Search</p>
        <h1 className="ds-headline-lg text-on-surface">
          {query ? `Results for “${query}”` : "Search the library"}
        </h1>
        {query && (
          <span className="ds-label-sm text-on-surface-variant">
            {books.length} {books.length === 1 ? "match" : "matches"}
          </span>
        )}
      </header>
      {query ? (
        <BookGrid books={books} emptyLabel={`No books match “${query}”.`} />
      ) : (
        <EmptyState>Search by title or author from the bar above.</EmptyState>
      )}
    </div>
  );
}
