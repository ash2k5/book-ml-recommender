import type { Book } from "../lib/api";
import BookCard from "./BookCard";
import EmptyState from "./EmptyState";

interface BookGridProps {
  books: Book[];
  emptyLabel?: string;
}

export default function BookGrid({
  books,
  emptyLabel = "No books to show.",
}: BookGridProps) {
  if (!books.length) return <EmptyState>{emptyLabel}</EmptyState>;
  return (
    <div className="grid grid-cols-2 gap-x-6 gap-y-10 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-6">
      {books.map((book) => (
        <BookCard key={book.id} book={book} />
      ))}
    </div>
  );
}
