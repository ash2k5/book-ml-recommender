import { Badge } from "@ash2k5/ui";
import type { Book, ScoredBook } from "../lib/api";
import BookCard from "./BookCard";

interface BookRowProps {
  title: string;
  books: Array<Book | ScoredBook>;
}

function matchBadge(book: Book | ScoredBook) {
  if (!("similarity_score" in book)) return undefined;
  return <Badge>{Math.round(book.similarity_score * 100)}% match</Badge>;
}

export default function BookRow({ title, books }: BookRowProps) {
  if (!books.length) return null;
  return (
    <section className="flex flex-col gap-5">
      <h2 className="ds-label-caps text-on-surface-variant">{title}</h2>
      <div className="-mx-6 grid auto-cols-[8.5rem] grid-flow-col gap-6 overflow-x-auto px-6 pb-4 [scrollbar-width:thin] sm:auto-cols-[10rem]">
        {books.map((book) => (
          <BookCard key={book.id} book={book} badge={matchBadge(book)} />
        ))}
      </div>
    </section>
  );
}
