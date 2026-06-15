import { describe, expect, test } from "vitest";
import { render, screen } from "@testing-library/react";
import BookGrid from "../app/components/BookGrid";
import type { Book } from "../app/lib/api";

function makeBook(id: string, title: string): Book {
  return {
    id,
    title,
    author: "Author",
    genre: "Fiction",
    genres: "Fiction",
    description: "",
    rating: 4,
    num_ratings: 100,
    year: 2000,
    publisher: "Pub",
    cover_url: "https://example.com/c.jpg",
  };
}

describe("BookGrid", () => {
  test("renders the empty label when there are no books", () => {
    render(<BookGrid books={[]} emptyLabel="Nothing here." />);
    expect(screen.getByText("Nothing here.")).toBeInTheDocument();
    expect(screen.queryByRole("link")).not.toBeInTheDocument();
  });

  test("renders one linked card per book", () => {
    render(
      <BookGrid books={[makeBook("1", "A"), makeBook("2", "B")]} />,
    );
    expect(screen.getAllByRole("link")).toHaveLength(2);
  });
});
