import { describe, expect, test } from "vitest";
import { render, screen } from "@testing-library/react";
import BookCard from "../src/components/BookCard";
import type { Book } from "../src/lib/api";

const book: Book = {
  id: "42",
  title: "Dune",
  author: "Frank Herbert",
  genre: "Science Fiction",
  genres: "Science Fiction, Fantasy",
  description: "Spice.",
  rating: 4.27,
  num_ratings: 1_200_000,
  year: 1965,
  publisher: "Ace",
  cover_url: "https://example.com/dune.jpg",
};

describe("BookCard", () => {
  test("links to the book detail page", () => {
    render(<BookCard book={book} />);
    expect(screen.getByRole("link")).toHaveAttribute("href", "/books/42");
  });

  test("shows the title, author, and rating", () => {
    render(<BookCard book={book} />);
    expect(screen.getByText("Dune")).toBeInTheDocument();
    expect(screen.getByText("Frank Herbert")).toBeInTheDocument();
    expect(
      screen.getByRole("img", { name: "Rated 4.27 out of 5" }),
    ).toBeInTheDocument();
  });
});
