import { describe, expect, test } from "vitest";
import { render, screen } from "@testing-library/react";
import RatingStars from "../app/components/RatingStars";

describe("RatingStars", () => {
  test("exposes the numeric rating via an accessible name", () => {
    render(<RatingStars rating={4.27} />);
    expect(
      screen.getByRole("img", { name: "Rated 4.27 out of 5" }),
    ).toBeInTheDocument();
  });

  test("labels a missing rating as not rated", () => {
    render(<RatingStars rating={null} />);
    expect(screen.getByRole("img", { name: "Not rated" })).toBeInTheDocument();
  });
});
