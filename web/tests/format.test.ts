import { describe, expect, test } from "vitest";
import { formatCount, formatRating, genreList } from "../src/lib/format";

describe("formatRating", () => {
  test("renders an em dash for missing ratings", () => {
    expect(formatRating(null)).toBe("—");
  });
  test("fixes to two decimals", () => {
    expect(formatRating(4.3)).toBe("4.30");
    expect(formatRating(5)).toBe("5.00");
  });
});

describe("formatCount", () => {
  test("blanks zero and null", () => {
    expect(formatCount(0)).toBe("");
    expect(formatCount(null)).toBe("");
  });
  test("keeps small counts literal", () => {
    expect(formatCount(950)).toBe("950");
  });
  test("abbreviates thousands and millions", () => {
    expect(formatCount(1500)).toBe("1.5K");
    expect(formatCount(6376780)).toBe("6.4M");
  });
});

describe("genreList", () => {
  test("splits, trims, and drops empties", () => {
    expect(genreList("Fiction, Dystopia ,, Fantasy")).toEqual([
      "Fiction",
      "Dystopia",
      "Fantasy",
    ]);
  });
  test("returns an empty array for an empty string", () => {
    expect(genreList("")).toEqual([]);
  });
});
