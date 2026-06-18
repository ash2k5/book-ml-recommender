import { afterEach, beforeEach, describe, expect, test, vi } from "vitest";
import {
  ApiError,
  getBook,
  getByGenre,
  getRecommendations,
  searchBooks,
} from "../src/lib/api";

const fetchMock = vi.fn();

beforeEach(() => {
  fetchMock.mockReset();
  vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
  vi.unstubAllGlobals();
});

function calledUrl(): string {
  return String(fetchMock.mock.calls[0][0]);
}

describe("api client", () => {
  test("encodes the genre into the books query", async () => {
    fetchMock.mockResolvedValue({ ok: true, json: async () => [] });
    await getByGenre("Science Fiction & Fantasy");
    expect(calledUrl()).toContain(
      "/books?genre=Science%20Fiction%20%26%20Fantasy",
    );
  });

  test("passes n through to recommendations", async () => {
    fetchMock.mockResolvedValue({ ok: true, json: async () => [] });
    await getRecommendations("1", 8);
    expect(calledUrl()).toContain("/books/1/recommendations?n=8");
  });

  test("getBook resolves null on 404", async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 404 });
    expect(await getBook("missing")).toBeNull();
  });

  test("getBook surfaces non-404 errors", async () => {
    fetchMock.mockResolvedValue({ ok: false, status: 500 });
    await expect(getBook("boom")).rejects.toBeInstanceOf(ApiError);
  });

  test("searchBooks skips the request for a blank query", async () => {
    expect(await searchBooks("   ")).toEqual([]);
    expect(fetchMock).not.toHaveBeenCalled();
  });
});
