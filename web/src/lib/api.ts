import type { components } from "./schema";

export type Book = components["schemas"]["Book"];
export type ScoredBook = components["schemas"]["ScoredBook"];
export type Related = components["schemas"]["Related"];

const API_BASE_URL = (
  process.env.API_BASE_URL ?? "https://tome-api.onrender.com"
).replace(/\/+$/, "");

const REVALIDATE_SECONDS = 3600;

export class ApiError extends Error {
  constructor(
    readonly status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

async function getJson<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    next: { revalidate: REVALIDATE_SECONDS },
  });
  if (!res.ok) {
    throw new ApiError(res.status, `GET ${path} failed (${res.status})`);
  }
  return res.json() as Promise<T>;
}

export function getGenres(): Promise<string[]> {
  return getJson<string[]>("/genres");
}

export function getFeatured(limit = 60): Promise<Book[]> {
  return getJson<Book[]>(`/books?limit=${limit}`);
}

export function getByGenre(genre: string): Promise<Book[]> {
  return getJson<Book[]>(`/books?genre=${encodeURIComponent(genre)}`);
}

export async function getBook(id: string): Promise<Book | null> {
  try {
    return await getJson<Book>(`/books/${encodeURIComponent(id)}`);
  } catch (err) {
    if (err instanceof ApiError && err.status === 404) return null;
    throw err;
  }
}

export function getRecommendations(id: string, n = 6): Promise<ScoredBook[]> {
  return getJson<ScoredBook[]>(
    `/books/${encodeURIComponent(id)}/recommendations?n=${n}`,
  );
}

export function getRelated(id: string): Promise<Related> {
  return getJson<Related>(`/books/${encodeURIComponent(id)}/related`);
}

export function searchBooks(query: string, limit = 24): Promise<Book[]> {
  const q = query.trim();
  if (!q) return Promise.resolve([]);
  return getJson<Book[]>(`/search?q=${encodeURIComponent(q)}&limit=${limit}`);
}
