import Link from "next/link";
import { Search } from "lucide-react";
import { ThemeToggle } from "@ash2k5/ui";

export default function SiteHeader() {
  return (
    <header className="sticky top-0 z-40 border-b border-outline-variant bg-[var(--glass-fill)] [backdrop-filter:blur(16px)_saturate(1.1)] [@media(prefers-reduced-transparency:reduce)]:bg-surface [@media(prefers-reduced-transparency:reduce)]:[backdrop-filter:none]">
      <div className="ds-container flex items-center gap-4 py-4 md:gap-8">
        <Link
          href="/"
          className="shrink-0 font-display text-xl leading-none text-on-surface"
        >
          Tome
        </Link>
        <form
          action="/search"
          method="get"
          role="search"
          className="relative max-w-md flex-1"
        >
          <Search
            size={16}
            className="pointer-events-none absolute left-0 top-1/2 -translate-y-1/2 text-on-surface-variant"
            aria-hidden
          />
          <input
            name="q"
            type="search"
            aria-label="Search books or authors"
            placeholder="Search books or authors"
            className="ds-input pl-6!"
          />
        </form>
        <ThemeToggle className="shrink-0" />
      </div>
    </header>
  );
}
