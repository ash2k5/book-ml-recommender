import Link from "next/link";

export default function NotFound() {
  return (
    <div className="ds-container ds-section flex flex-col items-start gap-6">
      <p className="ds-label-caps text-primary">404</p>
      <h1 className="ds-headline-lg text-on-surface">
        We couldn’t find that page.
      </h1>
      <p className="ds-body-md max-w-md text-on-surface-variant">
        The book or page you’re looking for isn’t here. It may have been moved,
        or the link is wrong.
      </p>
      <Link
        href="/"
        className="ds-label-sm inline-flex items-center gap-2 text-on-surface underline underline-offset-4 hover:text-primary"
      >
        Back to the library
      </Link>
    </div>
  );
}
