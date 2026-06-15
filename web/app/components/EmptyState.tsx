import type { ReactNode } from "react";

export default function EmptyState({ children }: { children: ReactNode }) {
  return (
    <div className="border border-dashed border-outline-variant px-8 py-16 text-center">
      <p className="ds-body-md text-on-surface-variant">{children}</p>
    </div>
  );
}
