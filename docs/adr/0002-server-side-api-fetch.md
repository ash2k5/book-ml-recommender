# 2. fetch the api server-side with ISR

- Status: accepted
- Date: 2026-06-20

## Context
tome's reads are small and cacheable (book lists, details, recommendations). Unlike site-audit, there
is no long-running request.

## Decision
The web app fetches the API in server components with ISR (`revalidate: 3600`), using a runtime
`API_BASE_URL` (not `NEXT_PUBLIC_`), so the API URL stays server-side and is not shipped to the browser.

## Consequences
Pages are cached and regenerated hourly. No browser CORS is needed for reads. The API URL is set at
runtime (Render/Vercel env), not baked at build.
