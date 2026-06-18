import { defineConfig, globalIgnores } from "eslint/config";
import nextVitals from "eslint-config-next/core-web-vitals";

const eslintConfig = defineConfig([
  ...nextVitals,
  globalIgnores([
    ".next/**",
    "out/**",
    "build/**",
    "next-env.d.ts",
    "src/lib/schema.d.ts",
  ]),
  {
    // External book covers from one host; next/image adds an allowlist + an
    // image-optimization round trip with no benefit over a lazy <img> here.
    rules: { "@next/next/no-img-element": "off" },
  },
]);

export default eslintConfig;
