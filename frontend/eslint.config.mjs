import js from "@eslint/js";
import { defineConfig, globalIgnores } from "eslint/config";
import prettier from "eslint-config-prettier";
import globals from "globals";
import tseslint from "typescript-eslint";

export default defineConfig(
  globalIgnores(["dist/**", "coverage/**", "node_modules/**"]),
  {
    files: ["**/*.{js,mjs,cjs,jsx,ts,mts,cts,tsx}"],
    extends: [js.configs.recommended, tseslint.configs.recommended],
    languageOptions: {
      parserOptions: { ecmaFeatures: { jsx: true } },
    },
  },
  {
    files: ["src/**/*.{js,jsx,ts,tsx}"],
    languageOptions: { globals: globals.browser },
  },
  {
    files: ["*.config.{js,mjs,cjs,ts,mts,cts}"],
    languageOptions: { globals: globals.node },
  },
  prettier,
);
