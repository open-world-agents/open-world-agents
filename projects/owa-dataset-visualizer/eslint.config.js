import js from "@eslint/js";
import globals from "globals";
import eslintConfigPrettier from "eslint-config-prettier/flat";

export default [
  js.configs.recommended,
  { languageOptions: { globals: globals.browser } },
  eslintConfigPrettier,
];

