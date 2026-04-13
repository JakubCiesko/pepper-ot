const js = require("@eslint/js");
const globals = require("globals");
const html = require("eslint-plugin-html");

module.exports = [
  js.configs.recommended,
  {
    files: ["**/*.js"],
    ignores: ["node_modules/**"],
    languageOptions: {
      ecmaVersion: "latest",
      sourceType: "module",
      globals: {
        ...globals.browser
      }
    },
    rules: {
      "no-unused-vars": ["warn", { "argsIgnorePattern": "^_" }],
      "no-unreachable": "error",
      "no-undef": "error",
      "no-useless-assignment": "error"
    }
  },
  {
    files: ["**/*.html"],
    plugins: {
      html
    }
  }
];
