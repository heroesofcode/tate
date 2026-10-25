import js from '@eslint/js';
import svelte from 'eslint-plugin-svelte';
import globals from 'globals';
import ts from 'typescript-eslint';

export default ts.config(
  {
    ignores: [
      'dist/**',
      'dev-dist/**',
      'node_modules/**',
      'playwright-report/**',
      'test-results/**',
      'coverage/**',
      'public/**',
    ],
  },
  js.configs.recommended,
  ...ts.configs.strict,
  ...svelte.configs['flat/recommended'],
  ...svelte.configs['flat/prettier'],
  {
    languageOptions: {
      globals: { ...globals.browser, ...globals.node },
    },
    rules: {
      'no-undef': 'off',
      '@typescript-eslint/no-unused-vars': [
        'error',
        { argsIgnorePattern: '^_', varsIgnorePattern: '^_' },
      ],
    },
  },
  {
    files: ['**/*.svelte', '**/*.svelte.ts'],
    languageOptions: {
      parserOptions: {
        parser: ts.parser,
        extraFileExtensions: ['.svelte'],
      },
    },
  },
  {
    files: ['src/sw.ts'],
    languageOptions: {
      globals: { ...globals.serviceworker },
    },
  },
);
