/** @type {import('prettier').Config} */
export default {
  plugins: ['prettier-plugin-svelte'],
  singleQuote: true,
  semi: true,
  trailingComma: 'all',
  printWidth: 100,
  overrides: [{ files: '*.svelte', options: { parser: 'svelte' } }],
};
