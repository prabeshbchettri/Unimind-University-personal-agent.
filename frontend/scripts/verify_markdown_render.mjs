// Deterministic check of the app's Markdown renderer (no browser, no backend).
//
// Builds a tiny SSR entry with Vite (the project's own toolchain, so the JSX
// and react/jsx-runtime handling are exactly the app's), renders sample
// assistant output to static HTML, and asserts that headings, lists, tables,
// code blocks, bold and inline code all produce real elements while no raw
// Markdown or HTML leaks through.
//
// Usage: node scripts/verify_markdown_render.mjs
import { build } from 'vite'
import react from '@vitejs/plugin-react'
import { readdir, rm, writeFile } from 'node:fs/promises'
import { fileURLToPath } from 'node:url'

const SAMPLE = [
  '## Heading two',
  '',
  'A paragraph with **bold**, *italic* and `inline code`.',
  '',
  '- first bullet',
  '- second bullet',
  '',
  '1. first step',
  '2. second step',
  '',
  '| Topic | Hours |',
  '| --- | --- |',
  '| Intro | 4 |',
  '| Simulation | 6 |',
  '',
  '```',
  'print("hello")',
  '```',
  '<strong>raw strong should become an element</strong>',
  '<script>alert(1)</script>',
  '<img src=x onerror="alert(1)">',
].join('\n')

const root = new URL('..', import.meta.url)
const component = new URL('../src/api-markdown.jsx', import.meta.url).href
const entry = new URL('../.tmp-md-entry.jsx', import.meta.url)
const outDir = new URL('../.tmp-md-out/', import.meta.url)

await writeFile(
  entry,
  [
    "import { createElement } from 'react'",
    "import { renderToStaticMarkup } from 'react-dom/server'",
    `import Markdown from ${JSON.stringify(component)}`,
    `export const html = renderToStaticMarkup(createElement(Markdown, { text: ${JSON.stringify(SAMPLE)} }))`,
    '',
  ].join('\n'),
)

try {
  await rm(outDir, { recursive: true, force: true })
  await build({
    root: fileURLToPath(root),
    configFile: false,
    logLevel: 'error',
    plugins: [react()],
    build: {
      ssr: fileURLToPath(entry),
      outDir: fileURLToPath(outDir),
      emptyOutDir: true,
      minify: false,
    },
  })

  const produced = (await readdir(outDir)).find((name) => name.endsWith('.js') || name.endsWith('.mjs'))
  if (!produced) throw new Error('SSR build produced no entry')

  const { html } = await import(new URL(produced, outDir).href)

  const checks = {
    heading: /<h3>/.test(html),
    paragraphBold: /<strong>bold<\/strong>/.test(html),
    inlineCode: /<code>inline code<\/code>/.test(html),
    bulletList: /<ul>[\s\S]*<li>first bullet<\/li>/.test(html),
    numberedList: /<ol>[\s\S]*<li>first step<\/li>/.test(html),
    table: /<table class="answer-table">/.test(html),
    tableHeaders: /<th>Topic<\/th>/.test(html) && /<th>Hours<\/th>/.test(html),
    tableCells: /<td>Intro<\/td>/.test(html) && /<td>4<\/td>/.test(html),
    codeBlock: /<pre class="code-block"><code>print\(&quot;hello&quot;\)<\/code><\/pre>/.test(html),
    noRawMarkdown: !/(^|\n)\s*#{2,4}\s|\|\s*-{3,}/.test(html),
    strongTagRendered: /<strong>raw strong should become an element<\/strong>/.test(html),
    noScriptTag: !/<script/i.test(html),
    noEventHandlers: !/onerror=/i.test(html),
  }

  console.log(JSON.stringify(checks, null, 2))
  const failed = Object.entries(checks).filter(([, ok]) => !ok).map(([name]) => name)
  if (failed.length > 0) {
    console.error('MARKDOWN RENDERER CHECK FAILED ✘:', failed.join(', '))
    process.exitCode = 1
  } else {
    console.log('MARKDOWN RENDERER CHECK PASSED ✔')
  }
} finally {
  await rm(entry, { force: true })
  await rm(outDir, { recursive: true, force: true })
}
