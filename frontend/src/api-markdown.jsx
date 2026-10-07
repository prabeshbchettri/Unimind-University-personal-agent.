/**
 * Minimal, safe Markdown -> React renderer for grounded assistant answers.
 *
 * Deliberately tiny: answers are produced by our own generation layer and may
 * only use headings, bold, italics, lists, tables and inline citations, so a
 * small hand-written renderer avoids adding a runtime dependency to the app.
 *
 * Safety: the input is NEVER injected as HTML. Text is escaped by React and
 * structure is produced by elements, so model output cannot break out of the
 * UI; unknown Markdown is simply shown as plain text.
 */

/** Inline pass: `**bold**`, `*italic*`, `` `code` ``, and plain text -> elements. */
function renderInline(text, keyPrefix) {
  // Residual model HTML (e.g. "<br>", "<p>", "<strong>x</strong>") must never
  // appear as literal text: the prompt forbids HTML, and this is the safety
  // net for models that emit it anyway.
  const withoutHtml = String(text)
    .replace(/<br\s*\/?>/gi, '\n')
    .replace(/<\/(p|div|li|tr)>/gi, '\n')
    .replace(/<(p|div|li|ul|ol|tr|table|thead|tbody|h[1-6])[^>]*>/gi, '')
    .replace(/<strong[^>]*>(.*?)<\/strong>/gis, '**$1**')
    .replace(/<(b|em|i|code)[^>]*>(.*?)<\/\1>/gis, (_, tag, inner) => {
      if (tag === 'b') return `**${inner}**`
      if (tag === 'em' || tag === 'i') return `*${inner}*`
      return `\`${inner}\``
    })
    .replace(/<[^>]*>/g, '')
  const parts = []
  let rest = withoutHtml
  let key = 0
  const pattern = /\*\*([^*]+)\*\*|\*([^*]+)\*|`([^`]+)`/g
  while (rest.length > 0) {
    const match = pattern.exec(rest)
    if (!match) {
      parts.push(rest)
      break
    }
    if (match.index > 0) parts.push(rest.slice(0, match.index))
    if (match[1] !== undefined) parts.push(<strong key={`${keyPrefix}-${key++}`}>{match[1]}</strong>)
    else if (match[2] !== undefined) parts.push(<em key={`${keyPrefix}-${key++}`}>{match[2]}</em>)
    else parts.push(<code key={`${keyPrefix}-${key++}`}>{match[3]}</code>)
    rest = rest.slice(match.index + match[0].length)
    pattern.lastIndex = 0
  }
  return parts
}

/** Split a table block into header row and body rows. */
function parseTable(lines) {
  const rows = lines
    .filter((line) => /^\s*\|.*\|\s*$/.test(line))
    .map((line) =>
      line
        .trim()
        .replace(/^\|/, '')
        .replace(/\|$/, '')
        .split('|')
        .map((cell) => cell.trim()),
    )
  const header = rows[0]
  // Second row of a Markdown table is the |---|---| separator.
  const body = rows.slice(2).filter((cells) => !cells.every((cell) => /^:?-{3,}:?$/.test(cell)))
  return { header, body }
}

export default function Markdown({ text }) {
  const lines = String(text).split(/\r?\n/)
  const blocks = []
  let i = 0
  let key = 0

  while (i < lines.length) {
    const line = lines[i]

    if (!line.trim()) {
      i += 1
      continue
    }

    // Fenced code blocks: ``` ... ```
    if (line.trim().startsWith('```')) {
      const codeLines = []
      i += 1
      while (i < lines.length && !lines[i].trim().startsWith('```')) {
        codeLines.push(lines[i])
        i += 1
      }
      i += 1 // closing fence (or end of input)
      blocks.push(
        <pre key={`b-${key++}`} className="code-block">
          <code>{codeLines.join('\n')}</code>
        </pre>,
      )
      continue
    }

    // Headings: ## / ###
    const heading = line.match(/^(#{2,4})\s+(.*)$/)
    if (heading) {
      const level = heading[1].length
      const content = renderInline(heading[2], `h-${key}`)
      blocks.push(
        level === 2 ? (
          <h3 key={`b-${key++}`}>{content}</h3>
        ) : (
          <h4 key={`b-${key++}`}>{content}</h4>
        ),
      )
      i += 1
      continue
    }

    // Tables: consecutive | ... | lines
    if (/^\s*\|.*\|\s*$/.test(line)) {
      const tableLines = []
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) {
        tableLines.push(lines[i])
        i += 1
      }
      const { header, body } = parseTable(tableLines)
      blocks.push(
        <table key={`b-${key++}`} className="answer-table">
          {header && (
            <thead>
              <tr>{header.map((cell, c) => <th key={c}>{renderInline(cell, `th-${key}-${c}`)}</th>)}</tr>
            </thead>
          )}
          <tbody>
            {body.map((cells, r) => (
              <tr key={r}>
                {cells.map((cell, c) => (
                  <td key={c}>{renderInline(cell, `td-${key}-${r}-${c}`)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>,
      )
      continue
    }

    // Numbered lists: "1. ", "2) " ...
    if (/^\s*\d+[.)]\s+/.test(line)) {
      const items = []
      while (i < lines.length && /^\s*\d+[.)]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*\d+[.)]\s+/, ''))
        i += 1
      }
      blocks.push(
        <ol key={`b-${key++}`}>
          {items.map((item, n) => (
            <li key={n}>{renderInline(item, `li-${key}-${n}`)}</li>
          ))}
        </ol>,
      )
      continue
    }

    // Bullet lists: "- " or "* "
    if (/^\s*[-*]\s+/.test(line)) {
      const items = []
      while (i < lines.length && /^\s*[-*]\s+/.test(lines[i])) {
        items.push(lines[i].replace(/^\s*[-*]\s+/, ''))
        i += 1
      }
      blocks.push(
        <ul key={`b-${key++}`}>
          {items.map((item, n) => (
            <li key={n}>{renderInline(item, `li-${key}-${n}`)}</li>
          ))}
        </ul>,
      )
      continue
    }

    // Paragraph: consecutive non-structural lines.
    const paragraph = []
    while (
      i < lines.length &&
      lines[i].trim() &&
      !/^(#{2,4})\s+/.test(lines[i]) &&
      !/^\s*\d+[.)]\s+/.test(lines[i]) &&
      !/^\s*[-*]\s+/.test(lines[i]) &&
      !/^\s*\|.*\|\s*$/.test(lines[i])
    ) {
      paragraph.push(lines[i])
      i += 1
    }
    if (paragraph.length > 0) {
      blocks.push(<p key={`b-${key++}`}>{renderInline(paragraph.join(' '), `p-${key}`)}</p>)
    }
  }

  return <div className="markdown">{blocks}</div>
}
