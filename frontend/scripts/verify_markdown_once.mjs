// One-off CDP check: structured (Markdown) answers render in the real UI
// (headings, tables, lists) and citations survive.
const DEBUG_HOST = 'http://127.0.0.1:9223'
const APP_URL = 'http://localhost:5174/'
const QUESTION = 'Explain Monte Carlo simulation in detail.'

async function getContextId() {
  const list = await (await fetch(`${DEBUG_HOST}/json`)).json()
  const page = list.find((t) => t.type === 'page')
  if (!page) throw new Error('no page target in headless Chrome')
  return page.webSocketDebuggerUrl
}

class Cdp {
  constructor(ws) {
    this.ws = ws
    this.id = 0
    this.pending = new Map()
    this.ws.addEventListener('message', (event) => {
      const msg = JSON.parse(event.data)
      if (msg.id && this.pending.has(msg.id)) {
        const { resolve, reject } = this.pending.get(msg.id)
        this.pending.delete(msg.id)
        if (msg.error) reject(new Error(JSON.stringify(msg.error)))
        else resolve(msg.result)
      }
    })
  }
  static async connect(url) {
    const ws = new WebSocket(url)
    await new Promise((resolve, reject) => {
      ws.addEventListener('open', () => resolve(), { once: true })
      ws.addEventListener('error', () => reject(new Error('websocket error')), { once: true })
    })
    return new Cdp(ws)
  }
  send(method, params = {}) {
    const id = ++this.id
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject })
      this.ws.send(JSON.stringify({ id, method, params }))
    })
  }
  async eval(expression) {
    const result = await this.send('Runtime.evaluate', {
      expression,
      awaitPromise: true,
      returnByValue: true,
    })
    if (result.exceptionDetails) {
      throw new Error('page JS error: ' + JSON.stringify(result.exceptionDetails).slice(0, 300))
    }
    return result.result.value
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))

const cdp = await Cdp.connect(await getContextId())
await cdp.send('Runtime.enable')
await cdp.send('Page.enable')
await cdp.send('Page.navigate', { url: APP_URL })
await sleep(2500)

await cdp.eval(
  `(() => new Promise((resolve) => {
    const input = document.querySelector('textarea[aria-label="Question"]')
    const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set
    setter.call(input, ${JSON.stringify(QUESTION)})
    input.dispatchEvent(new Event('input', { bubbles: true }))
    document.querySelector('.composer button').click()
    const start = Date.now()
    const timer = setInterval(() => {
      const meta = document.querySelector('.answer-meta')
      if (meta) { clearInterval(timer); resolve(true) }
      if (Date.now() - start > 120000) { clearInterval(timer); resolve(false) }
    }, 500)
  }))()`,
)

const dom = await cdp.eval(
  `(() => ({
    headings: Array.from(document.querySelectorAll('.markdown h3, .markdown h4')).map((h) => h.textContent.trim()),
    tables: document.querySelectorAll('.markdown table').length,
    tableHeaders: Array.from(document.querySelectorAll('.markdown th')).slice(0, 6).map((th) => th.textContent.trim()),
    listItems: document.querySelectorAll('.markdown li').length,
    paragraphs: document.querySelectorAll('.markdown p').length,
    rawMarkdownLeft: /(^|\\n)\\s*#{2,4}\\s|\\|\\s*-{3,}/.test(document.querySelector('.markdown')?.textContent ?? ''),
    answerLength: (document.querySelector('.markdown')?.textContent ?? '').length,
  }))()`,
)
console.log(JSON.stringify(dom, null, 2))

const ok =
  dom.headings.length >= 2 &&
  dom.tables >= 1 &&
  dom.listItems >= 3 &&
  dom.rawMarkdownLeft === false &&
  dom.answerLength > 300
console.log(ok ? 'MARKDOWN RENDER CHECK PASSED ✔' : 'MARKDOWN RENDER CHECK FAILED ✘')
process.exit(ok ? 0 : 1)
