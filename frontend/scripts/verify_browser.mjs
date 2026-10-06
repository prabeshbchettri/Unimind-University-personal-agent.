// Drives the real React UI in headless Chrome over CDP (no extra packages).
// Usage: chrome --headless --remote-debugging-port=9223 ... & node scripts/verify_browser.mjs
const DEBUG_HOST = 'http://127.0.0.1:9223'
const APP_URL = 'http://localhost:5174/'
const QUESTION = 'What are the credit requirements for CS201?'

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

const wsUrl = await getContextId()
const cdp = await Cdp.connect(wsUrl)
await cdp.send('Runtime.enable')

await cdp.send('Page.enable')
await cdp.send('Page.navigate', { url: APP_URL })
await sleep(2500)

// 1. The React app actually rendered.
const rendered = await cdp.eval(
  `({ root: !!document.querySelector('#root .app'), h1: document.querySelector('h1')?.textContent ?? '' })`,
)
console.log('1. rendered:', JSON.stringify(rendered))
if (!rendered.root) throw new Error('React app did not render')

// 2. Health badge from the real /health check.
const badge = await cdp.eval(`document.querySelector('.status')?.textContent ?? ''`)
console.log('2. status badge:', badge)
if (!/connected/i.test(badge)) throw new Error('health badge does not show a real connection: ' + badge)

// 3. Ask a hybrid-routed question through the real input + button.
const asked = await cdp.eval(
  `(() => new Promise((resolve) => {
    const input = document.querySelector('input[aria-label=\"Question\"]')
    const setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set
    setter.call(input, ${JSON.stringify(QUESTION)})
    input.dispatchEvent(new Event('input', { bubbles: true }))
    document.querySelector('.composer button').click()
    const start = Date.now()
    const timer = setInterval(() => {
      const meta = document.querySelector('.metadata')
      if (meta) { clearInterval(timer); resolve({ ok: true, ms: Date.now() - start }) }
      if (Date.now() - start > 90000) { clearInterval(timer); resolve({ ok: false, ms: Date.now() - start }) }
    }, 500)
  }))()`,
)
console.log('3. question answered through the real UI:', JSON.stringify(asked))
if (!asked.ok) throw new Error('no answer metadata appeared within 90s')

// 4. The completion must show question, answer, sources, strategy and provider.
const turn = await cdp.eval(
  `(() => {
    const text = (sel) => Array.from(document.querySelectorAll(sel)).map((el) => el.textContent.trim())
    return {
      questions: text('.message-user'),
      answers: text('.message-assistant'),
      strategy: document.querySelector('.metadata dd')?.textContent ?? '',
      sources: text('.sources li'),
    }
  })()`,
)
console.log('4. user turn:', turn.questions)
console.log('   answer:', turn.answers[0])
console.log('   strategy cells:', turn.strategy)
console.log('   sources rows:', turn.sources)

const answerText = turn.answers.join(' ')
const lower = answerText.toLowerCase()
const checks = {
  questionEchoed: turn.questions.some((q) => q.includes('CS201')),
  answerPresent: answerText.length > 40 && /credit/i.test(answerText),
  strategyShown: /HYBRID/.test(answerText),
  providerShown: /Groq/i.test(answerText),
  sourceShown: turn.sources.some((s) => /cs201_syllabus\.pdf/i.test(s) && /page/i.test(s)),
  onlyBackendCitations: turn.sources.every((s) => s.includes('Page')),
}
console.log('5. assertions:', JSON.stringify(checks, null, 2))

const failed = Object.entries(checks).filter(([, v]) => !v)
if (failed.length > 0) {
  throw new Error('FAILED checks: ' + failed.map(([k]) => k).join(', '))
}
console.log('\nBROWSER VERIFICATION PASSED ✔')
process.exit(0)
