// Verifies the UniMind chat UI end-to-end in headless Chrome (real streaming):
// branding, empty state, right/left alignment, true incremental deltas from
// the backend SSE stream, markdown rendering, citations, history, Enter-to-send.
const DEBUG_HOST = 'http://127.0.0.1:9223'
const APP_URL = 'http://localhost:5174/'

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
      throw new Error('page JS error: ' + JSON.stringify(result.exceptionDetails).slice(0, 400))
    }
    return result.result.value
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const cdp = await Cdp.connect(await getContextId())
await cdp.send('Runtime.enable')
await cdp.send('Page.enable')
await cdp.send('Emulation.setDeviceMetricsOverride', {
  width: 1280, height: 800, deviceScaleFactor: 1, mobile: false,
})
await cdp.send('Page.navigate', { url: APP_URL })
await sleep(2500)

const checks = {}
const fails = []

function record(name, ok, detail) {
  checks[name] = ok
  if (!ok) fails.push(`${name}: ${JSON.stringify(detail)}`)
}

// 1. Header branding.
const header = await cdp.eval(
  `({
    h1: document.querySelector('.sidebar-brand h1')?.textContent ?? '',
    subtitle: document.querySelector('.subtitle')?.textContent ?? '',
    mark: !!document.querySelector('.sidebar-brand .unimind-mark'),
    status: document.querySelector('.status')?.textContent ?? '',
  })`,
)
record('header UniMind', header.h1 === 'UniMind', header)
record('header subtitle', /University Academic Assistant/i.test(header.subtitle), header.subtitle)
record('header mark', header.mark, header)
record('backend badge', /connected/i.test(header.status), header.status)

// 2. Empty state with suggestions; clicking one sends the prompt.
const empty = await cdp.eval(
  `({
    title: document.querySelector('.empty-state h2')?.textContent ?? '',
    subtitle: document.querySelector('.empty-subtitle')?.textContent ?? '',
    suggestions: Array.from(document.querySelectorAll('.suggestion')).map((b) => b.textContent.trim()),
  })`,
)
record('empty state title', empty.title === 'UniMind', empty)
record('suggestions present', empty.suggestions.length >= 3, empty.suggestions)

// Send the Monte Carlo suggestion by clicking it.
await cdp.eval(
  `(() => {
    const button = Array.from(document.querySelectorAll('.suggestion'))
      .find((b) => /monte carlo/i.test(b.textContent))
    button.click()
  })()`,
)
await sleep(400)

// 3. User bubble on the right, assistant area on the left.
const layout = await cdp.eval(
  `(() => {
    const user = document.querySelector('.turn-user .user-bubble')
    const assistant = document.querySelector('.turn-assistant .assistant-body')
    const chat = document.querySelector('.chat')
    const ur = user.getBoundingClientRect()
    const cr = chat.getBoundingClientRect()
    const userCenter = (ur.left + ur.right) / 2
    return {
      userPresent: !!user,
      assistantPresent: !!assistant,
      userOnRight: userCenter > cr.left + cr.width / 2,
      userText: user?.textContent.trim() ?? '',
      assistantName: document.querySelector('.assistant-name')?.textContent ?? '',
    }
  })()`,
)
record('user bubble right', layout.userPresent && layout.userOnRight, layout)
record('user text sent', /monte carlo/i.test(layout.userText), layout.userText)
record('assistant left area', layout.assistantPresent && layout.assistantName === 'UniMind', layout)

// 4. TRUE streaming: the answer text must grow between samples (backend SSE).
const growth = await cdp.eval(
  `(() => new Promise((resolve) => {
    const readLen = () => (document.querySelector('.turn-assistant .markdown')?.textContent ?? '').length
    const samples = []
    const start = Date.now()
    const timer = setInterval(() => {
      samples.push({ t: Date.now() - start, len: readLen() })
      const done = document.querySelector('.turn-assistant .answer-meta')
      if (done || Date.now() - start > 120000) { clearInterval(timer); resolve(samples) }
    }, 250)
  }))()`,
)
const grewSamples = growth.filter((s, i) => i > 0 && s.len > growth[0].len && growth[0].len >= 0)
const sawIntermediate = growth.some((s, i) => i > 0 && s.len > growth[0].len && !growth[i + 1] === undefined && false)
const lenValues = growth.map((s) => s.len)
const distinctGrowth = lenValues.some((len, i) => i > 0 && len > lenValues[i - 1])
record('streaming text grew incrementally', growth.length >= 3 && distinctGrowth, growth.slice(0, 6))
const finalEntry = await cdp.eval(
  `(() => ({
    answerLen: (document.querySelector('.turn-assistant .markdown')?.textContent ?? '').length,
    hasCursor: !!document.querySelector('.stream-cursor'),
    metaVisible: !!document.querySelector('.turn-assistant .answer-meta'),
    strategy: document.querySelector('.turn-assistant .meta-tags')?.textContent ?? '',
    sources: Array.from(document.querySelectorAll('.turn-assistant .sources-list li')).map((li) => li.textContent.trim()),
  }))()`,
)
record('final answer present', finalEntry.answerLen > 100, finalEntry.answerLen)
record('citations shown', finalEntry.sources.length >= 1, finalEntry.sources)
record('cursor gone after finish', finalEntry.hasCursor === false, finalEntry)

// 5. Second turn: history kept + Enter-to-send works.
await cdp.eval(
  `(() => {
    const textarea = document.querySelector('textarea[aria-label="Question"]')
    const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set
    setter.call(textarea, 'Hello')
    textarea.dispatchEvent(new Event('input', { bubbles: true }))
    textarea.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }))
  })()`,
)
await sleep(1200)
const history = await cdp.eval(
  `(() => ({
    turns: document.querySelectorAll('.turn').length,
    userMessages: Array.from(document.querySelectorAll('.turn-user .user-bubble')).map((p) => p.textContent.trim()),
    greeting: Array.from(document.querySelectorAll('.turn-assistant'))
      .map((t) => t.textContent)
      .find((text) => /How can I help you with your university materials/i.test(text)) ?? '',
  }))()`,
)
record('history kept (2 user turns)', history.userMessages.length === 2, history.userMessages)
record('greeting answered conversationally', /How can I help you/i.test(history.greeting), history.greeting)

// 6. Mobile viewport: no horizontal overflow, alignment kept.
await cdp.send('Emulation.setDeviceMetricsOverride', {
  width: 390, height: 780, deviceScaleFactor: 2, mobile: true,
})
await sleep(600)
const mobile = await cdp.eval(
  `(() => {
    const user = document.querySelector('.turn-user .user-bubble')
    const chat = document.querySelector('.chat')
    const ur = user.getBoundingClientRect()
    const cr = chat.getBoundingClientRect()
    return {
      overflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
      userStillRight: (ur.left + ur.right) / 2 > cr.left + cr.width / 2,
      assistantVisible: !!document.querySelector('.turn-assistant .assistant-body'),
      composerVisible: !!document.querySelector('.composer textarea'),
    }
  })()`,
)
record('mobile no horizontal overflow', mobile.overflow === false, mobile)
record('mobile user still right', mobile.userStillRight, mobile)
record('mobile composer visible', mobile.composerVisible, mobile)

// 7. RAG regression: out-of-corpus question declines via the stream.
await cdp.eval(
  `(() => {
    const textarea = document.querySelector('textarea[aria-label="Question"]')
    const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set
    setter.call(textarea, 'Who won the football World Cup?')
    textarea.dispatchEvent(new Event('input', { bubbles: true }))
    textarea.dispatchEvent(new KeyboardEvent('keydown', { key: 'Enter', bubbles: true }))
  })()`,
)
const declineOk = await cdp.eval(
  `(() => new Promise((resolve) => {
    const start = Date.now()
    const timer = setInterval(() => {
      const turns = document.querySelectorAll('.turn-assistant')
      const declined = Array.from(turns).find((t) =>
        /couldn't find enough information/i.test(t.textContent))
      const stillStreaming = document.querySelector('.stream-cursor, .thinking-dot')
      if (declined && !stillStreaming) { clearInterval(timer); resolve(true) }
      if (Date.now() - start > 60000) { clearInterval(timer); resolve(false) }
    }, 400)
  }))()`,
)
record('decline still works (grounding intact)', declineOk, declineOk)

console.log('checks:', JSON.stringify(checks, null, 2))
if (fails.length > 0) {
  console.error('FAILED:', fails.join(' | '))
  process.exit(1)
}
console.log('UNIMIND UI VERIFICATION PASSED ✔')
process.exit(0)
