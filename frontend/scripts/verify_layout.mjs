// Checks the chat layout at the target laptop sizes: no horizontal overflow,
// sidebar + composer present, conversation column bounded, and captures a
// screenshot per size. Uses the running headless Chrome on port 9223 and the
// app at localhost:5174.
//
// Usage: node scripts/verify_layout.mjs
import { writeFile } from 'node:fs/promises'
import { tmpdir } from 'node:os'
import { join } from 'node:path'

const DEBUG_HOST = 'http://127.0.0.1:9223'
const SIZES = [
  { width: 1366, height: 768 },
  { width: 1440, height: 900 },
  { width: 1920, height: 1080 },
]

async function contextId() {
  const list = await (await fetch(`${DEBUG_HOST}/json`)).json()
  const page = list.find((t) => t.type === 'page')
  if (!page) throw new Error('no page target')
  return page.webSocketDebuggerUrl
}

class Cdp {
  constructor(ws) {
    this.ws = ws
    this.id = 0
    this.pending = new Map()
    ws.addEventListener('message', (event) => {
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
    await new Promise((res, rej) => {
      ws.addEventListener('open', () => res(), { once: true })
      ws.addEventListener('error', () => rej(new Error('ws error')), { once: true })
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
    const r = await this.send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true })
    if (r.exceptionDetails) throw new Error('page error: ' + JSON.stringify(r.exceptionDetails).slice(0, 300))
    return r.result.value
  }
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms))
const cdp = await Cdp.connect(await contextId())
await cdp.send('Runtime.enable')
await cdp.send('Page.enable')

const failures = []
for (const size of SIZES) {
  await cdp.send('Emulation.setDeviceMetricsOverride', { ...size, deviceScaleFactor: 1, mobile: false })
  await sleep(400)
  const m = await cdp.eval(`(() => {
    const de = document.documentElement
    const sidebar = document.querySelector('.sidebar')
    const composer = document.querySelector('.composer')
    const input = document.querySelector('.composer textarea')
    const thread = document.querySelector('.thread')
    const sr = sidebar?.getBoundingClientRect()
    return {
      pageOverflow: de.scrollWidth > de.clientWidth + 1,
      docWidth: de.scrollWidth,
      viewport: de.clientWidth,
      sidebarVisible: !!sr && sr.width > 100 && sr.left >= 0 && sr.right <= de.clientWidth + 1,
      sidebarWidth: sr ? Math.round(sr.width) : 0,
      composerVisible: !!composer && composer.getBoundingClientRect().width > 0,
      inputFits: !!input && input.getBoundingClientRect().right <= de.clientWidth + 1,
      threadWidth: thread ? Math.round(thread.getBoundingClientRect().width) : 0,
      turns: document.querySelectorAll('.turn').length,
    }
  })()`)
  const shot = await cdp.send('Page.captureScreenshot', { format: 'png' })
  const file = join(tmpdir(), `ui-${size.width}x${size.height}.png`)
  await writeFile(file, Buffer.from(shot.data, 'base64'))
  console.log(`${size.width}x${size.height}`, JSON.stringify(m), '->', file)

  const ok =
    !m.pageOverflow && m.sidebarVisible && m.composerVisible && m.inputFits && m.threadWidth <= 780
  if (!ok) failures.push(`${size.width}x${size.height}: ${JSON.stringify(m)}`)
}

if (failures.length) {
  console.error('LAYOUT CHECK FAILED ✘:\n' + failures.join('\n'))
  process.exit(1)
}
console.log('LAYOUT CHECK PASSED ✔')
process.exit(0)
