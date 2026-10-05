// Takes the README screenshots (docs/screenshots/*.png) with Playwright.
//
//   pnpm screenshots                 build, serve dist/ with `vite preview` on a free port, take every shot
//   pnpm screenshots overview memory only the shots whose name contains one of these words
//   SHOTS_URL=http://localhost:5173 pnpm screenshots   reuse a running server (no build)
//   SHOTS_NO_BUILD=1 pnpm screenshots                  serve the existing dist/ without rebuilding
//   SHOTS_SCALE=1 pnpm screenshots                     deviceScaleFactor 1 (default 2) → ~4× smaller files
//   SHOTS_DEBUG=1 pnpm screenshots                     log the viewport after each zoom / pan
//
// Every shot gets a fresh browser context (empty localStorage → the default CS336 graph) at 1440×900.
// Chromium must be installed into the project first (`pnpm screenshots:install`):
//   PLAYWRIGHT_BROWSERS_PATH=./.playwright-browsers pnpm exec playwright install chromium
// The script launches it with `channel: 'chromium'` (full Chromium in headless mode, no separate headless shell).
import { spawn, spawnSync } from 'node:child_process'
import { mkdirSync } from 'node:fs'
import { createServer } from 'node:net'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import { chromium } from 'playwright'

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..')
const OUT = join(ROOT, 'docs', 'screenshots')
const VIEWPORT = { width: 1440, height: 900 }
const SCALE = Number(process.env.SHOTS_SCALE ?? 2) // deviceScaleFactor (retina); SHOTS_SCALE=1 for smaller files

// ───────────────────────────── server ─────────────────────────────

function freePort() {
  return new Promise((resolve, reject) => {
    const srv = createServer()
    srv.once('error', reject)
    srv.listen(0, '127.0.0.1', () => {
      const { port } = srv.address()
      srv.close(() => resolve(port))
    })
  })
}

async function waitFor(url, timeoutMs = 30_000) {
  const start = Date.now()
  while (Date.now() - start < timeoutMs) {
    try {
      const res = await fetch(url)
      if (res.ok) return
    } catch {
      // not up yet
    }
    await new Promise((r) => setTimeout(r, 250))
  }
  throw new Error(`Server at ${url} did not come up within ${timeoutMs} ms`)
}

async function startServer() {
  if (process.env.SHOTS_URL) return { url: process.env.SHOTS_URL, stop: () => {} }
  if (!process.env.SHOTS_NO_BUILD) {
    console.log('Building…')
    const build = spawnSync('pnpm', ['build'], { cwd: ROOT, stdio: 'inherit' })
    if (build.status !== 0) throw new Error('pnpm build failed')
  }
  const port = await freePort()
  const vite = join(ROOT, 'node_modules', '.bin', 'vite')
  const proc = spawn(vite, ['preview', '--port', String(port), '--strictPort', '--host', '127.0.0.1'], { cwd: ROOT, stdio: 'ignore' })
  const url = `http://127.0.0.1:${port}/`
  await waitFor(url)
  console.log(`Serving dist/ at ${url}`)
  return { url, stop: () => proc.kill('SIGTERM') }
}

// ───────────────────────────── helpers ─────────────────────────────

const pause = (page, ms = 400) => page.waitForTimeout(ms)

/** Open the app in a fresh context (default graph) and wait until the canvas has settled. */
async function openApp(browser, url) {
  const context = await browser.newContext({ viewport: VIEWPORT, deviceScaleFactor: SCALE })
  const page = await context.newPage()
  await page.goto(url)
  await page.locator('.react-flow__node').first().waitFor()
  await pause(page, 800)
  return { context, page }
}

/** Screen box of a node (by React Flow id). */
async function nodeBox(page, id) {
  const box = await page.locator(`.react-flow__node[data-id="${id}"]`).boundingBox()
  if (!box) throw new Error(`Node ${id} is not visible`)
  return box
}

/** Click a node's body (top-left area, away from ports). */
async function clickNode(page, id, dx = 30, dy = 22) {
  const b = await nodeBox(page, id)
  await page.mouse.click(b.x + dx, b.y + dy)
  await pause(page)
}

/** Zoom with Ctrl+wheel around a screen point (React Flow: pinch / ⌘/Ctrl+wheel zooms). */
async function zoomAt(page, x, y, deltaY, steps = 1) {
  await page.mouse.move(x, y)
  await page.keyboard.down('Control')
  for (let i = 0; i < steps; i++) {
    await page.mouse.wheel(0, deltaY)
    await pause(page, 60)
  }
  await page.keyboard.up('Control')
  await pause(page, 500)
}

/** Drag the empty canvas to pan (plain drag on the pane pans). */
async function pan(page, from, by) {
  await page.mouse.move(from.x, from.y)
  await page.mouse.down()
  await page.mouse.move(from.x + by.x / 2, from.y + by.y / 2, { steps: 5 })
  await page.mouse.move(from.x + by.x, from.y + by.y, { steps: 5 })
  await page.mouse.up()
  await pause(page)
}

async function save(page, name, opts = {}) {
  const path = join(OUT, `${name}.png`)
  await page.screenshot({ path, ...opts })
  console.log(`  saved docs/screenshots/${name}.png`)
}

/** React Flow viewport: translate (x, y) and zoom. */
async function viewport(page) {
  return page.evaluate(() => {
    const el = document.querySelector('.react-flow__viewport')
    const m = /translate\(([-\d.e]+)px,\s*([-\d.e]+)px\)\s*scale\(([-\d.e]+)\)/.exec(el.style.transform)
    return { x: Number(m[1]), y: Number(m[2]), zoom: Number(m[3]) }
  })
}

const paneBox = (page) => page.locator('.react-flow').boundingBox()

/**
 * Pan screen point (sx, sy) to the centre of the canvas, then zoom to `zoom` around it.
 * Uses the same wheel events as a trackpad: Ctrl+wheel zooms (React Flow: zoom × 2^(−deltaY·0.02) on macOS
 * user agents, × 2^(−deltaY·0.002) elsewhere), a plain wheel pans by deltaY·0.5 screen px. Chromium divides
 * synthetic wheel deltas by the device scale factor, hence `* SCALE`.
 */
async function focusAt(page, sx, sy, zoom) {
  const pane = await paneBox(page)
  const mac = await page.evaluate(() => navigator.userAgent.includes('Mac'))
  const cx = pane.x + pane.width / 2
  const cy = pane.y + pane.height / 2
  await page.mouse.move(cx, cy)
  await page.mouse.wheel(2 * (sx - cx) * SCALE, 2 * (sy - cy) * SCALE) // pan the point to the centre (works for off-screen points too)
  await pause(page, 200)
  const v = await viewport(page)
  await page.keyboard.down('Control')
  await page.mouse.wheel(0, (-Math.log2(zoom / v.zoom) / (mac ? 0.02 : 0.002)) * SCALE) // zoom around the centre
  await page.keyboard.up('Control')
  await pause(page, 600)
  if (process.env.SHOTS_DEBUG) console.log('  viewport', await viewport(page), { sx, sy, zoom })
}

/** Centre a node on the canvas at the given zoom (it must be visible at the current zoom level). */
async function focusNode(page, id, zoom, offset = { x: 0, y: 0 }) {
  const b = await nodeBox(page, id)
  await focusAt(page, b.x + b.width / 2 + offset.x, b.y + b.height / 2 + offset.y, zoom)
}

async function closeDrawer(page) {
  await page.getByRole('button', { name: 'Details' }).click()
  await pause(page)
}

/**
 * The analysis panel body is 288 px tall and scrolls. For the analysis shots, make the window taller and the
 * panel body `height` px (display-only style tweak) so a whole tab fits in one picture, scrolled to the top.
 */
async function tallAnalysisPanel(page, height = 540, windowHeight = VIEWPORT.height) {
  if (windowHeight !== VIEWPORT.height) await page.setViewportSize({ width: VIEWPORT.width, height: windowHeight })
  await page.evaluate((h) => {
    const body = document.querySelector('section .h-72')
    if (body) {
      body.style.height = `${h}px`
      body.scrollTop = 0
    }
  }, height)
  await pause(page)
}

const union = (boxes, pad = 0) => {
  const x = Math.min(...boxes.map((b) => b.x)) - pad
  const y = Math.min(...boxes.map((b) => b.y)) - pad
  const r = Math.max(...boxes.map((b) => b.x + b.width)) + pad
  const btm = Math.max(...boxes.map((b) => b.y + b.height)) + pad
  return { x: Math.max(0, x), y: Math.max(0, y), width: Math.min(VIEWPORT.width, r) - Math.max(0, x), height: Math.min(VIEWPORT.height, btm) - Math.max(0, y) }
}

const hyperparamInput = (page, key) => page.locator(`xpath=//span[normalize-space()="${key}"]/following-sibling::input[1]`)

// ───────────────────────────── shots ─────────────────────────────

const SHOTS = {
  /** Hero: the default CS336 model at the start view (blocks open, attention / SwiGLU as cards), drawer = model summary. */
  async overview(page) {
    await focusNode(page, 'b1.attn', 0.42, { x: -40, y: -20 })
    await save(page, 'overview')
  },

  /** Semantic zoom: three canvas crops side by side (blocks as cards / block internals / MHA internals). */
  async 'zoom-levels'(page, { browser }) {
    await closeDrawer(page)
    const pane = await paneBox(page)
    const W = 760
    const H = Math.min(780, pane.height)
    const crop = { x: pane.x + (pane.width - W) / 2, y: pane.y + (pane.height - H) / 2, width: W, height: H }
    const shots = []

    // Level 0: zoomed out, Transformer Blocks are single cards.
    await page.locator('.react-flow__pane').click({ button: 'right', position: { x: pane.width - 60, y: 60 } })
    await page.getByRole('menuitem', { name: 'Fit view' }).click()
    await pause(page, 800)
    await focusNode(page, 'b1', 0.165)
    const b2 = await nodeBox(page, 'b2')
    await focusAt(page, b2.x + b2.width / 2, b2.y - 20, 0.165) // centre between the two blocks
    shots.push({ img: await page.screenshot({ clip: crop }), title: 'Zoomed out', text: 'Each Transformer Block is one card (shapes, params, problems).' })

    // Level 1: block internals; attention and SwiGLU are cards.
    await focusNode(page, 'b1', 0.36, { x: 0, y: -40 })
    shots.push({ img: await page.screenshot({ clip: crop }), title: 'Mid zoom', text: 'Block internals: RMSNorms, residual Adds; attention and SwiGLU still cards.' })

    // Level 2: everything open (multi-head attention internals).
    await focusNode(page, 'b1.attn', 0.85)
    shots.push({ img: await page.screenshot({ clip: crop }), title: 'Zoomed in', text: 'Inside attention: q/k/v projections, split heads, RoPE, SDPA, merge, output_proj.' })

    const html = `<!doctype html><html><body style="margin:0;background:#f8fafc;font-family:Inter,system-ui,sans-serif">
      <div style="display:flex;gap:16px;padding:16px">${shots
        .map(
          (s) => `<figure style="margin:0;width:${W}px">
            <img src="data:image/png;base64,${s.img.toString('base64')}" style="display:block;width:${W}px;height:${H}px;border:1px solid #e2e8f0;border-radius:8px">
            <figcaption style="padding:8px 2px 0;font-size:15px;color:#334155"><b>${s.title}</b> — ${s.text}</figcaption>
          </figure>`,
        )
        .join('')}</div></body></html>`
    const ctx = await browser.newContext({ viewport: { width: 3 * W + 4 * 16, height: H + 80 }, deviceScaleFactor: SCALE })
    const p = await ctx.newPage()
    await p.setContent(html)
    await save(p, 'zoom-levels', { fullPage: true })
    await ctx.close()
  },

  /** Detail drawer for the attention (SDPA) part inside Block 1. */
  async 'drawer-part'(page) {
    await focusNode(page, 'b1.attn', 0.85)
    await clickNode(page, 'b1.sdpa')
    await save(page, 'drawer-part')
  },

  /** Port popover: click SDPA's output port. */
  async 'port-shape'(page) {
    await focusNode(page, 'b1.attn', 0.95, { x: 0, y: 60 })
    const handle = page.locator('.react-flow__node[data-id="b1.sdpa"] .react-flow__handle.source')
    await handle.click()
    await pause(page)
    const pop = await page.getByRole('dialog').boundingBox()
    const node = await nodeBox(page, 'b1.sdpa')
    const q = await nodeBox(page, 'b1.rope_q')
    const v = await nodeBox(page, 'b1.split_v')
    await save(page, 'port-shape', { clip: union([pop, node, q, v], 40) })
  },

  /** Shape error: d_model = 500 is not divisible by num_heads = 16 → split heads errors, downstream unknown. */
  async 'shape-error'(page) {
    await page.getByTitle('Global hyperparameters').click()
    await hyperparamInput(page, 'd_model').fill('500')
    await pause(page)
    await page.mouse.click(700, 600) // close the popover (backdrop)
    await pause(page)
    await focusNode(page, 'b1.attn', 0.8, { x: 0, y: -60 })
    await clickNode(page, 'b1.split_q')
    await save(page, 'shape-error')
  },

  /** Analysis → Parameters with the Attention category highlighted on the canvas. */
  async 'params-tab'(page) {
    await closeDrawer(page)
    await page.getByRole('button', { name: /^Params / }).click()
    await pause(page)
    await page.locator('section button').filter({ has: page.locator('span', { hasText: /^Attention$/ }) }).first().click()
    await pause(page)
    await tallAnalysisPanel(page, 430)
    await save(page, 'params-tab')
  },

  /** Analysis → Memory in Train mode, attention probabilities highlighted on the canvas. */
  async 'memory-tab'(page) {
    await closeDrawer(page)
    await page.locator('header button', { hasText: 'Mem' }).click()
    await pause(page)
    await page.locator('section button').filter({ has: page.locator('span', { hasText: /^Attention probs/ }) }).first().click()
    await pause(page)
    await tallAnalysisPanel(page, 700, 1180)
    await focusNode(page, 'b1', 0.3, { x: 0, y: -40 })
    await save(page, 'memory-tab')
  },

  /** Forward mode + Generation (KV cache). */
  async 'kv-cache'(page) {
    await closeDrawer(page)
    await page.getByRole('radio', { name: 'Forward' }).click()
    await page.locator('header button', { hasText: 'Mem' }).click()
    await pause(page)
    await page.getByLabel('Generation (KV cache)').check()
    await pause(page)
    await tallAnalysisPanel(page, 700, 1180)
    await save(page, 'kv-cache')
  },

  /** Drag from token_embeddings' output and let go on empty canvas → quick-add menu. */
  async 'quick-add'(page) {
    await focusNode(page, 'embed', 0.9, { x: 120, y: 120 })
    const handle = await page.locator('.react-flow__node[data-id="embed"] .react-flow__handle.source').boundingBox()
    const from = { x: handle.x + handle.width / 2, y: handle.y + handle.height / 2 }
    const to = { x: from.x + 230, y: from.y + 30 }
    await page.mouse.move(from.x, from.y)
    await page.mouse.down()
    await page.mouse.move(from.x + 60, from.y + 20, { steps: 5 })
    await page.mouse.move(to.x, to.y, { steps: 10 })
    await page.mouse.up()
    await pause(page)
    await page.getByPlaceholder('Search parts…').fill('norm')
    await pause(page)
    const menu = await page.getByRole('dialog', { name: 'Add a part' }).boundingBox()
    const embed = await nodeBox(page, 'embed')
    await save(page, 'quick-add', { clip: union([menu, embed], 60) })
  },

  /** Right-click Block 2 → Put in a frame, rename + recolour it, then the frame's context menu. */
  async 'frames-context-menu'(page) {
    const b = await nodeBox(page, 'b2')
    await focusAt(page, b.x + b.width / 2, b.y + 260, 0.3)
    const b2 = await nodeBox(page, 'b2')
    await page.mouse.click(b2.x + 60, b2.y + 6, { button: 'right' })
    await page.getByRole('menuitem', { name: 'Put in a frame' }).click()
    await pause(page)
    await page.getByRole('textbox', { name: 'Rename' }).fill('Layer 2 (pre-norm block)')
    await page.getByTitle('#f5f3ff').click()
    await page.getByTitle('#a78bfa').click()
    await pause(page)
    // Right-click the frame's own padding (left of Block 2): the title overlaps Block 1's bottom edge here.
    const f = await page.locator('.react-flow__node-frame').boundingBox()
    await page.mouse.click(f.x + 5, f.y + f.height * 0.4, { button: 'right' })
    await pause(page)
    await save(page, 'frames-context-menu')
  },

  /** Toolbar → Hyperparams popover. */
  async hyperparams(page) {
    await page.getByTitle('Global hyperparameters').click()
    await pause(page)
    const pop = await page.locator('header .absolute.z-50').boundingBox()
    await save(page, 'hyperparams', { clip: union([{ x: 0, y: 0, width: 10, height: 10 }, pop], 16) })
  },
}

// ───────────────────────────── main ─────────────────────────────

const filters = process.argv.slice(2)
const names = Object.keys(SHOTS).filter((n) => filters.length === 0 || filters.some((f) => n.includes(f)))
mkdirSync(OUT, { recursive: true })

const server = await startServer()
const browser = await chromium.launch({ channel: 'chromium' })
const failed = []
try {
  for (const name of names) {
    console.log(`→ ${name}`)
    const { context, page } = await openApp(browser, server.url)
    try {
      await SHOTS[name](page, { browser, url: server.url })
    } catch (err) {
      failed.push(name)
      console.error(`  ✗ ${name}: ${err instanceof Error ? err.message : err}`)
      await page.screenshot({ path: join(OUT, `_failed-${name}.png`) }).catch(() => {})
    } finally {
      await context.close()
    }
  }
} finally {
  await browser.close()
  server.stop()
}
if (failed.length) {
  console.error(`Failed: ${failed.join(', ')}`)
  process.exitCode = 1
}
