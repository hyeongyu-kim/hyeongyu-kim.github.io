// A small shell; Save an offline copy embeds all measured traces in the browser.
import fs from "node:fs";
import * as sass from "sass";
const read = (path) => fs.readFileSync(path, "utf8");
const site = "https://hyeongyu-kim.github.io/";
let content = read("_pages/lab.md").replace(/^---[\s\S]*?---\s*/, "");
content = content.replace(/\{\{\s*'([^']+)'\s*\|\s*(?:relative_url|absolute_url)\s*\}\}/g, (_, path) => new URL(path, site).href);
if (content.includes("{{") || content.includes("{%")) throw new Error("Unresolved template in the lab shell");
const math = read("assets/js/tta-math.js").replace(/^export /gm, "");
const ui = read("assets/js/research-lab.js").replace(/^import\s*\{[\s\S]*?\}\s*from\s*"\.\/tta-math.js";\s*/, "");
if (/^import /m.test(ui)) throw new Error("Unresolved offline module import");
const css = sass.compileString('@use "editorial"; @use "research-lab";', { loadPaths: ["_sass"], style: "compressed", logger: sass.Logger.silent }).css;
const script = math + "\n" + ui;
new Function(script);
const output = `<!doctype html><html lang="en" data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>What changes at test time? · Hyeongyu Kim</title><style>:root{--global-bg-color:#f8f7f3;--global-text-color:#303c34;--global-text-color-light:#717a70;--global-divider-color:#dfded4;--global-theme-color:#50684e}html[data-theme=dark]{--global-bg-color:#191e1a;--global-text-color:#e0e6dd;--global-text-color-light:#a3afa0;--global-divider-color:#3a443a;--global-theme-color:#a6c09a}${css}\nbody{margin:0;font-family:system-ui,sans-serif;color:var(--global-text-color);background:var(--global-bg-color)}.offline-wrap{max-width:1060px;padding:32px 24px;margin:auto}.offline-heading{display:flex;justify-content:space-between;align-items:center;gap:1rem;margin-bottom:3rem}.offline-heading>a{font-family:Georgia,serif;color:var(--editorial-title-color);font-size:1.2rem}a{color:var(--global-theme-color)}</style></head><body><div class="offline-wrap"><header class="offline-heading"><a href="${site}">Hyeongyu Kim</a><button class="quiet-button" id="offline-theme" type="button">Dark mode</button></header><main class="editorial-content"><header class="editorial-page-heading"><p class="editorial-eyebrow">Test-time adaptation</p><h1>What changes at test time?</h1><p>The same images. Three different places to adapt.</p></header>${content}</main></div><!--TTA_DATA--><script>${script}\ndocument.getElementById("offline-theme").addEventListener("click",function(){const theme=document.documentElement.getAttribute("data-theme")==="dark"?"light":"dark";document.documentElement.setAttribute("data-theme",theme);this.textContent=theme==="dark"?"Light mode":"Dark mode";});</script></body></html>`;
fs.writeFileSync("assets/lab/research-lab.html", output);
process.stdout.write(`Lab shell created: ${Buffer.byteLength(output)} bytes\n`);
