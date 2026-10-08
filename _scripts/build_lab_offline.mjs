// Produce a self-contained copy with the same UI, numerical code, and model.
import fs from "node:fs";
import * as sass from "sass";
const read = (path) => fs.readFileSync(path, "utf8");
const site = "https://hyeongyu-kim.github.io/";
let content = read("_pages/lab.md").replace(/^---[\s\S]*?---\s*/, "");
content = content.replace(/\{\{\s*'([^']+)'\s*\|\s*(?:relative_url|absolute_url)\s*\}\}/g, (_, path) => new URL(path, site).href);
if (content.includes("{{") || content.includes("{%")) throw new Error("Unresolved template in the offline lab");
const model = JSON.stringify(JSON.parse(read("assets/lab/digits-model.json"))).replace(/</g, "\\u003c");
const math = read("assets/js/lab-math.js").replace(/^export /gm, "");
let ui = read("assets/js/research-lab.js").replace(/^import\s*\{[\s\S]*?\}\s*from\s*"\.\/lab-math.js";\s*/, "");
ui = ui.replace(/const response = await fetch[\s\S]*?const model = await response\.json\(\);/, 'const model = JSON.parse(document.getElementById("embedded-digit-model").textContent);');
if (ui.includes("await fetch") || ui.includes("import ")) throw new Error("Offline lab still has a network dependency");
const css = sass.compileString('@use "editorial"; @use "research-lab";', {loadPaths:["_sass"],style:"compressed",logger:sass.Logger.silent}).css;
const script = math + "\n" + ui;
new Function(script);
const output = `<!doctype html><html lang="en" data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Interactive lab · Hyeongyu Kim</title><style>${css}\nbody{margin:0;color:var(--global-text-color);background:var(--global-bg-color)}.offline-wrap{max-width:980px;padding:32px 24px;margin:auto}.offline-heading{display:flex;justify-content:space-between;align-items:center;gap:1rem;margin-bottom:3rem}.offline-heading>a{font-family:Georgia,serif;color:var(--editorial-title-color);font-size:1.2rem}</style></head><body><div class="offline-wrap"><header class="offline-heading"><a href="${site}">Hyeongyu Kim</a><button class="quiet-button" id="offline-theme" type="button">Dark mode</button></header><main class="editorial-content"><header class="editorial-page-heading"><p class="editorial-eyebrow">Research, in small experiments</p><h1>Interactive lab</h1><p>Both experiments run locally. No connection is needed after saving this file.</p></header>${content}</main></div><script type="application/json" id="embedded-digit-model">${model}</script><script>${script}\ndocument.getElementById("offline-theme").addEventListener("click",function(){const theme=document.documentElement.getAttribute("data-theme")==="dark"?"light":"dark";document.documentElement.setAttribute("data-theme",theme);this.textContent=theme==="dark"?"Light mode":"Dark mode";});</script></body></html>`;
fs.writeFileSync("assets/lab/research-lab.html", output);
process.stdout.write(`Offline lab created: ${Buffer.byteLength(output)} bytes\n`);
