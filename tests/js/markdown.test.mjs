// Regression tests for the viewer's markdown renderer.
//
// Run with `node --test tests/js`. No dependencies and no browser: the
// parser is pure, and the DOM builder is exercised against the tiny stub
// below, which implements only the handful of methods the renderer uses.
// If the renderer ever reaches for innerHTML or any other escape hatch,
// the stub has no such property and the test throws.

import assert from "node:assert/strict";
import { test } from "node:test";

import {
  parseInline,
  parseMarkdown,
  renderMarkdown,
  safeHref,
} from "../../src/keycall/viewer/static/markdown.js";

const BASE = "http://127.0.0.1:8823/";

// --- a DOM small enough to audit ------------------------------------------

class StubNode {
  constructor(tag) {
    this.tag = tag;
    this.children = [];
    this.attrs = {};
    this._text = "";
  }
  set textContent(value) {
    this._text = String(value);
    this.children = [];
  }
  get textContent() {
    return this.children.length
      ? this.children.map((c) => c.textContent).join("")
      : this._text;
  }
  set href(v) { this.attrs.href = v; }
  get href() { return this.attrs.href; }
  set target(v) { this.attrs.target = v; }
  set rel(v) { this.attrs.rel = v; }
  set className(v) { this.attrs.class = v; }
  appendChild(node) {
    if (node.tag === "#fragment") this.children.push(...node.children);
    else this.children.push(node);
    return node;
  }
  get childNodes() { return this.children; }
  /** Every element of a given tag, at any depth. */
  find(tag) {
    const hits = [];
    for (const child of this.children) {
      if (child.tag === tag) hits.push(child);
      if (child.find) hits.push(...child.find(tag));
    }
    return hits;
  }
  /** Every tag name produced anywhere in the tree. */
  tags() {
    const seen = [];
    for (const child of this.children) {
      if (child.tag && child.tag !== "#text") seen.push(child.tag);
      if (child.tags) seen.push(...child.tags());
    }
    return seen;
  }
}

class StubText extends StubNode {
  constructor(text) {
    super("#text");
    this._text = String(text);
  }
}

const doc = {
  createElement: (tag) => new StubNode(tag),
  createTextNode: (text) => new StubText(text),
  createDocumentFragment: () => new StubNode("#fragment"),
};

const render = (md) => renderMarkdown(md, doc, BASE);

// --- security --------------------------------------------------------------

test("javascript: and data: links never become anchors", () => {
  for (const href of [
    "javascript:alert(1)",
    "JavaScript:alert(1)",
    "  javascript:alert(1)",
    "data:text/html,<h1>x</h1>",
    "vbscript:msgbox(1)",
    "file:///etc/passwd",
  ]) {
    assert.equal(safeHref(href, BASE), null, `${href} must be refused`);
    const root = render(`[click me](${href})`);
    assert.equal(root.find("a").length, 0, `${href} produced an anchor`);
    // The words survive even though the link does not.
    assert.match(root.textContent, /click me/);
  }
});

test("http and https links are kept, and carry noopener", () => {
  const root = render("see [Example](https://example.com/x?a=1) now");
  const [a] = root.find("a");
  assert.ok(a, "expected an anchor");
  assert.equal(a.href, "https://example.com/x?a=1");
  assert.equal(a.attrs.rel, "noopener noreferrer");
  assert.equal(a.attrs.target, "_blank");
  assert.equal(a.textContent, "Example");
});

test("HTML in a reply stays text and never becomes an element", () => {
  const hostile = [
    "<img src=x onerror=alert(1)>",
    "<script>alert(2)</script>",
    "<iframe src=//evil></iframe>",
    "<svg onload=alert(3)>",
    "<a href='javascript:alert(4)'>x</a>",
  ].join("\n\n");
  const root = render(hostile);
  const produced = new Set(root.tags());
  for (const forbidden of ["img", "script", "iframe", "svg", "a", "object", "embed"]) {
    assert.ok(!produced.has(forbidden), `renderer produced a <${forbidden}>`);
  }
  // The markup is preserved as words, so nothing is silently swallowed.
  assert.match(root.textContent, /<script>alert\(2\)<\/script>/);
});

test("only a known vocabulary of tags can ever be produced", () => {
  const root = render(
    "# h\n\ntext **b** *i* `c` [l](https://e.com)\n\n- a\n\n1. b\n\n```\nx\n```"
  );
  const allowed = new Set(["h3", "h4", "h5", "h6", "p", "ul", "ol", "li", "pre", "code", "strong", "em", "a"]);
  for (const tag of root.tags()) {
    assert.ok(allowed.has(tag), `unexpected tag <${tag}>`);
  }
});

test("heading level is clamped, so '#######' cannot escape the range", () => {
  for (const [md, tag] of [["# a", "h3"], ["### a", "h5"], ["###### a", "h6"]]) {
    assert.equal(render(md).children[0].tag, tag);
  }
  // Seven hashes is not a heading at all, and must not become <h9>.
  const root = render("####### a");
  assert.equal(root.children[0].tag, "p");
});

// --- robustness ------------------------------------------------------------

test("a reply full of inline spans does not exhaust the stack", () => {
  // The previous recursive scanner threw RangeError near 20k spans, which
  // killed the turn and left the Send button disabled.
  for (const count of [1000, 20000, 50000]) {
    const hostile = "**a**".repeat(count);
    const root = render(hostile);
    assert.equal(root.find("strong").length, count);
  }
});

test("spans nest to the depth the patterns permit", () => {
  // Each emphasis pattern excludes its own marker from the content it
  // matches, so `*` can hold `_` but never another `*`. Genuine nesting
  // therefore tops out around three levels, which is what this pins. The
  // MAX_INLINE_DEPTH guard sits above that and is unreachable today; it is
  // there so a future pattern that does allow same-marker nesting cannot
  // reintroduce unbounded recursion silently.
  const root = render("**a _b `c` d_ e**");
  const [strong] = root.find("strong");
  assert.ok(strong, "outer strong missing");
  const [em] = strong.find("em");
  assert.ok(em, "nested em missing");
  assert.equal(em.find("code")[0].textContent, "c");
  assert.equal(root.textContent, "a b c d e");
});

test("an unclosed code fence still renders instead of throwing", () => {
  const root = render("before\n\n```\nnever closed");
  assert.equal(root.find("pre").length, 1);
  assert.match(root.textContent, /never closed/);
});

test("empty and non-string input are handled", () => {
  assert.deepEqual(parseMarkdown(""), []);
  assert.deepEqual(parseMarkdown(null), []);
  assert.equal(render("").textContent, "");
});

// --- formatting ------------------------------------------------------------

test("emphasis, code, lists and fences parse as expected", () => {
  const root = render(
    ["## Title", "", "a **bold** and *italic* and `code` here", "",
     "- one", "- two", "", "1. first", "2. second", "",
     "```", "const x = 1;", "```"].join("\n")
  );
  assert.deepEqual(
    root.children.map((c) => c.tag),
    ["h4", "p", "ul", "ol", "pre"]
  );
  assert.equal(root.find("strong")[0].textContent, "bold");
  assert.equal(root.find("em")[0].textContent, "italic");
  assert.equal(root.find("li").length, 4);
  assert.equal(root.find("pre")[0].textContent, "const x = 1;");
  assert.ok(!root.textContent.includes("**"), "raw markers left in output");
});

test("markers inside inline code stay literal", () => {
  const nodes = parseInline("use `**not bold**` here");
  const code = nodes.find((n) => n.type === "code");
  assert.equal(code.text, "**not bold**");
  assert.ok(!nodes.some((n) => n.type === "strong"));
});

test("a fenced block does not have its contents parsed", () => {
  const root = render("```\n**not bold** and [not a link](https://e.com)\n```");
  assert.equal(root.find("strong").length, 0);
  assert.equal(root.find("a").length, 0);
});
