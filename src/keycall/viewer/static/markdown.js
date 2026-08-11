// Markdown rendering for model replies.
//
// Models answer in markdown whether or not you ask them to, so the raw text
// reads as punctuation. This file turns it into DOM nodes.
//
// Two rules govern everything here, both because model output is untrusted
// input that arrives in a page holding a live credential token:
//
//   1. No HTML is ever parsed. There is no sanitiser, no blocklist, and no
//      innerHTML. Elements are constructed from a fixed vocabulary, so the
//      only question a reviewer has to answer is "which tags can this
//      create", not "which attacks did the filter miss".
//   2. The only attacker-controlled value that reaches an attribute is a
//      link's href, and it must survive an explicit scheme allowlist first.
//
// Parsing is split from DOM building so the parser can be tested without a
// browser: parseMarkdown() is pure and returns plain objects, and
// buildNode() is a short, auditable switch over that fixed vocabulary.

// A ceiling on nested spans. It is unreachable with the patterns below,
// because each emphasis marker excludes itself from the content it matches
// (`*` can contain `_` but never another `*`), which tops real nesting out
// around three. It stays as a guard: a future pattern that does permit
// same-marker nesting would otherwise reintroduce unbounded recursion with
// nothing to stop it. What actually protects against a hostile reply is
// the iterative scanner in parseInline, not this number.
const MAX_INLINE_DEPTH = 8;

const FENCE = /^\s*```/;
const HEADING = /^(#{1,6})\s+(.*)$/;
const BULLET = /^\s*[-*+]\s+(.*)$/;
const NUMBERED = /^\s*\d+[.)]\s+(.*)$/;

// Sticky patterns, matched at a position rather than searched for, so the
// scanner walks the string once. Order matters: code before emphasis so
// `**x**` inside backticks stays literal, and the two-character markers
// before the one-character ones.
const INLINE = [
  { re: /`([^`\n]+)`/y, make: (m) => ({ type: "code", text: m[1] }) },
  { re: /\*\*([^*\n]+)\*\*/y, make: (m) => ({ type: "strong", raw: m[1] }) },
  { re: /__([^_\n]+)__/y, make: (m) => ({ type: "strong", raw: m[1] }) },
  { re: /\*([^*\n]+)\*/y, make: (m) => ({ type: "em", raw: m[1] }) },
  { re: /_([^_\n]+)_/y, make: (m) => ({ type: "em", raw: m[1] }) },
  {
    re: /\[([^\]\n]*)\]\(([^)\s]*)\)/y,
    make: (m) => ({ type: "link", raw: m[1], href: m[2] }),
  },
];

/** Schemes a link may use. http and https only: `javascript:` executes,
 *  and `data:` can carry a whole HTML document. Anything else is shown as
 *  the text it was and never made clickable. */
const SAFE_SCHEMES = new Set(["http:", "https:"]);

/** The href to use, or null when the link must stay inert. `base` is
 *  passed in rather than read from location so this stays pure. */
export function safeHref(href, base) {
  if (!href) return null;
  let url;
  try {
    url = new URL(href, base);
  } catch {
    return null;
  }
  return SAFE_SCHEMES.has(url.protocol) ? url.href : null;
}

/** Inline spans of one line's worth of text, as plain objects.
 *
 *  The scan is iterative: an earlier version recursed on the remainder of
 *  the string after each match, which made stack depth grow with the number
 *  of spans, and roughly 20k of them threw RangeError and killed the turn.
 *  Recursion now happens only for genuinely nested spans, and is capped. */
export function parseInline(text, depth = 0) {
  const out = [];
  if (!text) return out;
  let i = 0;
  let plain = 0;

  const flush = (end) => {
    if (end > plain) out.push({ type: "text", text: text.slice(plain, end) });
  };

  while (i < text.length) {
    let matched = null;
    for (const { re, make } of INLINE) {
      re.lastIndex = i;
      const m = re.exec(text);
      if (m) {
        matched = { node: make(m), length: m[0].length };
        break;
      }
    }
    if (!matched) {
      i++;
      continue;
    }
    flush(i);
    const node = matched.node;
    if (Object.hasOwn(node, "raw")) {
      // Nested spans, unless that would go too deep, in which case the
      // inner text is kept verbatim rather than dropped.
      node.children =
        depth < MAX_INLINE_DEPTH
          ? parseInline(node.raw, depth + 1)
          : [{ type: "text", text: node.raw }];
      delete node.raw;
    }
    out.push(node);
    i += matched.length;
    plain = i;
  }
  flush(text.length);
  return out;
}

/** Block structure, as plain objects. Pure: no DOM, no globals. */
export function parseMarkdown(text) {
  const lines = String(text ?? "").split("\n");
  const blocks = [];
  let i = 0;

  while (i < lines.length) {
    const line = lines[i];

    if (FENCE.test(line)) {
      const body = [];
      i++;
      while (i < lines.length && !FENCE.test(lines[i])) body.push(lines[i++]);
      i++; // closing fence, or the end if the model never wrote one
      blocks.push({ type: "code_block", text: body.join("\n") });
      continue;
    }

    const heading = line.match(HEADING);
    if (heading) {
      blocks.push({
        type: "heading",
        // Offset so a model's "#" can't outrank the page's own headings,
        // and clamped so the tag name is always one we recognise.
        level: Math.min(6, heading[1].length + 2),
        children: parseInline(heading[2]),
      });
      i++;
      continue;
    }

    const bullet = BULLET.test(line);
    const numbered = !bullet && NUMBERED.test(line);
    if (bullet || numbered) {
      const items = [];
      while (i < lines.length) {
        const item = lines[i].match(numbered ? NUMBERED : BULLET);
        if (!item) break;
        items.push(parseInline(item[1]));
        i++;
      }
      blocks.push({ type: "list", ordered: numbered, items });
      continue;
    }

    if (!line.trim()) {
      i++;
      continue;
    }

    const para = [];
    while (
      i < lines.length &&
      lines[i].trim() &&
      !FENCE.test(lines[i]) &&
      !HEADING.test(lines[i]) &&
      !BULLET.test(lines[i]) &&
      !NUMBERED.test(lines[i])
    ) {
      para.push(lines[i++]);
    }
    blocks.push({ type: "paragraph", children: parseInline(para.join("\n")) });
  }
  return blocks;
}

// Every tag this renderer can produce. A reviewer checking for injection
// only has to be satisfied with this list and with safeHref above.
const INLINE_TAGS = { strong: "strong", em: "em", code: "code" };

function buildInline(nodes, doc, base) {
  const frag = doc.createDocumentFragment();
  for (const node of nodes) {
    if (node.type === "text") {
      frag.appendChild(doc.createTextNode(node.text));
    } else if (node.type === "code") {
      const el = doc.createElement("code");
      el.textContent = node.text;
      frag.appendChild(el);
    } else if (node.type === "link") {
      const href = safeHref(node.href, base);
      const label = buildInline(node.children, doc, base);
      if (href) {
        const a = doc.createElement("a");
        a.href = href;
        a.target = "_blank";
        a.rel = "noopener noreferrer";
        a.appendChild(label);
        frag.appendChild(a);
      } else {
        // Refused scheme: keep the words, drop the link.
        frag.appendChild(label);
      }
    } else if (INLINE_TAGS[node.type]) {
      const el = doc.createElement(INLINE_TAGS[node.type]);
      el.appendChild(buildInline(node.children, doc, base));
      frag.appendChild(el);
    }
    // Anything unrecognised is skipped rather than guessed at.
  }
  return frag;
}

function buildNode(block, doc, base) {
  if (block.type === "code_block") {
    const pre = doc.createElement("pre");
    const code = doc.createElement("code");
    code.textContent = block.text;
    pre.appendChild(code);
    return pre;
  }
  if (block.type === "heading") {
    const el = doc.createElement(`h${block.level}`);
    el.appendChild(buildInline(block.children, doc, base));
    return el;
  }
  if (block.type === "list") {
    const list = doc.createElement(block.ordered ? "ol" : "ul");
    for (const item of block.items) {
      const li = doc.createElement("li");
      li.appendChild(buildInline(item, doc, base));
      list.appendChild(li);
    }
    return list;
  }
  const p = doc.createElement("p");
  p.appendChild(buildInline(block.children, doc, base));
  return p;
}

/** The rendered reply, as a single element ready to append. */
export function renderMarkdown(text, doc = document, base = undefined) {
  const root = doc.createElement("div");
  root.className = "md";
  const blocks = parseMarkdown(text);
  for (const block of blocks) root.appendChild(buildNode(block, doc, base));
  if (!root.childNodes.length) root.appendChild(doc.createTextNode(String(text ?? "")));
  return root;
}
