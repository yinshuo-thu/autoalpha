async (page) => {
  const fs = require("fs/promises");
  const path = require("path");

  const BASE_URL = "https://labs.scientechresearch.io";
  const OUTPUT_DIR = path.join(process.cwd(), "scientech_challenge_2_markdown");
  const USERNAME = "syin26";
  const PASSWORD = "UU65nRvWCVHhpQCJ";

  const pages = [
    {
      slug: "submission",
      path: "/web/challenges/challenge-page/2/submission",
      contentSelector: ".submission-guidelines.trix-container",
    },
    {
      slug: "overview",
      path: "/web/challenges/challenge-page/2/overview",
      contentSelector: ".challenge-description.trix-container",
    },
    {
      slug: "evaluation",
      path: "/web/challenges/challenge-page/2/evaluation",
      contentSelector: ".challenge-description.trix-container",
    },
    {
      slug: "phases",
      path: "/web/challenges/challenge-page/2/phases",
      contentSelector: ".challenge-description.trix-container",
    },
  ];

  async function ensureLoggedIn(targetUrl) {
    await page.goto(targetUrl, { waitUntil: "domcontentloaded", timeout: 120000 });
    if (!page.url().includes("/auth/login")) {
      await page.waitForLoadState("networkidle").catch(() => {});
      return;
    }

    await page.getByRole("textbox", { name: /username/i }).fill(USERNAME);
    await page.getByRole("textbox", { name: /password/i }).fill(PASSWORD);
    await page.getByRole("button", { name: /log in/i }).click();
    await page.waitForTimeout(2000);
    await page.goto(targetUrl, { waitUntil: "domcontentloaded", timeout: 120000 });
    await page.waitForLoadState("networkidle").catch(() => {});
  }

  async function extractPage(config) {
    const url = `${BASE_URL}${config.path}`;
    await ensureLoggedIn(url);

    const data = await page.evaluate(({ selector, pageUrl }) => {
      function normalizeText(text) {
        return (text || "").replace(/\u00a0/g, " ").replace(/\r/g, "").trim();
      }

      function escapePipes(text) {
        return normalizeText(text).replace(/\|/g, "\\|");
      }

      function stripPilcrow(text) {
        return normalizeText(text).replace(/\s*¶\s*$/g, "");
      }

      function inline(node) {
        if (node.nodeType === Node.TEXT_NODE) {
          return node.textContent || "";
        }
        if (node.nodeType !== Node.ELEMENT_NODE) {
          return "";
        }

        const tag = node.tagName.toLowerCase();
        if (tag === "br") {
          return "  \n";
        }

        const content = Array.from(node.childNodes).map(inline).join("");
        const text = normalizeText(content);

        if (tag === "strong" || tag === "b") {
          return text ? `**${text}**` : "";
        }
        if (tag === "em" || tag === "i") {
          return text ? `*${text}*` : "";
        }
        if (tag === "code") {
          return text ? `\`${text}\`` : "";
        }
        if (tag === "a") {
          const href = node.getAttribute("href") || "";
          const absolute = href.startsWith("http") ? href : new URL(href, pageUrl).toString();
          const label = text || absolute;
          return `[${label}](${absolute})`;
        }
        return content;
      }

      function convertTable(table) {
        const rows = Array.from(table.querySelectorAll("tr")).map((row) =>
          Array.from(row.children).map((cell) => escapePipes(cell.innerText))
        );
        if (!rows.length) {
          return "";
        }

        const header = rows[0];
        const body = rows.slice(1);
        return [
          `| ${header.join(" | ")} |`,
          `| ${header.map(() => "---").join(" | ")} |`,
          ...body.map((row) => `| ${row.join(" | ")} |`),
        ].join("\n");
      }

      function convertList(list, depth = 0) {
        const ordered = list.tagName.toLowerCase() === "ol";
        const items = Array.from(list.children).filter((child) => child.tagName?.toLowerCase() === "li");
        return items
          .map((item, index) => {
            const prefix = `${"  ".repeat(depth)}${ordered ? `${index + 1}.` : "-"}`;
            const parts = [];
            const nested = [];

            Array.from(item.childNodes).forEach((child) => {
              if (child.nodeType === Node.ELEMENT_NODE && ["ul", "ol"].includes(child.tagName.toLowerCase())) {
                nested.push(convertList(child, depth + 1));
                return;
              }
              const piece = inline(child);
              if (normalizeText(piece)) {
                parts.push(piece.trim());
              }
            });

            return [`${prefix} ${normalizeText(parts.join(" "))}`, ...nested.filter(Boolean)]
              .filter(Boolean)
              .join("\n");
          })
          .join("\n");
      }

      function block(node) {
        if (node.nodeType === Node.TEXT_NODE) {
          return normalizeText(node.textContent);
        }
        if (node.nodeType !== Node.ELEMENT_NODE) {
          return "";
        }

        const tag = node.tagName.toLowerCase();
        if (["script", "style", "button", "input", "textarea", "label", "svg", "img"].includes(tag)) {
          return "";
        }

        if (/^h[1-6]$/.test(tag)) {
          const level = Number(tag.slice(1));
          const headingText = stripPilcrow(node.textContent || "");
          return headingText ? `${"#".repeat(level)} ${headingText}` : "";
        }
        if (tag === "p") {
          return normalizeText(Array.from(node.childNodes).map(inline).join(""));
        }
        if (tag === "pre") {
          return `\`\`\`\n${node.innerText.trim()}\n\`\`\``;
        }
        if (tag === "blockquote") {
          return Array.from(node.childNodes)
            .map(block)
            .filter(Boolean)
            .join("\n\n")
            .split("\n")
            .map((line) => `> ${line}`)
            .join("\n");
        }
        if (tag === "hr") {
          return "---";
        }
        if (tag === "table") {
          return convertTable(node);
        }
        if (tag === "ul" || tag === "ol") {
          return convertList(node);
        }

        const childBlocks = Array.from(node.childNodes).map(block).filter(Boolean);
        if (childBlocks.length) {
          return childBlocks.join("\n\n");
        }
        return normalizeText(Array.from(node.childNodes).map(inline).join(""));
      }

      function extractChallengeMeta() {
        const titleContainer = document.querySelector(".challenge-title-container");
        if (!titleContainer) {
          return null;
        }

        const title = normalizeText(titleContainer.querySelector("h4")?.innerText || "");
        const fields = Array.from(titleContainer.querySelectorAll(".card-content > div"))
          .map((row) => normalizeText(row.innerText))
          .filter(Boolean);
        const tags = Array.from(titleContainer.querySelectorAll("li"))
          .map((li) => normalizeText(li.innerText))
          .filter(Boolean);
        return { title, fields, tags };
      }

      function extractSubmissionPhaseInfo() {
        const container = document.querySelector(".phase-container");
        if (!container) {
          return null;
        }

        const phases = Array.from(container.querySelectorAll("li"))
          .map((item) => normalizeText(item.innerText))
          .filter((text) => text.startsWith("Phase:"));

        return {
          heading: normalizeText(container.querySelector("h5")?.innerText || ""),
          phases,
        };
      }

      const root =
        document.querySelector(selector) ||
        Array.from(document.querySelectorAll(".trix-container")).sort((a, b) => b.innerText.length - a.innerText.length)[0];

      if (!root) {
        throw new Error(`Could not find content root for selector: ${selector}`);
      }

      return {
        pageTitle: document.title,
        challenge: extractChallengeMeta(),
        submissionPhaseInfo: extractSubmissionPhaseInfo(),
        markdown: Array.from(root.childNodes).map(block).filter(Boolean).join("\n\n"),
      };
    }, { selector: config.contentSelector, pageUrl: url });

    return { url, ...data };
  }

  function buildDocument(config, data) {
    const lines = [];
    const title = data.challenge?.title || "Scientech Challenge Page";

    lines.push(`# ${title} - ${config.slug}`);
    lines.push("");
    lines.push(`Source: ${data.url}`);
    lines.push(`Captured: ${new Date().toISOString()}`);
    lines.push("");

    if (data.challenge) {
      lines.push("## Challenge Metadata");
      lines.push("");
      data.challenge.fields.forEach((field) => lines.push(`- ${field}`));
      if (data.challenge.tags.length) {
        lines.push(`- Tags: ${data.challenge.tags.join(", ")}`);
      }
      lines.push("");
    }

    if (config.slug === "submission" && data.submissionPhaseInfo?.phases?.length) {
      lines.push("## Submission Portal");
      lines.push("");
      if (data.submissionPhaseInfo.heading) {
        lines.push(`### ${data.submissionPhaseInfo.heading}`);
        lines.push("");
      }
      data.submissionPhaseInfo.phases.forEach((phase) => lines.push(`- ${phase}`));
      lines.push("");
    }

    lines.push("## Page Content");
    lines.push("");
    lines.push(data.markdown);
    lines.push("");

    return lines.join("\n").replace(/\n{3,}/g, "\n\n");
  }

  await fs.mkdir(OUTPUT_DIR, { recursive: true });

  for (const config of pages) {
    const data = await extractPage(config);
    const filePath = path.join(OUTPUT_DIR, `${config.slug}.md`);
    await fs.writeFile(filePath, buildDocument(config, data), "utf8");
    console.log(`Saved ${filePath}`);
  }
}
