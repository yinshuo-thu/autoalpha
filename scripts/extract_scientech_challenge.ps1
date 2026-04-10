$ErrorActionPreference = "Stop"

$session = "scientech_export_" + ([guid]::NewGuid().ToString("N").Substring(0, 8))
$outputDir = Join-Path (Get-Location) "scientech_challenge_2_markdown"
$baseUrl = "https://labs.scientechresearch.io"
$username = "syin26"
$password = "UU65nRvWCVHhpQCJ"

$pages = @(
    @{
        Slug = "submission"
        Path = "/web/challenges/challenge-page/2/submission"
        Selector = ".submission-guidelines.trix-container"
    },
    @{
        Slug = "overview"
        Path = "/web/challenges/challenge-page/2/overview"
        Selector = ".challenge-description.trix-container"
    },
    @{
        Slug = "evaluation"
        Path = "/web/challenges/challenge-page/2/evaluation"
        Selector = ".challenge-description.trix-container"
    },
    @{
        Slug = "phases"
        Path = "/web/challenges/challenge-page/2/phases"
        Selector = ".challenge-description.trix-container"
    }
)

$loginCode = @'
async page => {
  await page.getByRole('textbox', { name: /username/i }).fill('syin26');
  await page.getByRole('textbox', { name: /password/i }).fill('UU65nRvWCVHhpQCJ');
  await page.getByRole('button', { name: /log in/i }).click();
  await page.waitForTimeout(2000);
}
'@

$extractCode = @'
() => {
  const selector = '__SELECTOR__';
  const pageUrl = '__PAGEURL__';
  const TICK = String.fromCharCode(96);

  function normalizeText(text) {
    return (text || '').replace(/\u00a0/g, ' ').replace(/\r/g, '').trim();
  }

  function escapePipes(text) {
    return normalizeText(text).replace(/\|/g, '\|');
  }

  function stripPilcrow(text) {
    return normalizeText(text).replace(/\s*¶\s*$/g, "");
  }

  function inline(node) {
    if (node.nodeType === Node.TEXT_NODE) {
      return node.textContent || '';
    }
    if (node.nodeType !== Node.ELEMENT_NODE) {
      return '';
    }

    const tag = node.tagName.toLowerCase();
    if (tag === 'br') {
      return '  \n';
    }

    const content = Array.from(node.childNodes).map(inline).join('');
    const text = normalizeText(content);

    if (tag === "strong" || tag === "b") {
      return text ? "**" + text + "**" : "";
    }
    if (tag === "em" || tag === "i") {
      return text ? "*" + text + "*" : "";
    }
    if (tag === "code") {
      return text ? TICK + text + TICK : "";
    }
    if (tag === "a") {
      const href = node.getAttribute("href") || "";
      const absolute = href ? new URL(href, pageUrl).toString() : pageUrl;
      const label = text || absolute;
      return "[" + label + "](" + absolute + ")";
    }
    return content;
  }

  function convertTable(table) {
    const rows = Array.from(table.querySelectorAll("tr")).map(row =>
      Array.from(row.children).map(cell => escapePipes(cell.innerText))
    );
    if (!rows.length) {
      return "";
    }

    const header = rows[0];
    const body = rows.slice(1);
    return [
      "| " + header.join(" | ") + " |",
      "| " + header.map(() => "---").join(" | ") + " |",
      ...body.map(row => "| " + row.join(" | ") + " |")
    ].join("\n");
  }

  function convertList(list, depth = 0) {
    const ordered = list.tagName.toLowerCase() === "ol";
    const items = Array.from(list.children).filter(child => child.tagName && child.tagName.toLowerCase() === "li");
    return items.map((item, index) => {
      const prefix = "  ".repeat(depth) + (ordered ? String(index + 1) + "." : "-");
      const parts = [];
      const nested = [];

      Array.from(item.childNodes).forEach(child => {
        if (child.nodeType === Node.ELEMENT_NODE && ["ul", "ol"].includes(child.tagName.toLowerCase())) {
          nested.push(convertList(child, depth + 1));
          return;
        }

        const piece = inline(child);
        if (normalizeText(piece)) {
          parts.push(piece.trim());
        }
      });

      return [prefix + " " + normalizeText(parts.join(" ")), ...nested.filter(Boolean)]
        .filter(Boolean)
        .join("\n");
    }).join("\n");
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
      return headingText ? "#".repeat(level) + " " + headingText : "";
    }
    if (tag === "p") {
      return normalizeText(Array.from(node.childNodes).map(inline).join(""));
    }
    if (tag === "pre") {
      return TICK.repeat(3) + "\n" + node.innerText.trim() + "\n" + TICK.repeat(3);
    }
    if (tag === "blockquote") {
      return Array.from(node.childNodes)
        .map(block)
        .filter(Boolean)
        .join("\n\n")
        .split("\n")
        .map(line => "> " + line)
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
      .map(row => normalizeText(row.innerText))
      .filter(Boolean);
    const tags = Array.from(titleContainer.querySelectorAll("li"))
      .map(li => normalizeText(li.innerText))
      .filter(Boolean);
    return { title, fields, tags };
  }

  function extractSubmissionPhaseInfo() {
    const container = document.querySelector(".phase-container");
    if (!container) {
      return null;
    }

    const phases = Array.from(container.querySelectorAll("li"))
      .map(item => normalizeText(item.innerText))
      .filter(text => text.startsWith("Phase:"));

    return {
      heading: normalizeText(container.querySelector("h5")?.innerText || ""),
      phases
    };
  }

  const root =
    document.querySelector(selector) ||
    Array.from(document.querySelectorAll(".trix-container")).sort((a, b) => b.innerText.length - a.innerText.length)[0];

  if (!root) {
    throw new Error("Could not find content root for selector: " + selector);
  }

  return {
    pageTitle: document.title,
    challenge: extractChallengeMeta(),
    submissionPhaseInfo: extractSubmissionPhaseInfo(),
    markdown: Array.from(root.childNodes).map(block).filter(Boolean).join("\n\n")
  };
}
'@

function Invoke-PlaywrightText {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    & npx --yes --package @playwright/cli playwright-cli @Args
}

function Invoke-PlaywrightRaw {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Args
    )

    & npx --yes --package @playwright/cli playwright-cli --raw @Args
}

function Ensure-LoggedIn {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Url
    )

    Invoke-PlaywrightText @("-s=$session", "goto", $Url) | Out-Null
    $currentUrl = Invoke-PlaywrightRaw @("-s=$session", "eval", "() => location.href")
    $currentUrl = $currentUrl.Trim('"')

    if ($currentUrl -like "*auth/login*") {
        Invoke-PlaywrightText @("-s=$session", "run-code", (Flatten-Js $loginCode)) | Out-Null
        Invoke-PlaywrightText @("-s=$session", "goto", $Url) | Out-Null
    }
}

function Build-Markdown {
    param(
        [Parameter(Mandatory = $true)]
        [hashtable]$Page,
        [Parameter(Mandatory = $true)]
        [pscustomobject]$Data,
        [Parameter(Mandatory = $true)]
        [string]$SourceUrl
    )

    $lines = New-Object System.Collections.Generic.List[string]
    $title = if ($Data.challenge -and $Data.challenge.title) { $Data.challenge.title } else { "Scientech Challenge Page" }
    $captured = [DateTimeOffset]::Now.ToString("o")

    $lines.Add("# $title - $($Page.Slug)")
    $lines.Add("")
    $lines.Add("Source: $SourceUrl")
    $lines.Add("Captured: $captured")
    $lines.Add("")

    if ($Data.challenge) {
        $lines.Add("## Challenge Metadata")
        $lines.Add("")
        foreach ($field in $Data.challenge.fields) {
            $lines.Add("- $field")
        }
        if ($Data.challenge.tags.Count -gt 0) {
            $lines.Add("- Tags: $($Data.challenge.tags -join ', ')")
        }
        $lines.Add("")
    }

    if ($Page.Slug -eq "submission" -and $Data.submissionPhaseInfo -and $Data.submissionPhaseInfo.phases.Count -gt 0) {
        $lines.Add("## Submission Portal")
        $lines.Add("")
        if ($Data.submissionPhaseInfo.heading) {
            $lines.Add("### $($Data.submissionPhaseInfo.heading)")
            $lines.Add("")
        }
        foreach ($phase in $Data.submissionPhaseInfo.phases) {
            $lines.Add("- $phase")
        }
        $lines.Add("")
    }

    $lines.Add("## Page Content")
    $lines.Add("")
    $lines.Add($Data.markdown)
    $lines.Add("")

    return (($lines -join "`n") -replace "(`n){3,}", "`n`n")
}

function Flatten-Js {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Code
    )

    return (($Code -replace "(`r`n|`n|`r)+", " ") -replace "\s{2,}", " ").Trim()
}

New-Item -ItemType Directory -Force -Path $outputDir | Out-Null

$firstUrl = $baseUrl + $pages[0].Path
Invoke-PlaywrightText @("-s=$session", "open", $firstUrl, "--browser", "msedge", "--headed") | Out-Null

try {
    foreach ($page in $pages) {
        $url = $baseUrl + $page.Path
        Ensure-LoggedIn -Url $url

        $pageExtractCode = $extractCode.Replace("__SELECTOR__", $page.Selector.Replace("\", "\\")).Replace("__PAGEURL__", $url.Replace("\", "\\"))
        $pageExtractCode = Flatten-Js $pageExtractCode
        $rawJson = Invoke-PlaywrightRaw @("-s=$session", "eval", $pageExtractCode)
        $data = $rawJson | ConvertFrom-Json

        $markdown = Build-Markdown -Page $page -Data $data -SourceUrl $url
        $outputFile = Join-Path $outputDir ($page.Slug + ".md")
        Set-Content -Path $outputFile -Value $markdown -Encoding UTF8
        Write-Output "Saved $outputFile"
    }
}
finally {
    Invoke-PlaywrightText @("-s=$session", "close") | Out-Null
}
