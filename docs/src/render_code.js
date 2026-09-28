// Print code.html to ../Source-Code.pdf with page numbers.
const puppeteer = require("puppeteer-core");
const path = require("path");
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new",
                                           args: ["--allow-file-access-from-files"] });
  const page = await browser.newPage();
  await page.goto("file://" + path.join(__dirname, "code.html"), { waitUntil: "networkidle0" });
  await page.evaluateHandle("document.fonts.ready");
  const out = path.join(__dirname, "../Source-Code.pdf");
  await page.pdf({ path: out, format: "A4", printBackground: true,
    margin: { top: "16mm", bottom: "18mm", left: "14mm", right: "14mm" },
    displayHeaderFooter: true, headerTemplate: "<span></span>",
    footerTemplate: `<div style="font-family:Helvetica,Arial;font-size:7.5px;color:#8aa0a6;width:100%;padding:0 14mm;display:flex;justify-content:space-between"><span>Marathi News Intelligence Pipeline · Source code</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>` });
  console.log("wrote", path.resolve(out));
  await browser.close();
})();
