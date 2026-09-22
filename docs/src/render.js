// Print manual.html and guide.html to ../*.pdf with page numbers.
const puppeteer = require("puppeteer-core");
const path = require("path");
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const jobs = [["manual.html", "../Demo-User-Manual.pdf", "Demo user manual"],
              ["guide.html", "../Project-Explanation-Guide.pdf", "Project explanation guide"]];
(async () => {
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new",
                                           args: ["--allow-file-access-from-files"] });
  for (const [src, out, label] of jobs) {
    const page = await browser.newPage();
    await page.goto("file://" + path.join(__dirname, src), { waitUntil: "networkidle0" });
    await page.evaluateHandle("document.fonts.ready");
    await page.pdf({ path: path.join(__dirname, out), format: "A4", printBackground: true,
      margin: { top: "18mm", bottom: "20mm", left: "17mm", right: "17mm" },
      displayHeaderFooter: true, headerTemplate: "<span></span>",
      footerTemplate: `<div style="font-family:Helvetica,Arial;font-size:7.5px;color:#8aa0a6;width:100%;padding:0 17mm;display:flex;justify-content:space-between"><span>Marathi News Intelligence Pipeline · ${label}</span><span><span class="pageNumber"></span> / <span class="totalPages"></span></span></div>` });
    console.log("wrote", path.resolve(__dirname, out));
    await page.close();
  }
  await browser.close();
})();
