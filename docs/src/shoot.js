// Screenshots of the demo page for the manual and guide.
//   node shoot.js lite            (no model server running)
//   node shoot.js full            (with `python app/server.py` running)
const puppeteer = require("puppeteer-core");
const path = require("path");
const CHROME = process.env.CHROME || "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PAGE = "file://" + path.resolve(__dirname, "../../Marathi-news-intelligence-pipeline/Marathi-NLP-Demo.html");
const OUT = path.join(__dirname, "shots");
const mode = process.argv[2] || "lite";

const TITLE = "भारताचा इंग्लंडवर शानदार विजय";
const TEXT = "भारतीय क्रिकेट संघाने दुसऱ्या कसोटी सामन्यात इंग्लंडचा नऊ गडी राखून पराभव केला. " +
  "कर्णधार रोहित शर्माने मुंबईत पत्रकार परिषदेत संघाचे कौतुक केले. " +
  "मात्र दुखापतीमुळे जसप्रीत बुमराह पुढील सामन्याला मुकणार आहे. " +
  "हा सामना वानखेडे स्टेडियमवर खेळवण्यात आला होता. " +
  "प्रेक्षकांनी मोठ्या संख्येने उपस्थित राहून संघाला पाठिंबा दिला.";
const SENTENCE = "मुंबईतील शाळांमध्ये विद्यार्थ्यांसाठी नवीन अभ्यासक्रम सुरू झाला. सरकारने याबाबत घोषणा केली.";
const sleep = ms => new Promise(r => setTimeout(r, ms));

(async () => {
  require("fs").mkdirSync(OUT, { recursive: true });
  const browser = await puppeteer.launch({ executablePath: CHROME, headless: "new",
                                           args: ["--allow-file-access-from-files"] });
  const page = await browser.newPage();
  await page.setViewport({ width: 1000, height: 820, deviceScaleFactor: 2 });
  const errors = [];
  page.on("pageerror", e => errors.push(String(e)));
  await page.goto(PAGE, { waitUntil: "networkidle0" });
  await sleep(1500);

  const shot = async (name, sel, pad = 0) => {
    if (!sel) return page.screenshot({ path: `${OUT}/${name}.png` });
    const el = await page.$(sel);
    await el.evaluate(e => e.scrollIntoView({ block: "start" }));
    await sleep(200);
    const b = await el.boundingBox();
    await page.screenshot({ path: `${OUT}/${name}.png`, captureBeyondViewport: true,
      clip: { x: Math.max(b.x - pad, 0), y: b.y - pad + (await page.evaluate(() => window.scrollY)),
              width: b.width + 2 * pad, height: b.height + 2 * pad } });
  };
  const setVal = (sel, v) => page.$eval(sel, (e, v) => { e.value = v; e.dispatchEvent(new Event("input")); }, v);
  const clickTab = async i => { await page.click(`#t${i}`); await sleep(300); };
  const analyse = async () => {
    await clickTab(2);
    await page.click("#clear");
    await setVal("#ptitle", TITLE);
    await setVal("#ptext", TEXT);
    await page.click("#run");
    await page.waitForFunction(() => !document.querySelector("#run").disabled &&
      document.querySelector("#artview").children.length > 0, { timeout: 300000 });
    await sleep(500);
  };
  const tagStages = p => page.evaluate(p => document.querySelectorAll("#artview .stage")
    .forEach((s, i) => { s.id = p + i; }), p);

  if (mode === "lite") {
    await page.evaluate(() => window.scrollTo(0, 0));
    await shot("01_open");
    await setVal("#mword", "विद्यार्थ्यांसाठी");
    await shot("04_morph_own_word", "#p0 .card", 4);
    await setVal("#mword", SENTENCE);
    await shot("05_morph_text", "#p0 .card", 4);
    await setVal("#mword", "घरांमधून");
    await shot("06_families", "#fams", 4);

    await clickTab(1);
    await shot("07_search_court", "#p1 .card", 4);
    await page.evaluate(() => [...document.querySelectorAll("#qchips .chip")].find(c => c.dataset.q === "पाऊस").click());
    await sleep(200);
    await shot("08_search_rain", "#p1 .card", 4);
    await page.evaluate(() => { document.querySelector("details.add").open = true; });
    await setVal("#atitle", "पुण्यात नवीन मेट्रो मार्ग");
    await setVal("#atext", "पुणे महानगरपालिकेने शहरात नवीन मेट्रो मार्गाची घोषणा केली. या मेट्रोमुळे पुणेकरांचा प्रवास सोपा होणार आहे.");
    await shot("09_add_article_form", "details.add", 4);
    await page.click("#aadd");
    await setVal("#q", "मेट्रो");
    await sleep(200);
    await shot("10_add_article_found", "#p1 .card", 4);

    await analyse();
    await shot("12_pipeline_input", "#p2 .card", 4);
    await shot("13_pipeline_results", "#artview", 4);
    await tagStages("st");
    await shot("15_stage_category", "#st1", 2);
    await shot("17_stage_entities", "#st3", 2);
    await shot("18_stage_sentiment", "#st4", 2);
    await page.$eval("#p2 .card .actions", e => { e.style.width = "max-content"; });
    await shot("20_mode_badge", "#p2 .card .actions", 4);
  } else {
    await analyse();
    await page.waitForFunction(() => document.querySelector("#mode").textContent.includes("Full"));
    await page.$eval("#p2 .card .actions", e => { e.style.width = "max-content"; });
    await shot("21_full_badge", "#p2 .card .actions", 4);
    await tagStages("fs");
    await shot("23_full_entities", "#fs3", 2);
    await page.click("#trbtn");
    await page.waitForFunction(() => { const st = [...document.querySelectorAll("#artview .stage")].pop();
      return st && st.querySelector(".en"); }, { timeout: 300000 });
    await tagStages("fs");
    await shot("24_translation", "#fs5", 2);
  }
  console.log(mode, "done; page errors:", errors.length ? errors : "none");
  await browser.close();
})();
