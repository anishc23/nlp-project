# Sources for the two PDFs in `docs/`

`manual.html` and `guide.html` are printed to `../Demo-User-Manual.pdf` and
`../Project-Explanation-Guide.pdf`. The screenshots in `shots/` are taken from
the real demo page.

```bash
cd docs/src
npm install                       # puppeteer-core; uses your installed Chrome
node shoot.js lite                # screenshots with the in-browser models
python ../../Marathi-news-intelligence-pipeline/app/server.py &   # then:
node shoot.js full                # screenshots with the full models
node render.js                    # writes both PDFs
```

The source-code listing (`../Source-Code.pdf`) is built from the project's own
files, so it stays in step with what is committed:

```bash
python build_code_doc.py          # writes code.html
node render_code.js               # writes ../Source-Code.pdf
```

Set `CHROME=/path/to/chrome` if Chrome is not in the default macOS location.
