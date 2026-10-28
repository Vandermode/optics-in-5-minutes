# Optics in 5 Minutes · Interactive teaching course

Thirteen interactive lessons connect geometric optics, wave propagation, imaging, and optical design. Each lesson combines a 12–15-minute explanation and experiment, three concept checks, a standalone 25–30-minute Python lab, optional deeper calculations, and instructor notes with worked reasoning and a ten-point rubric.

The website opens in English on a first visit. Use **English / 中文** in the header or chapter sidebar to switch to Simplified Chinese. The choice is remembered, and switching keeps the current section and experiment settings, including controls changed before selecting Run. The homepage, preparation guide, and all thirteen lessons include Chinese prose, controls, chart labels, quiz feedback, and term explanations. The films and downloadable Python notebooks retain their original language and code.

The homepage starts with a concept map. Fifteen concept groups connect the thirteen chapters across four themes; selecting a concept highlights its relationships and opens the relevant lesson links. Solid arrows show conceptual foundations, and dashed connections show shared methods or applications. These are not episode completion requirements. Expand “Foundations & connections” to find focused refreshers and explanations of each relationship. Concept links support browser history and direct URLs, and language switching preserves the selection. The episode index remains available below the map.

Search above the map on the homepage or in any chapter to find concepts across the whole course. Either page language accepts English or Chinese names, aliases, episode-card keywords, and abbreviations such as PSF or FDTD. Results show both names and related episode links. Select a concept or press Enter to open its explanation; arrow keys move between results. Searches are preserved in `?q=…` URLs, through language changes, and when returning from an episode. Six matches appear initially, with the rest available through “Show all”.

Each episode card lists four or five key concepts beneath its introduction, in English or Chinese. All 63 keyword links open a definition with related concepts and episode destinations. The homepage and chapter readers share the same explanations; language switching keeps the selected term open.

Every lesson and notebook opens with a concrete optical problem, why it matters, and what studying the topic enables you to do. This motivation comes before the prerequisite checklist.

Each lesson page has a navigation sidebar for its own sections, with the current section highlighted as you read. Terms marked with a dotted underline open an explanation in the selected language with a concise definition, its role in that chapter, and a formula where helpful. The explanation can be closed with its close button or Escape. On narrow screens, the section list expands from “On this page” and explanations open in a modal side panel.

Every film now has a **Video logic** companion: 95 chronological argument steps across the twelve episodes and 41 expandable explanations of the reasoning the short films compress. Approximate time links cue the embedded film and also open its YouTube or local-file source. The guides are written in English and Chinese together. E04 includes a two-control spectrum/coherence experiment, distinguishing the interference scan from a pulse, the Fourier relation from a width estimate, and Gaussian FWHM from the lesson experiment’s standard-deviation convention. The reasoning and a numerical example remain readable without JavaScript.

E11 and E12 add a geometric-optics route: physical surface intersection and vector Snell refraction, followed by fixed-power lens bending and a common-plane spot merit. Every lesson now places its prediction and guided steps before the experiment controls, including E03’s custom page. The ray model, film data, and self-contained notebook engines share one documented specification; independent scalar-angle and ABCD checks test it. The new E11/E12 films and revised E04/E07 films use local previews until their publication metadata is supplied. After building their releases, run `python teaching/tools/stage_film_previews.py E04 E07 E11 E12` to copy the films and English WebVTT captions into the ignored preview directory.

The course is for learners with undergraduate calculus, linear algebra, and basic Python. Each lesson lists the specific prior skills it uses, where they enter the argument, and focused resources for review. Complex amplitudes and Fourier decomposition are useful from E01; probability, electromagnetic fields, and gradients are called out where needed. The optical methods are developed in the lessons. See the [prerequisites and learning resources](https://vandermode.github.io/optics-in-5-minutes-course/preparation/) and the matching preparation block at the start of each notebook.

All numerical data are synthetic and seeded. The notebooks use CPU NumPy, SciPy, and Matplotlib; no renderer imports, repository clone, GPU, private data, or credentials are required.

[Open the course homepage](https://vandermode.github.io/optics-in-5-minutes-course/) · [Public teaching repository](https://github.com/Vandermode/optics-in-5-minutes-course)

| Episode | Lesson | Standalone notebook |
| --- | --- | --- |
| E01 | [The angular spectrum method](https://vandermode.github.io/optics-in-5-minutes-course/e01-angular-spectrum/) | [Open Colab](https://colab.research.google.com/github/Vandermode/optics-in-5-minutes-course/blob/main/teaching/e01-angular-spectrum/e01_angular_spectrum.ipynb) |
| E02 | [Why a lens computes a Fourier transform](e02-lens-fourier/index.html) | [Download revised notebook](e02-lens-fourier/e02_lens_fourier.ipynb) |
| E03 | [The diffraction limit](https://vandermode.github.io/optics-in-5-minutes-course/e03-diffraction/) | [Open Colab](https://colab.research.google.com/github/Vandermode/optics-in-5-minutes-course/blob/main/teaching/e03-diffraction/e03_diffraction_limit.ipynb) |
| E04 | [Coherence and what a camera averages](e04-coherence/index.html) | [Download revised notebook](e04-coherence/e04_coherence.ipynb) |
| E05 | [Holography and the phase problem](e05-phase-problem/index.html) | [Download revised notebook](e05-phase-problem/e05_phase_problem.ipynb) |
| E06 | [Flat optics and wrapped phase](https://vandermode.github.io/optics-in-5-minutes-course/e06-flat-optics/) | [Open Colab](https://colab.research.google.com/github/Vandermode/optics-in-5-minutes-course/blob/main/teaching/e06-flat-optics/e06_flat_optics.ipynb) |
| E07 | [Gratings, Bloch waves, and band gaps](e07-periodic-structures/index.html) | [Download revised notebook](e07-periodic-structures/e07_periodic_structures.ipynb) |
| E08 | [Waveguides, modes, and resonators](https://vandermode.github.io/optics-in-5-minutes-course/e08-trapping-light/) | [Open Colab](https://colab.research.google.com/github/Vandermode/optics-in-5-minutes-course/blob/main/teaching/e08-trapping-light/e08_trapping_light.ipynb) |
| E09 | [Maxwell on a grid](e09-maxwell-grid/index.html) | [Download revised notebook](e09-maxwell-grid/e09_maxwell_grid.ipynb) |
| E10 | [Designing light with gradients](e10-adjoint-design/index.html) | [Download revised notebook](e10-adjoint-design/e10_adjoint_design.ipynb) |
| E11 | [Ray tracing through a real lens](e11-ray-tracing/index.html) | [Download notebook](e11-ray-tracing/e11_ray_tracing.ipynb) |
| E12 | [Lens design with a measurable objective](e12-lens-design/index.html) | [Download notebook](e12-lens-design/e12_lens_design.ipynb) |
| E13 | [Differentiable optics: from images to design updates](e13-differentiable-optics/index.html) | [Download notebook](e13-differentiable-optics/e13_differentiable_optics.ipynb) |

E13 connects parameterized phase optics to a real intensity loss, explicit reverse-mode gradients, finite-difference checks and backtracking updates. A difficult initialization demonstrates local stagnation despite a known reachable target. The chapter distinguishes automatic differentiation, adjoint derivatives, differentiable ray intersections and end-to-end optics/reconstruction design. It includes a bilingual lesson, CPU notebook, saved figures and instructor pack; it has no companion film. E01–E12 retain their film companions.

## Teaching sequence

Choose a starting point from the concept map, then follow connections and review the preparation links for unfamiliar foundations. Episode numbers identify material rather than prescribing a learning order. A classroom session can use 3 minutes of prediction, 8 minutes of browser exploration, 6 minutes of explanation, 3 minutes of concept checks, 20–25 minutes of notebook work, and 5 minutes of reflection. Optional sections suit a longer lab or independent study.

E04 now states the same condition in the film and lesson: fields always superpose; measured intensities add when the averaged cross term vanishes. E07’s film and guide distinguish a single propagating transmitted order from a uniform-film approximation. Mutual-intensity propagation in E04 and RCWA convergence in E07 remain optional; their core goals, quizzes, reports, and ten-point rubrics can be completed without those extensions.

E02 computes both Fresnel propagation legs around a thin lens and compares complex fields with the Fourier result. E05 reports measurement residual and phase-aligned reconstruction error at the same completed iteration, with all three starts shown on common scales. E10 explicitly switches from E09’s Yee solver to a scalar Fourier pupil model; its optional Helmholtz example is a further, separate model.

## Run and rebuild

From the repository root:

```bash
python -m venv .course-venv
.course-venv/bin/pip install -r teaching/requirements-dev.txt
.course-venv/bin/python teaching/tools/build_course.py
npm ci --prefix teaching
.course-venv/bin/python teaching/tools/build_references.py
npm test --prefix teaching
.course-venv/bin/python teaching/tools/validate_notebooks.py
.course-venv/bin/python teaching/tools/render_heroes.py
python teaching/tools/serve_course.py
```

Open [the local course](http://127.0.0.1:8787/). In a second terminal, install the QA browser once with `npx --prefix teaching playwright install chromium`, then run `npm run test:browser --prefix teaching`. Set `COURSE_URL` to test another served copy.

The lesson code uses static HTML/CSS, native modules, and Canvas, with no build dependency or backend. Published films stream from YouTube; local preview films use a native video player. Both include direct viewing links. Heavy Yee and optimization loops run in cancellable workers. `tools/curriculum.py` contains authored content, `tools/motivation.py` holds the opening problems and learning payoffs, `tools/preparation.py` holds the shared prerequisite checklists and resource selections, `videos.json` holds film metadata for the generated lessons, and `course.json` is the generated navigation manifest. Each episode has a separate physical `model.js`; notebook source is the percent-format `lab.py`. Rebuilding preserves executed outputs when code cells are unchanged, including after edits to preparation or other Markdown.

`tools/differentiable_course.py` and `tools/differentiable_terms.py` author E13 and its paired translations. `shared/differentiable-optics.js` implements the browser calculation; the standalone NumPy lab independently checks it. The catalog’s `hasVideo` flag lets a complete lesson omit film controls without dead links.

`tools/geometric_course.py` holds the paired EN/ZH geometric lessons. `tools/geometric_terms.py` supplies their glossary and runtime translations. `tools/build_ray_labs.py` embeds the canonical Python ray engine into staged notebook sources, keeping the downloads independent of the repository.

`shared/glossary.js` contains the authored term definitions, matching aliases, chapter-specific uses, and explicit episode mappings for terms taught in films or optional material. `shared/reader.js` adds the section navigation and marks each term's first prose occurrence per section, excluding controls, code, equations, and quiz choices. `shared/term-panel.js` supplies the explanation panel to both the chapter reader and `shared/home-terms.js`. `shared/reader.css` adapts the content to the space beside the panels without reducing text size. `tests/reader.mjs` checks all thirteen chapters at six widths, including navigation, unchanged prose, term coverage, keyboard access, focus restoration, reading position, and modal behavior.

`tools/locales/zh.json` holds the reviewed Chinese phrases and term aliases. The course builder calls `tools/localization.py` to create sibling `index.zh.html` pages and `shared/zh.js`; missing page translations fail the build. Update the catalog when editing English content, then rebuild instead of editing generated Chinese files. Equations, code, and numerical configuration are preserved, and both languages share the same models. `shared/language.js` translates runtime text and handles preference and state-preserving navigation. `tests/languages.mjs` checks the English default, remembered choice, round-trip controls and section anchors, Chinese glossary and feedback, phone layouts, and navigation without JavaScript or browser storage.

`tools/video_logic_data.py` authors the paired EN/ZH outlines and explanations; `tools/video_logic.py` renders them and contributes their translations to the same language build. Times follow the published narration’s segment starts, rounded down to seconds, and do not require production files to build. `shared/video-guide.js` opens explanation anchors, cues the film, and retains disclosure and spectrum-control state through language changes. `shared/coherence-spectrum.js` computes the frequency-Gaussian example; `tests/check-spectrum.mjs` checks it against independent Fourier quadrature. `tests/video-guides.mjs` covers all twelve guides in both languages on desktop and phone, additional narrow layouts, direct timestamps, language/reload state, and static fallbacks. Browser checks isolate the external iframe and verify seek URLs, not YouTube streaming playback. `tests/course-logic.mjs` checks prediction/guide ordering in both languages at desktop and phone widths, the core/optional split, the E02/E05 evidence, and E10’s model comparison; it runs as part of the browser command and CI.

`tools/knowledge.py` authors concept names, descriptions, foundation links, episode mappings, and typed relationships independently of episode order. The builder renders every concept and destination into static HTML; `shared/knowledge.js` adds selection and responsive connection routing. With JavaScript disabled, concept links jump to their complete episode lists. `tests/knowledge.mjs` checks both languages at six widths, all thirteen chapter destinations, node and edge geometry, keyboard and history navigation, selection across language switches, and static fallback links.

Each chapter opens with links to its concepts and ends with the same interactive map in that chapter's context. `EPISODE_FOCUS` chooses an initial concept, while all concepts mapped to the chapter stay highlighted as readers explore. The cards offer other chapters using the selected concept and up to three routes through explicitly connected ideas, with the reason for each connection. Routes exclude the current chapter and duplicate destinations, and carry the connecting concept into the destination's map. `tests/episode-knowledge.mjs` checks all thirteen chapters in both languages, persistent highlights, explained destinations, language and experiment state, browser history, reading-panel layouts, and navigation without JavaScript.

The map has a core-concept layer and an on-demand term layer. `shared/term-network.js` maps all 104 glossary terms to one or more core concepts and curates their direct connections; sharing a category does not automatically connect two terms. The builder exports the overview's source data to `shared/knowledge-data.js`, so term cards and map panels use the same concept names and chapter routes. Expanding a concept reveals its specific terms without adding nodes to the overview. Each explanation links to its core concepts, nearby terms, and chapters that teach that exact term. Cards initially show three connections and three chapter routes, with the rest behind disclosure controls. A chapter-specific explanation appears only when one has been authored for that term.

Term destinations use `?term=<id>` alongside the selected concept and a normal section anchor. The destination opens the matching explanation; related-term navigation supports browser history, language switching, and direct links. The card's current-chapter entry locates the term in the text, or the video guide for concepts introduced there. With JavaScript disabled, homepage keywords link directly to their chapters, and ordinary chapter text and the static core-concept map remain usable. `tests/term-network.mjs` checks the complete data graph, all translated term contexts, progressive disclosure, exact-term destinations, navigation history, modal focus, responsive layouts, and preserved experiment controls. `tests/keyword-concepts.mjs` checks all 58 homepage keyword bindings in both languages on desktop and phone, the additional concept destinations, local graph links, keyboard and history behavior, language preservation, narrow layouts, new-tab links, and static fallbacks.

`tools/export_course.py <destination>` packages the complete static course, notebook downloads, and figures without development environments or tests. While E04/E07/E11/E12 use local film previews, use `--include-local-media` for a review package; a normal website export requires published film metadata first. The manual Pages workflow publishes that package only when dispatched. CI independently runs numerical checks, all notebooks, and browser tests; a push does not trigger Pages publication.

`shared/search-index.js` searches the same 108 terms, reviewed Chinese translations, aliases, and keyword bindings used by the explanations. Exact names rank ahead of partial matches and core-concept matches; case, full-width Latin text, spaces, and punctuation are normalized. `shared/concept-search.js` adds the search interface and its URL, keyboard, and Chinese-input handling. `tests/check-search.mjs` checks every authored name, alias, and episode keyword; `tests/concept-search.mjs` checks both page languages, all twelve chapter maps, explanation and episode routes, input composition, focus, history, language retention, new-tab behavior, and narrow layouts. Without JavaScript, the static concept map and its episode links remain available.

## Public interfaces and compatibility

Each `eNN-topic/` route is stable, with experiment values encoded in query parameters. E03 retains its public GitHub notebook path. Legacy root experiment parameters and section hashes redirect to `e03-diffraction/`, and the old root `e03_diffraction_limit.ipynb` download remains available. The root opens the concept map, followed by the thirteen-lesson index. On the homepage and every chapter, a `#concept-<id>` link selects a concept; the language switch also carries a `concept` query parameter so the selection survives while another section is in view.

Chinese pages use `index.zh.html` in the same folders, including the [Chinese homepage](https://vandermode.github.io/optics-in-5-minutes-course/index.zh.html). Their internal lesson and preparation links retain Chinese, and their experiment links include the page language as well as parameter values. Both languages have readable static content and ordinary language links when JavaScript is disabled.

Each notebook stores static PNG outputs. Every lesson also provides `figures.zip` and individual PNGs in `figures/`. In Colab, use Runtime → Run all; if an output iframe fails, use the notebook's Files sidebar or the public figure download. See [verification details and limits](VALIDATION.md) for the distinction between local execution and live Colab frontend checks.

## Model boundaries

The geometric-ray lessons use monochromatic transmitted rays without diffraction, polarization, Fresnel power weights or material dispersion. The design merit measures centroid-relative geometric spread, so it does not constrain distortion or certify PSF/MTF quality.

All demonstrations state their normalization, units, and limitations beside the experiment. Scalar angular-spectrum and thin-element models do not claim vector Maxwell accuracy. The grating-angle display does not invent efficiencies; the notebook's RCWA supplies those. The ring and coupler are reduced models. CPML error is measured against a reference. Adjoint derivatives are checked against finite differences, and optimization has no global-optimum guarantee.

The Lato fonts retain their [license](shared/fonts/LICENSE-Lato.txt). Primary references are listed with each lesson; figures are generated from the lesson models.
