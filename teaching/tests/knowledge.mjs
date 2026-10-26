import {isolateNativeMedia} from './native-media-fixture.mjs';
import {useCourseTransport} from './course-transport.mjs';
import {chromium} from '@playwright/test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {geometry} from './knowledge-geometry.mjs';
import {knowledge} from '../shared/knowledge-data.js';

const root = new URL('../', import.meta.url);
const base = process.env.COURSE_URL || 'http://127.0.0.1:8787/';
const catalog = JSON.parse(fs.readFileSync(new URL('course.json', root)));
const out = new URL('.qa/browser/knowledge/', root);
fs.mkdirSync(out, {recursive: true});
const browser = await chromium.launch({headless: true});
const results = [], errors = [], internalLinks = new Set(), mappings = new Map();
let activeCase = '';

async function ready(page, language) {
  await page.locator('.knowledge-ready').waitFor();
  await page.evaluate(() => document.fonts.ready);
  assert.equal(await page.locator('html').getAttribute('lang'), language === 'zh' ? 'zh-CN' : 'en');
}


try {
  for (const width of [1440, 1280, 1024, 768, 390, 320]) for (const language of ['en', 'zh']) {
    const context = await browser.newContext({viewport: {width, height: 1000}, reducedMotion: 'reduce', locale: 'zh-CN'});
    await useCourseTransport(context, base);
    activeCase = `${language} ${width}px`;
    context.setDefaultTimeout(20000);
    const page = await context.newPage();
    page.on('pageerror', error => errors.push(error.message));
    page.on('console', message => {if (message.type() === 'error') errors.push(message.text());});
    await page.goto(base + (language === 'zh' ? 'index.zh.html' : ''));
    await ready(page, language);
    const label = `${language} ${width}px`;
    assert.equal(await page.locator('.concept-node').count(), knowledge.concepts.length, label);
    assert.equal(await page.locator('.concept-region').count(), knowledge.regions.length, label);
    assert.equal(await page.locator('.concept-node[aria-current]').getAttribute('data-concept'), 'fourier');
    assert.ok(await page.locator('#knowledge-map').evaluate(map => Boolean(map.compareDocumentPosition(document.getElementById('course')) & Node.DOCUMENT_POSITION_FOLLOWING)), 'Map precedes episode index');
    assert.equal(await page.locator('.course-card').count(), catalog.length, 'Episode index remains available');
    await geometry(page, label);
    const episodes = new Set();
    for (const node of await page.locator('.concept-node').all()) {
      const id = await node.getAttribute('data-concept');
      await node.click();
      const panel = page.locator('#concept-' + id);
      assert.equal(new URL(page.url()).hash, '#concept-' + id);
      assert.equal(await page.locator('.concept-panel:visible').count(), 1);
      assert.equal(await node.getAttribute('aria-current'), 'true');
      assert.equal(await panel.locator('h3').innerText(), await node.innerText());
      const associated = await panel.locator('[data-episode-link]').evaluateAll(links => links.map(link => link.dataset.episodeLink));
      assert.ok(associated.length > 0, label + ' reachable episodes');
      for (const episode of associated) episodes.add(episode);
      if (mappings.has(id)) assert.deepEqual(associated, mappings.get(id), 'Same concept mappings across languages and widths');
      else mappings.set(id, associated);
      for (const link of await panel.locator('a[href]').all()) {
        const url = new URL(await link.getAttribute('href'), page.url());
        if (url.hash.startsWith('#concept-') || url.hash === '#knowledge-map') continue;
        assert.equal(url.origin, new URL(base).origin);
        if (language === 'zh') assert.ok(url.pathname.endsWith('/index.zh.html'), 'Chinese graph destinations');
        internalLinks.add(url.href);
      }
      if (width <= 1100) {
        assert.equal(await page.evaluate(() => document.activeElement.id), 'concept-' + id, 'Mobile focus follows episode routes');
        assert.ok(await panel.evaluate(el => el.getBoundingClientRect().top >= 0 && el.getBoundingClientRect().top < 100), 'Mobile routes brought into view');
      }
    }
    assert.deepEqual([...episodes].sort(), catalog.map(item => item.id), 'Every catalog episode is reachable');
    await geometry(page, label + ' after selections');

    // Native keyboard activation and browser history preserve the concept.
    const previousConcept = await page.locator('.concept-node[aria-current]').getAttribute('data-concept');
    await page.locator('[data-concept=field]').focus();
    await page.keyboard.press('Enter');
    assert.equal(new URL(page.url()).hash, '#concept-field');
    await page.goBack();
    assert.equal(await page.locator('.concept-node[aria-current]').getAttribute('data-concept'), previousConcept);
    await page.goForward();
    assert.equal(await page.locator('.concept-node[aria-current]').getAttribute('data-concept'), 'field');
    await page.locator('#concept-field .concept-context summary').click();
    await page.locator('#concept-field .concept-connections a[href="#concept-coherence"]').click();
    assert.equal(await page.locator('.concept-node[aria-current]').getAttribute('data-concept'), 'coherence');

    const otherLanguage = language === 'en' ? 'zh' : 'en';
    if (width <= 1100) await page.locator('#concept-coherence .concept-back').click();
    else await page.locator('header a[href="#knowledge-map"]').click();
    await page.locator('[data-concept=field]').click();
    await page.goBack();
    assert.equal(new URL(page.url()).hash, '#knowledge-map');
    assert.equal(await page.locator('.concept-node[aria-current]').getAttribute('data-concept'), 'coherence', 'Back to the map restores its earlier selection');
    await page.locator(`header [data-language="${otherLanguage}"]`).click();
    await ready(page, otherLanguage);
    assert.equal(await page.locator('.concept-node[aria-current]').getAttribute('data-concept'), 'coherence', 'Language switch keeps selected concept');
    await page.locator(`header [data-language="${language}"]`).click();
    await ready(page, language);
    assert.equal(await page.locator('.concept-node[aria-current]').getAttribute('data-concept'), 'coherence');

    await page.goto(base + (language === 'zh' ? 'index.zh.html' : 'index.html') + '#concept-adjoint');
    await ready(page, language);
    assert.equal(await page.locator('.concept-node[aria-current]').getAttribute('data-concept'), 'adjoint', 'Direct concept link');
    assert.deepEqual(await page.locator('#concept-adjoint [data-episode-link]').evaluateAll(links => links.map(link => link.dataset.episodeLink)), knowledge.concepts.find(c => c.id === 'adjoint').episodes);
    await page.goto(base + (language === 'zh' ? 'index.zh.html' : 'index.html') + '?concept=not-a-concept#knowledge-map');
    await ready(page, language);
    assert.equal(await page.locator('.concept-node[aria-current]').getAttribute('data-concept'), 'fourier', 'Unknown concept safely falls back');
    await geometry(page, label + ' after navigation');
    await page.screenshot({path: new URL(`${language}-${width}-home.png`, out).pathname});
    await page.locator('#knowledge-map').screenshot({path: new URL(`${language}-${width}-map.png`, out).pathname});
    results.push({language, width, concepts: knowledge.concepts.length, reachable_episodes: catalog.length, geometry: 'passed', navigation: 'passed'});
    console.log('PASS knowledge map ' + label);
    await context.unrouteAll({behavior: 'wait'});
    await context.close();
  }

  // Static concept links work without JavaScript, in either edition.
  for (const language of ['en', 'zh']) {
    const context = await browser.newContext({javaScriptEnabled: false, viewport: {width: 320, height: 900}, reducedMotion: 'reduce'});
    await useCourseTransport(context, base);
    activeCase = `No-JavaScript ${language}`;
    context.setDefaultTimeout(20000);
    await context.route('https://www.youtube-nocookie.com/embed/**', route => route.fulfill({status: 200, body: ''}));
    await isolateNativeMedia(context);
    const page = await context.newPage();
    await page.goto(base + (language === 'zh' ? 'index.zh.html' : 'index.html'));
    assert.equal(await page.locator('.concept-panel:visible').count(), knowledge.concepts.length);
    assert.ok(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth + 1), 'Static map fits a phone');
    await page.locator('[data-concept=adjoint]').click();
    assert.equal(new URL(page.url()).hash, '#concept-adjoint');
    await page.locator('#concept-adjoint [data-episode-link=E10]').click();
    assert.ok(new URL(page.url()).pathname.includes('/e10-adjoint-design/'));
    assert.equal(await page.locator('html').getAttribute('lang'), language === 'zh' ? 'zh-CN' : 'en');
    console.log('PASS static concept navigation ' + language);
    await context.unrouteAll({behavior: 'wait'});
    await context.close();
  }

  const context = await browser.newContext();
  activeCase = 'Destination anchors';
  const page = await context.newPage();
  for (const href of internalLinks) {
    const response = await context.request.get(href);
    assert.equal(response.status(), 200, href);
    const id = new URL(href).hash.slice(1);
    const html = await response.text();
    assert.ok(await page.evaluate(({html, id}) => Boolean(new DOMParser().parseFromString(html, 'text/html').getElementById(id)), {html, id}), 'Destination anchor: ' + href);
  }
  await context.close();
  assert.deepEqual(errors, []);
  fs.writeFileSync(new URL('report.json', out), JSON.stringify({results, checked_links: internalLinks.size, no_javascript: 'passed', errors}, null, 2) + '\n');
  console.log(`PASS concept navigation, bilingual state, no-JavaScript routes, and ${internalLinks.size} destination anchors.`);
} catch (error) {
  fs.writeFileSync(new URL('failure.json', out), JSON.stringify({activeCase, message: error.message, errors, results}, null, 2) + '\n');
  throw error;
} finally {
  await browser.close();
}
