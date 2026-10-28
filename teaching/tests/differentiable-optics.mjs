import {chromium} from '@playwright/test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {useCourseTransport} from './course-transport.mjs';
import {isolateNativeMedia} from './native-media-fixture.mjs';

const root=new URL('../',import.meta.url),base=process.env.COURSE_URL||'http://127.0.0.1:8787/';
const entry=JSON.parse(fs.readFileSync(new URL('course.json',root))).find(e=>e.id==='E13');
const film=JSON.parse(fs.readFileSync(new URL('videos.json',root))).E13;
assert.ok(entry,'E13 is part of the course catalog');
assert.ok(entry.hasVideo&&film?.file&&film?.captions,'E13 has a local film and captions');
const out=new URL('.qa/browser/differentiable/',root);
fs.mkdirSync(out,{recursive:true});
const browser=await chromium.launch({headless:true}),errors=[],requestFailures=[],cases=[];
let activeCase='';
const ready=(page,language)=>page.waitForFunction(text=>document.getElementById('status')?.textContent===text,language==='zh'?'计算完成。':'Calculation complete.');
const metric=(page,index)=>page.locator('#metrics .metric strong').nth(index).innerText();
async function losses(page) {return (await metric(page,0)).split('→').map(Number);}
function close(actual,expected,tolerance,label) {assert.ok(Number.isFinite(actual)&&Math.abs(actual-expected)<=tolerance,`${label}: ${actual} vs ${expected}`);}
async function preset(page,index,language) {await page.locator(`[data-preset="${index}"]`).click();await ready(page,language);}
async function download(page,link,expected,label) {
  const event=page.waitForEvent('download');await link.click();const item=await event;
  assert.equal(await item.failure(),null,label+' completes');
  const actual=fs.readFileSync(await item.path()),source=fs.readFileSync(expected);
  const digest=body=>createHash('sha256').update(body).digest('hex');
  assert.equal(digest(actual),digest(source),label+' matches the current local artifact');
  await item.delete();return actual;
}
try {
  for(const language of ['en','zh'])for(const width of [1440,390,320]) {
    activeCase=`E13 ${language} ${width}`;
    const context=await browser.newContext({viewport:{width,height:1000},reducedMotion:'reduce'});
    context.setDefaultTimeout(30000);
    await useCourseTransport(context,base);
    await isolateNativeMedia(context);
    context.on('requestfailed',request=>{if(request.failure()?.errorText!=='net::ERR_ABORTED')requestFailures.push({case:activeCase,url:request.url(),error:request.failure()?.errorText});});
    const page=await context.newPage();
    page.on('pageerror',error=>errors.push({case:activeCase,message:error.message}));
    page.on('console',message=>{if(message.type()==='error')errors.push({case:activeCase,message:message.text()});});
    const response=await page.goto(new URL(`${entry.slug}/index${language==='zh'?'.zh':''}.html`,base).href);
    assert.equal(response.status(),200,activeCase+' serves');
    await ready(page,language);await page.locator('.reader-ready').waitFor();await page.evaluate(()=>document.fonts.ready);
    assert.equal(await page.locator('#plots canvas').count(),4,activeCase+' four computed plots');
    assert.equal(await page.locator('#metrics .metric').count(),5,activeCase+' five computed measurements');
    assert.ok(await page.locator('#plots canvas').evaluateAll(nodes=>nodes.every(n=>n.width>0&&n.height>0)),activeCase+' nonempty plot canvases');
    assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),activeCase+' no page overflow');
    assert.equal(await page.locator('#video video').count(),1,activeCase+' native film player');
    assert.equal(await page.locator('#video video').getAttribute('src'),film.file,activeCase+' current local film route');
    assert.equal(await page.locator('#video track[kind="captions"]').getAttribute('src'),film.captions,activeCase+' English caption route');
    assert.equal(await page.locator('#video-guide .video-flow > li').count(),9,activeCase+' chronological film guide');
    assert.equal(await page.locator('#video iframe').count(),0,activeCase+' local film does not depend on an external player');
    const explanation=await page.locator('main').innerText();
    for(const term of (language==='zh'?['可微光学','自动微分','伴随法','端到端','有限差分']:['Differentiable optics','Automatic differentiation','adjoint','End-to-end','finite difference']))
      assert.ok(explanation.includes(term),activeCase+' explains '+term);
    assert.ok(await page.locator('#explain .equation').count()>=5,activeCase+' forward and reverse chain equations are present');
    const equations=await page.locator('.equation').evaluateAll(nodes=>nodes.map(n=>({client:n.clientWidth,scroll:n.scrollWidth,overflow:getComputedStyle(n).overflowX})));
    assert.ok(equations.every(e=>e.scroll<=e.client+1||['auto','scroll'].includes(e.overflow)),activeCase+' every long equation remains accessible');
    if(language==='zh') {
      assert.ok(await page.locator('#metrics .metric > span,#plots h3,#plots figcaption p,#run-note').evaluateAll(nodes=>nodes.every(n=>/[\u3400-\u9fff]/.test(n.textContent))),activeCase+' model labels and captions are translated');
      assert.ok(await page.locator('#plots .legend-item').evaluateAll(nodes=>nodes.every(n=>/[\u3400-\u9fff]/.test(n.textContent))),activeCase+' plot legends are translated');
    }
    const initial=await losses(page);
    close(initial[0],.2606,.0001,activeCase+' initial optical loss');
    assert.ok(initial[1]<initial[0]/1000,activeCase+' default optimization improves the measured objective');
    assert.equal(await metric(page,4),'40',activeCase+' forty accepted updates');
    assert.ok(Number(await metric(page,1))<1e-8,activeCase+' displayed independent gradient check');
    close(Number(await metric(page,3)),1,1e-8,activeCase+' fixed total power');
    await page.locator('#plots').screenshot({path:new URL(`${language}-${width}-plots.png`,out).pathname});
    await page.locator('#plots figure').nth(2).screenshot({path:new URL(`${language}-${width}-gradient.png`,out).pathname});
    await page.locator('#explain .equation').nth(1).screenshot({path:new URL(`${language}-${width}-reverse-equation.png`,out).pathname});
    await preset(page,0,language);
    const zero=await losses(page);
    close(zero[0],zero[1],0,activeCase+' zero updates retains the initial design');
    assert.equal(await metric(page,4),'0',activeCase+' zero accepted steps');
    assert.equal(await page.locator('#iterations').inputValue(),'0');
    await preset(page,2,language);
    const difficult=await losses(page);
    close(difficult[1],.2987,.0001,activeCase+' difficult start reveals nonzero plateau');
    assert.ok(difficult[1]<difficult[0]&&difficult[1]>.1,activeCase+' local improvement is distinguished from global success');
    assert.ok(await page.locator('#run-note').evaluate(n=>n.classList.contains('error')),activeCase+' unresolved target mismatch is visible');
    assert.equal(await page.locator('#tilt').inputValue(),'3');assert.equal(await page.locator('#defocus').inputValue(),'3');
    await page.locator('#plots figure').nth(3).screenshot({path:new URL(`${language}-${width}-plateau.png`,out).pathname});
    await preset(page,1,language);
    const restored=await losses(page);close(restored[1],initial[1],0,activeCase+' default preset restores reproducible result');
    // Language navigation carries parameter state; it must not silently change the task.
    await preset(page,0,language);
    const next=language==='en'?'zh':'en';
    await page.locator(`header [data-language="${next}"]`).click();await ready(page,next);
    assert.equal(await page.locator('#iterations').inputValue(),'0',activeCase+' language switch preserves zero updates');
    close((await losses(page))[0],zero[0],0,activeCase+' language switch preserves design loss');
    await page.reload();await ready(page,next);
    assert.equal(await metric(page,4),'0',activeCase+' shared URL survives reload');
    const notebookLink=page.locator('#notebook .actions a[download][href$=".ipynb"]');
    assert.equal(await notebookLink.count(),1,activeCase+' current notebook download');
    assert.equal(await notebookLink.getAttribute('href'),entry.notebook,activeCase+' catalog notebook route');
    assert.equal(await page.locator('#notebook .actions a[href*="colab.research.google.com"]').count(),0,activeCase+' no link to an unpublished Colab file');
    const notebook=await download(page,notebookLink,new URL(`${entry.slug}/${entry.notebook}`,root),activeCase+' notebook download');
    assert.equal(JSON.parse(notebook).nbformat,4,activeCase+' downloaded notebook format');
    const figurePath=new URL(`${entry.slug}/figures.zip`,root);
    let figureDownload='not yet generated';
    if(fs.existsSync(figurePath)) {
      const figures=await download(page,page.locator('#notebook .actions a[download][href="figures.zip"]'),figurePath,activeCase+' figure archive');
      assert.equal(figures.subarray(0,2).toString(),'PK',activeCase+' figure archive is a ZIP');figureDownload='passed';
    }
    cases.push({episode:'E13',language,width,plots:'passed',gradient:'passed',optimization:'passed',zeroSteps:'passed',localPlateau:'passed',languageReload:'passed',notebookDownload:'passed',figureDownload});
    console.log('PASS differentiable optics '+activeCase+': plots, gradient, zero/default/difficult starts, language state and downloads');
    await context.unrouteAll({behavior:'wait'});await context.close();
  }
  assert.deepEqual(errors,[],'No E13 browser errors');
  assert.deepEqual(requestFailures,[],'No E13 failed requests');
  fs.writeFileSync(new URL('report.json',out),JSON.stringify({cases,errors,requestFailures},null,2)+'\n');
  fs.rmSync(new URL('failure.json',out),{force:true});
  console.log('PASS differentiable optics: six bilingual desktop/mobile/narrow cases.');
} catch(error) {
  fs.writeFileSync(new URL('failure.json',out),JSON.stringify({activeCase,message:error.message,errors,requestFailures,cases},null,2)+'\n');throw error;
} finally {await browser.close();}
