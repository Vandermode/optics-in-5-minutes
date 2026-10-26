import {chromium} from '@playwright/test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {createHash} from 'node:crypto';
import {isolateNativeMedia} from './native-media-fixture.mjs';
import {useCourseTransport} from './course-transport.mjs';
import {phrases} from '../shared/zh.js';

const root=new URL('../',import.meta.url);
const base=process.env.COURSE_URL||'http://127.0.0.1:8787/';
const catalog=JSON.parse(fs.readFileSync(new URL('course.json',root)));
const videos=JSON.parse(fs.readFileSync(new URL('videos.json',root)));
const revisedNotebooks=new Set(['E02','E04','E05','E07','E09','E10']);
const out=new URL('.qa/browser/course-logic/',root);
fs.mkdirSync(out,{recursive:true});
for(let n=1;n<=12;n++) assert.ok(catalog.some(entry=>entry.id===`E${String(n).padStart(2,'0')}`),'Existing course episode retained');

const normalize=text=>text.replace(/\s+/g,' ').trim();
function translated(text,language) {
  if(language==='en')return text;
  const value=phrases[normalize(text)];
  assert.ok(value,'Chinese translation exists: '+text);
  return value;
}
const errors=[],requestFailures=[],cases=[];
let activeCase='';
const browser=await chromium.launch({headless:true});

async function newContext(width) {
  const context=await browser.newContext({viewport:{width,height:1000},reducedMotion:'reduce'});
  context.setDefaultTimeout(30000);
  await useCourseTransport(context,base);
  await context.route('https://www.youtube-nocookie.com/embed/**',route=>route.fulfill({status:200,contentType:'text/html',body:''}));
  await isolateNativeMedia(context);
  context.on('requestfailed',request=>{
    if(request.failure()?.errorText!=='net::ERR_ABORTED') requestFailures.push({case:activeCase,url:request.url(),error:request.failure()?.errorText});
  });
  return context;
}

async function coreAndExtension(page,entry,language,label) {
  const config=await page.locator('#lesson-config').evaluate(node=>JSON.parse(node.textContent));
  const optional=page.locator('#notebook details.optional-extension');
  assert.equal(await optional.count(),1,label+' one clearly separated extension');
  await optional.evaluate(node=>{node.open=false;});
  assert.ok(config.optional_goals.length>0&&config.optional_exercises.length>0,label+' advanced outcomes and work retained');
  const advanced=entry.id==='E04'?/coherent.mode|mutual.intensity|diagonali[sz]|eigen|相干模|互强度|对角化|特征向量/i:/RCWA|convergence audit|谐波截断|收敛审查/i;
  assert.doesNotMatch(JSON.stringify([config.goals,config.exercises,config.quiz]),advanced,label+' core grading does not require extension content');
  assert.doesNotMatch(await page.locator('#idea ol').first().innerText(),advanced,label+' rendered core goals');
  assert.doesNotMatch(await page.locator('#notebook ol').first().innerText(),advanced,label+' rendered required report');
  assert.equal(await page.locator('#quiz fieldset').count(),config.quiz.length,label+' only core questions graded');
  assert.equal(await page.locator('#notebook ol').first().locator(':scope > li').count(),config.exercises.length,label+' required report matches core work');

  // Completing the core quiz succeeds while the advanced work remains unopened.
  for(const [i,question] of config.quiz.entries()) await page.locator(`#quiz input[name="q${i}"][value="${question.answer}"]`).check();
  await page.locator('#quiz button[type="submit"]').click();
  assert.equal(await page.locator('#quiz .feedback.wrong').count(),0,label+' core quiz passes without extension');
  const score=translated('{correct} of {total} correct. Read the reasoning above.',language)
    .replace('{correct}',String(config.quiz.length)).replace('{total}',String(config.quiz.length));
  assert.equal(await page.locator('#quiz-status').innerText(),score,label+' complete core score');
  assert.equal(await optional.getAttribute('open'),null,label+' extension was not needed to complete quiz');
  await optional.locator(':scope > summary').click();
  assert.ok(await optional.locator('.answer').isVisible(),label+' advanced work is still accessible');
  const optionalText=await optional.innerText();
  assert.match(optionalText,entry.id==='E04'?/coherent.mode|相干模/i:/RCWA/i,label+' advanced topic retained');
  assert.ok(normalize(optionalText).includes(normalize(translated('These extension tasks are separate from the core ten-point rubric.',language))),label+' separate assessment stated');
  assert.equal(await optional.locator('input, form').count(),0,label+' optional work does not add hidden core quiz requirements');
  assert.equal(await page.locator('#quiz-status').innerText(),score,label+' opening extension does not change core score');
  await optional.screenshot({path:new URL(`${entry.id}-${language}-${page.viewportSize().width}-optional.png`,out).pathname});
}

async function correctedLocalFilm(page,entry,label) {
  const metadata=videos[entry.id],video=page.locator('#video video');
  assert.ok(metadata.file&&metadata.captions&&!metadata.youtubeId,label+' corrected local film replaces the older embed');
  assert.equal(await page.locator('#video iframe').count(),0,label+' no stale published player');
  assert.equal(await video.count(),1,label+' native player present');
  assert.equal(await video.getAttribute('src'),metadata.file,label+' native movie source');
  assert.equal(await video.locator('track').getAttribute('src'),metadata.captions,label+' matching English captions');
  assert.equal(await video.locator('track').getAttribute('srclang'),'en');
  assert.ok((await page.locator('.video-caption').innerText()).includes(metadata.duration),label+' displayed duration matches release metadata');
  const [minutes,seconds]=metadata.duration.split(':').map(Number),duration=60*minutes+seconds;
  assert.ok(duration>=285&&duration<=330,label+' release within the stated film-length range');
  const last=page.locator('.video-flow [data-video-seek]').last();
  const at=Number(await last.getAttribute('data-video-seek'));
  assert.ok(at<duration,label+' last guide entry falls inside the declared film');
  assert.ok((await last.getAttribute('href')).startsWith(metadata.file+'#t='),label+' guide targets corrected film');
  await last.click();
  await page.waitForFunction(time=>{
    const player=document.querySelector('#video video');
    return player.readyState>=1&&Math.abs(player.currentTime-time)<3;
  },at);
  await video.evaluate(player=>player.pause());
  assert.ok(await video.evaluate(player=>player.videoWidth>0&&player.error===null),label+' native seek fixture decodes');
}

async function currentNotebookDownload(page,entry,label) {
  const actions=page.locator('#notebook .actions');
  const hrefs=await actions.locator('a[href]').evaluateAll(links=>links.map(link=>link.href));
  assert.ok(hrefs.length>0,label+' notebook actions are present');
  assert.ok(hrefs.every(href=>new URL(href).hostname!=='colab.research.google.com'),label+' revised notebook does not point to the older public Colab copy');
  const link=actions.locator('a[download][href$=".ipynb"]');
  assert.equal(await link.count(),1,label+' one current notebook download');
  assert.equal(await link.getAttribute('href'),entry.notebook,label+' download targets the current lesson notebook');
  const pending=page.waitForEvent('download');
  await link.click();
  const download=await pending;
  assert.equal(download.suggestedFilename(),entry.notebook,label+' notebook download filename');
  assert.equal(await download.failure(),null,label+' notebook download completes');
  const actual=fs.readFileSync(await download.path());
  const expected=fs.readFileSync(new URL(`${entry.slug}/${entry.notebook}`,root));
  const digest=body=>createHash('sha256').update(body).digest('hex');
  assert.equal(digest(actual),digest(expected),label+' downloaded notebook matches the current local artifact');
  assert.equal(JSON.parse(actual).nbformat,4,label+' downloaded file is a valid notebook');
  await download.delete();
}

try {
  for(const language of ['en','zh']) for(const width of [1440,390]) {
    const context=await newContext(width),page=await context.newPage();
    page.on('pageerror',error=>errors.push({case:activeCase,message:error.message}));
    page.on('console',message=>{if(message.type()==='error')errors.push({case:activeCase,message:message.text()});});
    for(const entry of catalog) {
      const label=activeCase=`${entry.id} ${language} ${width}`;
      const response=await page.goto(new URL(`${entry.slug}/index${language==='zh'?'.zh':''}.html`,base).href);
      assert.equal(response.status(),200,label+' route');
      await page.locator('.reader-ready').waitFor();
      await page.evaluate(()=>document.fonts.ready);
      if(entry.id!=='E03') await page.waitForFunction(text=>document.querySelector('#status')?.textContent===text,language==='zh'?'计算完成。':'Calculation complete.');
      assert.equal(await page.locator('html').getAttribute('lang'),language==='zh'?'zh-CN':'en',label+' requested language');

      const order=await page.evaluate(()=>{
        const prediction=document.querySelector('#lab > .callout, #lab > .prediction');
        const guide=document.querySelector('#lab #lab-guide'),controls=document.querySelector('#lab .controls');
        if(!prediction||!guide||!controls)return null;
        const before=(a,b)=>Boolean(a.compareDocumentPosition(b)&Node.DOCUMENT_POSITION_FOLLOWING);
        const p=prediction.getBoundingClientRect(),g=guide.getBoundingClientRect(),c=controls.getBoundingClientRect();
        return {predictionBeforeGuide:before(prediction,guide),guideBeforeControls:before(guide,controls),
          predictionAboveGuide:p.bottom<=g.top+1,guideAboveControls:g.bottom<=c.top+1,
          steps:guide.querySelectorAll('ol > li').length,predictionText:prediction.textContent.trim().length};
      });
      assert.ok(order,label+' prediction, guide and controls all inside experiment');
      assert.ok(order.predictionBeforeGuide&&order.guideBeforeControls,label+' reading order is prediction → guidance → controls');
      assert.ok(order.predictionAboveGuide&&order.guideAboveControls,label+' visual order matches reading order');
      assert.ok(order.steps>=2&&order.predictionText>30,label+' instructions contain an actual prediction and multiple actions');
      assert.equal(await page.locator('#lab-guide').count(),1,label+' guidance is not duplicated');
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),label+' page has no horizontal overflow');

      if(revisedNotebooks.has(entry.id)) await currentNotebookDownload(page,entry,label);
      if(['E04','E07'].includes(entry.id)) {
        await coreAndExtension(page,entry,language,label);
        await correctedLocalFilm(page,entry,label);
      }
      if(entry.id==='E02') {
        const text=normalize(await page.locator('#explain').innerText());
        assert.ok(text.includes('U → L → P_f')&&text.includes('U → P_f → L → P_f'),label+' both physical paths explained');
        assert.match(text,language==='zh'?/积分/:/integral/i,label+' direct propagation is explained');
        assert.match(text,language==='zh'?/二次相位/:/quadratic phase/i,label+' remaining coordinate phase is explained');
        assert.match(text,language==='zh'?/复(?:数|光)?场/:/complex fields/i,label+' complex fields compared beyond intensity');
      }
      if(entry.id==='E05') {
        const figure=page.locator('#explain .evidence-figure'),img=figure.locator('img');
        assert.equal(await img.getAttribute('src'),'figures/03-phase-retrieval.png',label+' saved computed comparison embedded');
        await img.scrollIntoViewIfNeeded();
        // A lazy image may not have scheduled its fetch in the scroll event turn yet.
        await page.waitForFunction(()=>{const image=document.querySelector('#explain .evidence-figure img');return image.complete&&image.naturalWidth>0;});
        await img.evaluate(image=>image.decode());
        assert.ok(await img.isVisible(),label+' reconstruction evidence is visible');
        assert.ok(await img.evaluate(image=>{
          const box=image.getBoundingClientRect();
          return image.naturalWidth>1000&&image.naturalHeight>500&&box.width>200&&
            Math.abs(box.width/box.height-image.naturalWidth/image.naturalHeight)<.01;
        }),label+' full-resolution evidence is loaded without distortion');
        const text=await page.locator('#explain').innerText();
        assert.match(text,/r = ‖\|F\(u\)\| − b‖₂ \/ ‖b‖₂/,label+' measurement residual defined');
        assert.match(text,/e = min_α/,label+' phase-aligned field error defined separately');
        assert.match(text,language==='zh'?/同一.*迭代/:/same completed iteration/i,label+' both metrics share an iterate');
        assert.match(await figure.innerText(),/0\.654–0\.786%/,label+' measured residual range reported');
        assert.match(await figure.innerText(),/14\.6–18\.3%/,label+' field-error range reported');
        await figure.screenshot({path:new URL(`${entry.id}-${language}-${width}-evidence.png`,out).pathname});
      }
      if(entry.id==='E10') {
        const bridge=page.locator('#model-bridge');
        assert.equal(await bridge.locator('table').count(),1,label+' model comparison table');
        assert.equal(await bridge.locator('thead th').count(),3);
        assert.equal(await bridge.locator('tbody tr').count(),4);
        const cells=await bridge.locator('tbody tr').first().locator('td').allInnerTexts();
        assert.match(cells[0],/Yee/,label+' E09 time-domain field propagation');
        assert.match(cells[1],language==='zh'?/傅里叶/:/Fourier/,label+' E10 Fourier pupil propagation');
        assert.match(await page.locator('#idea').innerText(),language==='zh'?/而非对 Yee 时间更新/:/not the Yee time updates/,label+' model change explicit');
        await bridge.scrollIntoViewIfNeeded();
        const overflow=await bridge.evaluate(node=>({overflow:node.scrollWidth>node.clientWidth+1,mode:getComputedStyle(node).overflowX}));
        if(overflow.overflow) {
          assert.ok(['auto','scroll'].includes(overflow.mode),label+' narrow comparison can scroll');
          await bridge.evaluate(node=>{node.scrollLeft=node.scrollWidth;});
          assert.ok(await bridge.evaluate(node=>node.scrollLeft>0),label+' E10 comparison column reachable');
        }
        await bridge.screenshot({path:new URL(`${entry.id}-${language}-${width}-models.png`,out).pathname});
      }
      cases.push({episode:entry.id,language,width,order:'passed',focusedChecks:['E02','E04','E05','E07','E10'].includes(entry.id)?'passed':'not applicable'});
      console.log('PASS course logic '+label);
    }
    await context.unrouteAll({behavior:'wait'});
    await context.close();
  }
  assert.deepEqual(errors,[],'No browser errors in revised course flows');
  fs.writeFileSync(new URL('report.json',out),JSON.stringify({cases,errors,requestFailures,
    scope:'All lessons: prediction and guidance before controls. Six revised lessons: download the current notebook without stale Colab routes. E04/E07: core-only assessment and local-film routing. E02: complete propagation path. E05: computed comparison and both errors. E10: explicit model change.',
    media:'Synthetic WebM verifies native controls only; actual corrected films require separate production playback and content review.'},null,2)+'\n');
  console.log(`PASS course logic: ${cases.length} bilingual desktop/mobile cases.`);
} catch(error) {
  fs.writeFileSync(new URL('failure.json',out),JSON.stringify({activeCase,errors,requestFailures,message:error.message},null,2)+'\n');
  throw error;
} finally {await browser.close();}
