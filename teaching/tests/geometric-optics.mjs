import {chromium} from '@playwright/test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import {isolateNativeMedia} from './native-media-fixture.mjs';
import {useCourseTransport} from './course-transport.mjs';

const root=new URL('../',import.meta.url),base=process.env.COURSE_URL||'http://127.0.0.1:8787/';
const catalog=JSON.parse(fs.readFileSync(new URL('course.json',root))).filter(e=>['E11','E12'].includes(e.id));
const out=new URL('.qa/browser/geometric-optics/',root);
fs.mkdirSync(out,{recursive:true});
const browser=await chromium.launch({headless:true}),errors=[],cases=[];
let activeCase='';
const complete={en:'Calculation complete.',zh:'计算完成。'};
async function ready(page,language) {
  await page.waitForFunction(text=>document.getElementById('status')?.textContent===text,complete[language]);
}
async function setRange(page,id,value) {
  await page.locator('#'+id).evaluate((el,value)=>{el.value=value;el.dispatchEvent(new Event('input',{bubbles:true}));},String(value));
}
async function metric(page,index) {return page.locator('#metrics .metric strong').nth(index).innerText();}
async function seekReady(page,time) {
  await page.waitForFunction(time=>{const v=document.querySelector('#video video');return v.readyState>=1&&Math.abs(v.currentTime-time)<3;},time);
  await page.locator('#video video').evaluate(v=>v.pause());
}

try {
  for(const language of ['en','zh']) for(const width of [1440,390,320]) {
    const context=await browser.newContext({viewport:{width,height:1000},reducedMotion:'reduce'});
    await useCourseTransport(context,base);
    await isolateNativeMedia(context);
    const page=await context.newPage();
    page.on('pageerror',error=>errors.push(error.message));
    page.on('console',message=>{if(message.type()==='error')errors.push(message.text());});
    for(const entry of catalog) {
      const label=`${entry.id} ${language} ${width}`;
      activeCase=label;
      await page.goto(new URL(`${entry.slug}/index${language==='zh'?'.zh':''}.html`,base).href);
      await ready(page,language);
      await page.locator('.reader-ready').waitFor();
      assert.ok(await page.evaluate(()=>{
        const prediction=document.querySelector('#lab > .callout'),controls=document.querySelector('#lab .controls');
        return Boolean(prediction.compareDocumentPosition(controls)&Node.DOCUMENT_POSITION_FOLLOWING);
      }),label+' prediction precedes controls');
      assert.equal(await page.locator('#lab canvas').count(),entry.id==='E11'?4:5,label+' complete experiment figures');
      assert.equal(await page.locator('#metrics .metric').count(),5,label+' computed measurements');
      assert.ok(await page.locator('#lab canvas').evaluateAll(nodes=>nodes.every(c=>c.width>0&&c.height>0)),label+' all figures rendered');
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),label+' no horizontal overflow');
      if(language==='zh') {
        assert.ok(await page.locator('#metrics .metric > span, #plots h3, #run-note').evaluateAll(nodes=>nodes.every(n=>/[\u3400-\u9fff]/.test(n.textContent))),label+' translated model labels');
      }

      if(entry.id==='E11') {
        await setRange(page,'aperture',2);await ready(page,language);
        const narrow=Number.parseFloat(await metric(page,2));
        await setRange(page,'aperture',8);await ready(page,language);
        const wide=Number.parseFloat(await metric(page,2));
        assert.ok(wide>narrow*10,label+' widening the same lens reveals geometric aberration');
        assert.ok(Math.abs(Number.parseFloat(await metric(page,3)))<.001,label+' axial centroid remains centred');
        await setRange(page,'field',6);await ready(page,language);
        assert.ok(Number.parseFloat(await metric(page,3))>4,label+' field changes absolute image position');
        assert.equal(await metric(page,4),'96 / 96',label+' all launched rays accounted for');
      } else {
        const fieldBending=Number((await metric(page,0)).split('→')[1]);
        assert.ok(fieldBending>.45&&fieldBending<.49,label+' field objective reaches verified bending');
        const focalEdge=(await metric(page,3)).split('/').map(Number.parseFloat);
        assert.equal(focalEdge[0],50,label+' optimization keeps focal power fixed');
        assert.ok(focalEdge[1]>.5,label+' positive edge clearance');
        assert.equal(await metric(page,4),'0',label+' no audit ray loss');
        await page.locator('#objective').selectOption('axis');
        // Stop in the same event turn, before a worker can publish a new design.
        const previous=await page.locator('#metrics').innerText();
        await page.evaluate(()=>{document.getElementById('run').click();document.getElementById('cancel').click();});
        assert.equal(await page.locator('#cancel').isDisabled(),true,label+' cancelled worker releases controls');
        assert.equal(await page.locator('#metrics').innerText(),previous,label+' cancellation retains the last completed result');
        await page.locator('#run').click();await ready(page,language);
        const axisBending=Number((await metric(page,0)).split('→')[1]);
        assert.ok(axisBending>.67&&axisBending<.71&&axisBending>fieldBending+.15,label+' the objective changes the selected lens');
        assert.equal(await page.locator('#aperture').inputValue(),'6',label+' aperture remains fixed across objectives');
      }
      await page.locator('#plots').scrollIntoViewIfNeeded();
      await page.screenshot({path:new URL(`${entry.id}-${language}-${width}-experiment.png`,out).pathname});

      // Real HTMLMediaElement metadata and seek behavior, using a synthetic film.
      const timestamp=page.locator('.video-flow [data-video-seek]').last();
      const at=Number(await timestamp.getAttribute('data-video-seek'));
      await timestamp.click();await seekReady(page,at);
      assert.equal(await page.locator('#video video track').getAttribute('srclang'),'en');
      assert.ok(await page.locator('#video video').evaluate((v,at)=>v.duration>=at&&v.videoWidth>0,at),label+' native fixture decodes');
      const next=language==='en'?'zh':'en';
      await page.locator(`header [data-language="${next}"]`).click();
      await ready(page,next);await seekReady(page,at);
      assert.equal(new URL(page.url()).searchParams.get('videoAt'),String(at),label+' language change carries selected scene');
      if(entry.id==='E11') {
        assert.equal(await page.locator('#field').inputValue(),'6');
        assert.equal(await page.locator('#aperture').inputValue(),'8');
      } else assert.equal(await page.locator('#objective').inputValue(),'axis');
      await page.reload();await ready(page,next);await seekReady(page,at);
      assert.ok(await page.evaluate(()=>document.documentElement.scrollWidth<=innerWidth+1),label+' language/reload layout');
      cases.push({episode:entry.id,language,width,controls:'passed',nativeSeek:'passed',languageReload:'passed'});
      console.log('PASS geometric optics '+label+': physical control effects, worker stop, native seek, language/reload');
    }
    await context.unrouteAll({behavior:'wait'});
    await context.close();
  }
  assert.deepEqual(errors,[],'Geometric-optics browser errors');
  fs.writeFileSync(new URL('report.json',out),JSON.stringify({cases,errors,media:'Generated synthetic fixture validates native controls; production film playback is a separate release check.'},null,2)+'\n');
  console.log('PASS geometric optics: 12 bilingual desktop/mobile/narrow cases.');
} catch(error) {
  fs.writeFileSync(new URL('failure.json',out),JSON.stringify({activeCase,message:error.message,errors,cases},null,2)+'\n');
  throw error;
} finally {await browser.close();}
