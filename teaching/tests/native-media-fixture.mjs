// A generated, tiny, silent video exercises HTMLMediaElement metadata/seek without
// depending on unpublished production films. It is not a film-playback acceptance test.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import {spawnSync} from 'node:child_process';

let movie;
function fixture() {
  if(movie)return movie;
  const folder=fs.mkdtempSync(path.join(os.tmpdir(),'optics-media-test-'));
  // WebM works in Playwright's unbranded Chromium builds, which may lack H.264.
  // The original request path is retained to exercise the course's real links.
  const output=path.join(folder,'seek-fixture.webm');
  const result=spawnSync('ffmpeg',['-v','error','-y','-f','lavfi','-i','color=c=black:s=32x32:r=1:d=340',
    '-an','-c:v','libvpx','-deadline','realtime','-cpu-used','8','-threads','1','-pix_fmt','yuv420p',output],{encoding:'utf8'});
  if(result.status!==0)throw new Error('ffmpeg is required for the native-video control fixture: '+(result.error?.message||result.stderr));
  movie=fs.readFileSync(output);
  process.once('exit',()=>fs.rmSync(folder,{recursive:true,force:true}));
  return movie;
}

export async function isolateNativeMedia(context) {
  await context.route(/\/media\/e\d{2}-[^/]+\.en\.vtt(?:\?|$)/,route=>route.fulfill({status:200,contentType:'text/vtt',
    body:'WEBVTT\n\n00:00:00.000 --> 00:05:40.000\nSynthetic video-control fixture, not the course film.\n'}));
  await context.route(/\/media\/e\d{2}-[^/]+\.mp4(?:\?|$)/,route=>{
    const body=fixture(),range=route.request().headers().range?.match(/^bytes=(\d+)-(\d*)$/);
    if(range) {
      const start=Number(range[1]),end=Math.min(range[2]?Number(range[2]):body.length-1,body.length-1);
      return route.fulfill({status:206,contentType:'video/webm',headers:{'Accept-Ranges':'bytes','Content-Range':`bytes ${start}-${end}/${body.length}`},body:body.subarray(start,end+1)});
    }
    return route.fulfill({status:200,contentType:'video/webm',headers:{'Accept-Ranges':'bytes'},body});
  });
}
