import assert from 'node:assert/strict';
import fs from 'node:fs';
import {makeLens,paraxial,prescription,surfaceSag,intersectSurface,refract,pupilRay,pupilSamples,traceRay,traceBundle,bestImagePlane,designMerit,optimizeDesign} from '../shared/ray-optics.js';
import {compute as rayModel} from '../e11-ray-tracing/model.js';
import {compute as designModel} from '../e12-lens-design/model.js';
const reference=JSON.parse(fs.readFileSync(new URL('ray-references.json',import.meta.url)));
let count=0;
function close(a,b,tolerance,label){assert.ok(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=tolerance,`${label}: ${a} versus ${b}`);count++;}
function vector(a,b,tolerance,label){assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i],tolerance,label+' '+i));}
const norm=v=>Math.hypot(...v),dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
for(const theta of [0,.1,.5,.9]) {
  const d=[Math.sin(theta),0,Math.cos(theta)],out=refract(d,[0,0,1],1,1.5);
  assert.equal(out.status,'ok');close(1.5*out.direction[0],Math.sin(theta),1e-14,'Independent scalar Snell law');
  close(norm(out.direction),1,2e-15,'Unit transmitted direction');
  const reverse=refract(out.direction.map(v=>-v),[0,0,-1],1.5,1);
  vector(reverse.direction,d.map(v=>-v),1e-14,'Refraction reversal');
}
assert.equal(refract([Math.sin(Math.PI/3),0,.5],[0,0,1],1.5,1).status,'tir');
assert.equal(intersectSurface([13,0,-10],[0,0,1],0,50,12).status,'blocked');
assert.equal(intersectSurface([0,0,10],[0,0,1],0,50,12).status,'missed','Do not select the far hemisphere after the physical cap has already been passed');
vector(intersectSurface([0,0,-100],[0,0,1],4,-50,12).point,[0,0,4],1e-12,'Reject nearer wrong hemisphere for negative R');
close(intersectSurface([6,0,-10],[0,0,1],0,50,12).point[2],50-Math.sqrt(2500-36),1e-12,'Independent analytic sag');
vector(intersectSurface([2,3,-10],[0,0,1],0,null,12).point,[2,3,0],1e-14,'Planar surface');
for(const r of reference.scalar) {
  const result=pupilRay(r.height,0,r.field,makeLens({index:r.index}),r.image_z);
  assert.equal(result.status,'ok');
  result.points.forEach((p,i)=>vector(p,r.points[i],2e-10,'Independent bisection/scalar-Snell ray'));
  vector(result.direction,r.direction,2e-12,'Independent scalar transmitted direction');
}
for(const r of reference.bundles) {
  const b=traceBundle(prescription(r.shape),r.aperture,r.field,r.image_z,r.rings,r.spokes);
  vector(b.centroid,r.centroid,2e-11,'Python/JS 3D centroid');close(b.rms_mm,r.rms_mm,2e-11,'Python/JS 3D RMS');
  assert.equal(b.accepted,r.accepted);
}
const lens=makeLens(),firstOrder=paraxial(lens);
// ABCD in reduced angle n*theta: matrix multiplication is independent of ray intersections.
const multiply=(a,b)=>[[a[0][0]*b[0][0]+a[0][1]*b[1][0],a[0][0]*b[0][1]+a[0][1]*b[1][1]],[a[1][0]*b[0][0]+a[1][1]*b[1][0],a[1][0]*b[0][1]+a[1][1]*b[1][1]]];
const M=multiply([[1,0],[-(1-lens.index)/lens.r2,1]],multiply([[1,lens.thickness/lens.index],[0,1]],[[1,0],[-(lens.index-1)/lens.r1,1]]));
close(firstOrder.focal,-1/M[1][0],1e-12,'Independent ABCD EFL');
const paraxialImage=lens.thickness-M[0][0]/M[1][0];close(firstOrder.image_z,paraxialImage,1e-12,'Independent ABCD back focus');
const errors=[.8,.4,.2,.1].map(h=>{const r=pupilRay(h,0,0,lens);return Math.abs(r.exit[2]-r.exit[0]*r.direction[2]/r.direction[0]-paraxialImage);});
errors.slice(1).forEach((e,i)=>assert.ok(e<errors[i]/3.8,'Second-order approach to the independent paraxial limit'));
const forward=pupilRay(4,2,6,lens),direction0=forward.surfaces[0].incoming;
let origin=forward.exit.map((p,i)=>p+3*forward.direction[i]),direction=forward.direction.map(v=>-v);
for(const [vertex,radius,ni,nt,expected] of [[4,lens.r2,1,lens.index,forward.surfaces[1].point],[0,lens.r1,lens.index,1,forward.surfaces[0].point]]) {
  const hit=intersectSurface(origin,direction,vertex,radius,12);assert.equal(hit.status,'ok');vector(hit.point,expected,1e-10,'Full 3D ray reversibility');
  direction=refract(direction,hit.normal,ni,nt).direction;origin=hit.point;
}
vector(direction,direction0.map(v=>-v),1e-12,'Recover initial 3D direction');
const r1=pupilRay(4,2,0,lens,53),r2=pupilRay(-2,4,0,lens,53);
vector(r2.image,[-r1.image[1],r1.image[0],53],1e-12,'Rotational symmetry of full 3D tracing');
const samples=pupilSamples(6,6,16);close(samples.reduce((s,p)=>s+dot(p,p),0)/samples.length,18,1e-12,'Equal-area pupil second moment');
for(const q of [-1.2,-1,-.4,0,.6,1,1.2]) {
  const l=prescription(q);close(paraxial(l).power,.02,1e-14,'Fixed thick-lens power');assert.ok(l.edge_thickness>=.5);
}
const refined=[3,6,12,24].map(k=>traceBundle(lens,6,6,53.32432432432432,k,4*k).rms_mm);
assert.ok(Math.abs(refined[2]/refined[3]-1)<.01,'Independent pupil quadrature refinement');
for(const r of reference.designs) {
  const d=optimizeDesign(0,6,r.fields,24),b=d.best;
  close(b.shape,r.shape,1e-10,'Python/JS optimized bending');close(b.image_z,r.image_z,1e-10,'Python/JS common detector');
  close(b.value_mm2,r.value_mm2,1e-12,'Python/JS optimized merit');
  assert.ok(b.valid&&b.value_mm2<d.start.value_mm2/4);assert.equal(b.aperture,6);
  assert.ok(b.bundles.every(v=>v.image_z===b.image_z&&v.throughput===1));
  d.history.slice(1).forEach((v,i)=>assert.ok(v.value_mm2<=d.history[i].value_mm2+1e-14));
  const h=.001,plus=designMerit(b.shape,b.image_z+h,6,r.fields),minus=designMerit(b.shape,b.image_z-h,6,r.fields);
  close((plus.value_mm2-minus.value_mm2)/(2*h),0,1e-9,'Stationarity at the common least-squares image plane');
  assert.ok(plus.value_mm2>=b.value_mm2&&minus.value_mm2>=b.value_mm2);
  const audit=designMerit(b.shape,b.image_z,6,r.fields,12,48),finer=designMerit(b.shape,b.image_z,6,r.fields,24,96);
  assert.ok(audit.valid&&finer.valid&&Math.abs(audit.value_mm2/finer.value_mm2-1)<.01);
}
const failed=designMerit(.5,52,13,[0,3,6]);assert.ok(!failed.valid&&failed.loss_fraction>0&&failed.value_mm2>=1e6,'Clipping cannot improve a merit');
assert.throws(()=>prescription(1.3));
for(const state of [{aperture:6,field:0,image:53.3,index:1.5},{aperture:1,field:8,image:30,index:1.8},{aperture:10,field:8,image:90,index:1.3}]){
  const result=rayModel(state);for(const p of result.plots)for(const s of p.series)assert.ok(s.y.length&&s.y.every(Number.isFinite));
}
for(const state of [{shape:0,aperture:6,objective:'field',iterations:24},{shape:-1.2,aperture:10,objective:'axis',iterations:0},{shape:1.2,aperture:1,objective:'field',iterations:8}]){
  const result=await designModel(state);for(const p of result.plots)for(const s of p.series)assert.ok(s.y.length&&s.y.every(Number.isFinite));
}
console.log(`PASS ${count} geometric comparisons: independent cap/Snell and ABCD references, 3D reversibility, pupil convergence, fixed-power constrained design, common-plane optimization and audit.`);
