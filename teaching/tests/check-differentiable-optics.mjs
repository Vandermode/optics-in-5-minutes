import assert from 'node:assert/strict';
import {fourier,forwardIntensity,evaluateDesign,finiteDifferenceGradient,optimizePhase} from '../shared/differentiable-optics.js';
let checks=0;
function close(a,b,tolerance,label) {
  assert.ok(Number.isFinite(a)&&Number.isFinite(b)&&Math.abs(a-b)<=tolerance,`${label}: ${a} vs ${b}`);checks++;
}
function vector(a,b,tolerance,label) {assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i],tolerance,`${label} ${i}`));}
// Independent reference: NumPy's unitary centered FFT in E13 lab.py. The browser
// engine instead evaluates a dense DFT. These values lock normalization and axes.
const references=[
  {parameters:{tilt:1.5,defocus:2,target:0,n:128},loss:.2605818922015634,gradient:[.21889562490235487,-.04178200304057808],indices:[56,62,64,66,72],intensity:[.006761069984730808,.03328457196571879,.09087348479449764,.17363463247281755,.022243996927211497]},
  {parameters:{tilt:-1.2,defocus:-.7,target:1.5,n:128},loss:.7190966048573342,gradient:[-.22039514643525698,.12389539249146872],indices:[56,62,64,66,72],intensity:[.009520726714154645,.22873786241280702,.14581835248813907,.0087853214608089,.004270998806821557]},
  {parameters:{tilt:3,defocus:3,target:0,n:64},loss:.36500570498181395,gradient:[.05945122189739785,-.11519422819178429],indices:[24,30,32,34,40],intensity:[.001964088503771714,.03775708488512863,.05721920160900556,.07398434844728068,.057458143195674515]},
];
for(const ref of references) {
  const result=evaluateDesign(ref.parameters);
  close(result.loss,ref.loss,5e-14,'Independent NumPy FFT loss');
  vector(result.gradient,ref.gradient,5e-14,'Independent NumPy gradient');
  vector(ref.indices.map(i=>result.intensity[i]),ref.intensity,5e-14,'Independent NumPy intensity');
}
for(const n of [32,64,128]) {
  const result=evaluateDesign({tilt:0,defocus:0,n}),m=n/4;
  close(result.loss,0,1e-30,'Reachable zero-loss target');
  close(result.energy,1,2e-14,'Parseval power');
  // Finite uniform aperture: discrete geometric series, independent of any DFT.
  for(let k=0;k<n;k++) {
    const q=k-n/2;
    const expected=q===0?m/n:Math.sin(Math.PI*m*q/n)**2/(n*m*Math.sin(Math.PI*q/n)**2);
    close(result.intensity[k],expected,1e-14,'Analytic aperture diffraction');
  }
  const pupil=forwardIntensity({tilt:1.1,defocus:-.7,n}).pupil;
  const out=fourier(pupil.real,pupil.imaginary),back=fourier(out.real,out.imaginary,true);
  vector(Array.from(back.real),pupil.real,1e-14,'Adjoint propagation reconstructs real field');
  vector(Array.from(back.imaginary),pupil.imaginary,1e-14,'Adjoint propagation reconstructs imaginary field');
  // A linear phase 2π*x/8 is exactly one sensor-bin translation.
  const shifted=forwardIntensity({tilt:2*Math.PI/8,defocus:0,n}).intensity;
  vector(shifted,result.intensity.map((_,i)=>result.intensity[(i-1+n)%n]),1e-14,'Fourier shift theorem');
}
for(const state of [{tilt:1.5,defocus:2,target:0},{tilt:-2.3,defocus:-1.1,target:1},
  {tilt:.2,defocus:0,target:-2},{tilt:4,defocus:4,target:2},{tilt:0,defocus:0,target:0}]) {
  const result=evaluateDesign(state);
  close(result.energy,1,2e-14,'Phase-only power conservation');
  vector(result.gradient,finiteDifferenceGradient(state,1e-5),2e-9,'Complete-chain finite differences');
  close(result.phaseGradient.reduce((s,v)=>s+v,0),0,2e-14,'Global-phase derivative is zero');
  vector(forwardIntensity({...state,piston:.71}).intensity,result.intensity,1e-14,'Global phase cannot change intensity');
}
const coarse=evaluateDesign(),fine=evaluateDesign({n:256});
close(fine.loss,coarse.loss,5e-4,'Fixed-aperture grid refinement loss');
vector(fine.gradient,coarse.gradient,5e-4,'Fixed-aperture grid refinement gradient');
for(const state of [{tilt:1.5,defocus:2,target:0},{tilt:1.5,defocus:-2,target:0},{tilt:1.5,defocus:2,target:1}]) {
  const result=optimizePhase(state);
  assert.ok(result.best.loss<result.start.loss/1000,'Default neighborhood improves the actual objective');
  result.history.slice(1).forEach((row,i)=>assert.ok(row.loss<=result.history[i].loss,'Accepted losses are monotone'));
  close(evaluateDesign({...state,tilt:result.best.tilt,defocus:result.best.defocus}).loss,result.best.loss,1e-15,'Reported parameters reproduce loss');
}
const trapped=optimizePhase({tilt:3,defocus:3});
assert.ok(trapped.best.loss>.1,'Alternate start exposes a poor local stationary design');
assert.ok(evaluateDesign({tilt:0,defocus:0}).loss<trapped.best.loss,'Known reachable reference beats the trapped design');
assert.equal(optimizePhase({iterations:0}).completed,0);
const q=forwardIntensity({tilt:.83,defocus:.71,levels:4}).intensity;
vector(q,forwardIntensity({tilt:.83+1e-7,defocus:.71,levels:4}).intensity,0,'Hard quantization is locally constant away from thresholds');
assert.throws(()=>evaluateDesign({tilt:NaN}));assert.throws(()=>evaluateDesign({n:127}));assert.throws(()=>optimizePhase({iterations:-1}));
console.log(`PASS ${checks} differentiable-optics comparisons: independent NumPy FFT and aperture diffraction, shift and piston invariance, complete gradient checks, power/grid audits, accepted design updates and quantization limits.`);
