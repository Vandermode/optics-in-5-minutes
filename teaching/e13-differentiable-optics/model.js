import {line,trace} from '../shared/math.js';
import {optimizePhase,finiteDifferenceGradient} from '../shared/differentiable-optics.js';

export async function compute(s) {
  const result=optimizePhase(s),{start,best,history}=result;
  const fd=finiteDifferenceGradient(s),error=Math.max(...fd.map((v,i)=>Math.abs(v-start.gradient[i])));
  const gradientMin=Math.min(0,...fd,...start.gradient),gradientMax=Math.max(0,...fd,...start.gradient);
  const gradientPadding=Math.max(.01,(gradientMax-gradientMin)*.12);
  const pupil=start.x.map((x,i)=>({x,i})).filter(({i})=>start.amplitude[i]>0);
  const view=start.frequency.map((x,i)=>({x,i})).filter(({x})=>Math.abs(x)<=2);
  const series=(name,values,color)=>trace(name,view.map(p=>p.x),view.map(p=>values[p.i]),color);
  return {plots:[
    line('A fixed-power focal pattern','Spatial frequency (cycles / pupil half-width)','Fraction of total power per sample',[
      series('Target intensity',start.targetIntensity,'#a46841'),series('Initial intensity',start.intensity,'#8a8492'),series('Final intensity',best.intensity,'#267d85')],
      {yRange:[0,Math.max(...start.targetIntensity,...start.intensity,...best.intensity)*1.08],caption:'The curves share one power scale. The plot shows the central frequencies; the loss uses every detector sample.'}),
    line('Phase is the adjustable optical element','Normalized pupil position x','Phase (rad)',[
      trace('Initial phase',pupil.map(p=>p.x),pupil.map(p=>start.phase[p.i]),'#8a8492'),
      trace('Final phase',pupil.map(p=>p.x),pupil.map(p=>best.phase[p.i]),'#267d85')],
      {caption:'Only the illuminated pupil is drawn. The aperture and its amplitude remain fixed.'}),
    line('Two ways to check the same local derivative','Design parameter','Loss derivative (per radian)',[
      trace('Reverse-mode gradient',[0,1],start.gradient,'#267d85',{points:true,pointRadius:6}),
      trace('Central finite difference',[0,1],fd,'#a46841',{points:true,pointRadius:3})],
      {xRange:[-.5,1.5],xTicks:[{value:0,label:'Tilt a'},{value:1,label:'Defocus b'}],yRange:[gradientMin-gradientPadding,gradientMax+gradientPadding],caption:'Both derivatives are evaluated at the initial design, before any update. Agreement checks the derivative of this numerical model.'}),
    line('Recompute the loss after every update','Accepted gradient step','Target-normalized intensity loss L',[
      trace('Same target and objective',history.map(r=>r.iteration),history.map(r=>r.loss),'#267d85',{points:history.length===1})],
      {xRange:[0,Math.max(1,result.completed)],yRange:[0,Math.max(start.loss*1.05,1e-8)],caption:'Backtracking accepts only a decrease in this loss. A plateau may reflect a local stationary point; it is not a global-optimum certificate.'})],
    metrics:[{label:'Loss: initial → final',value:start.loss.toExponential(3)+' → '+best.loss.toExponential(3)},
      {label:'Gradient check: maximum absolute error',value:error.toExponential(2),help:'Central difference with h = 10⁻⁵ rad at the initial design.'},
      {label:'Final tilt a / defocus b',value:best.tilt.toFixed(4)+' / '+best.defocus.toFixed(4)+' rad'},
      {label:'Total detected / incident power',value:best.energy.toFixed(8),help:'The full discrete Fourier plane retains unit total power.'},
      {label:'Accepted steps',value:String(result.completed)}],
    warning:best.loss>.02&&s.iterations>0,
    note:'The model uses explicit reverse differentiation, not an autodiff library. Compare the same target and fixed power; a lower sampled loss does not establish uniqueness or fabricated performance.',
    numerical:{initial_loss:start.loss,final_loss:best.loss,gradient:start.gradient,finite_difference:fd,gradient_error:error,energy:best.energy,history:history.map(r=>r.loss),completed:result.completed}};
}
