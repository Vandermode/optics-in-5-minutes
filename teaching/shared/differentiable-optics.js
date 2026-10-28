// Fixed 1D phase-only aperture and real-valued intensity loss.
// All amplitudes have unit summed power. Phase coefficients are in radians;
// pupil x and image frequency (cycles per pupil half-width) are dimensionless.
const cache=new Map();
function grid(n) {
  if(!Number.isInteger(n)||n<16||n>512||n%8!==0)throw new Error('Use a grid size divisible by eight, between 16 and 512.');
  if(cache.has(n))return cache.get(n);
  const x=Array.from({length:n},(_,j)=>8*(j-(n-1)/2)/n),frequency=Array.from({length:n},(_,k)=>(k-n/2)/8);
  const amplitude=x.map(v=>Math.abs(v)<1?2/Math.sqrt(n):0),re=new Float64Array(n*n),im=new Float64Array(n*n);
  for(let k=0;k<n;k++)for(let j=0;j<n;j++) {
    const angle=-2*Math.PI*(j-n/2)*(k-n/2)/n;
    re[k*n+j]=Math.cos(angle)/Math.sqrt(n);im[k*n+j]=Math.sin(angle)/Math.sqrt(n);
  }
  const value={x,frequency,amplitude,re,im};cache.set(n,value);return value;
}
export function fourier(real,imaginary,inverse=false) {
  const n=real.length,{re,im}=grid(n);
  if(imaginary.length!==n)throw new Error('Complex arrays must have equal lengths.');
  const outR=new Float64Array(n),outI=new Float64Array(n);
  for(let k=0;k<n;k++)for(let j=0;j<n;j++) {
    const at=inverse?j*n+k:k*n+j,a=re[at],b=inverse?-im[at]:im[at];
    outR[k]+=a*real[j]-b*imaginary[j];outI[k]+=a*imaginary[j]+b*real[j];
  }
  return {real:outR,imaginary:outI};
}
function pupilField(tilt,defocus,n,levels=null,piston=0) {
  const {x,amplitude}=grid(n),phase=x.map(v=>tilt*v+defocus*v*v+piston);
  if(levels!==null) {
    if(!Number.isInteger(levels)||levels<2)throw new Error('Phase levels must be an integer of at least two.');
    const step=2*Math.PI/levels;
    for(let j=0;j<n;j++)phase[j]=step*Math.round(phase[j]/step);
  }
  const real=phase.map((v,j)=>amplitude[j]*Math.cos(v)),imaginary=phase.map((v,j)=>amplitude[j]*Math.sin(v));
  return {real,imaginary,phase};
}
export function forwardIntensity({tilt=1.5,defocus=2,n=128,levels=null,piston=0}={}) {
  if(![tilt,defocus,piston].every(Number.isFinite))throw new Error('Phase parameters must be finite.');
  const pupil=pupilField(tilt,defocus,n,levels,piston),field=fourier(pupil.real,pupil.imaginary);
  return {pupil,field,intensity:Array.from(field.real,(v,k)=>v*v+field.imaginary[k]**2)};
}
export function evaluateDesign({tilt=1.5,defocus=2,target=0,n=128}={}) {
  if(!Number.isFinite(target))throw new Error('Target tilt must be finite.');
  const {x,frequency,amplitude}=grid(n),{pupil,field,intensity}=forwardIntensity({tilt,defocus,n});
  const targetIntensity=forwardIntensity({tilt:target,defocus:0,n}).intensity;
  const scale=targetIntensity.reduce((sum,v)=>sum+v*v,0),residual=intensity.map((v,k)=>v-targetIntensity[k]);
  const loss=.5*residual.reduce((sum,v)=>sum+v*v,0)/scale;
  const back=fourier(residual.map((v,k)=>v*field.real[k]/scale),residual.map((v,k)=>v*field.imaginary[k]/scale),true);
  const phaseGradient=x.map((_,j)=>2*(pupil.real[j]*back.imaginary[j]-pupil.imaginary[j]*back.real[j]));
  const gradient=[1,2].map(power=>phaseGradient.reduce((sum,v,j)=>sum+v*x[j]**power,0));
  return {tilt,defocus,target,n,x:[...x],frequency:[...frequency],amplitude:[...amplitude],phase:pupil.phase,
    intensity,targetIntensity,loss,gradient,phaseGradient,energy:intensity.reduce((sum,v)=>sum+v,0)};
}
export function finiteDifferenceGradient(parameters={},h=1e-5) {
  if(!Number.isFinite(h)||h<=0)throw new Error('The difference step must be positive.');
  const p={tilt:1.5,defocus:2,target:0,n:128,...parameters};
  return ['tilt','defocus'].map(key=>(evaluateDesign({...p,[key]:p[key]+h}).loss-evaluateDesign({...p,[key]:p[key]-h}).loss)/(2*h));
}
export function optimizePhase({tilt=1.5,defocus=2,target=0,n=128,iterations=40,learningRate=12}={}) {
  if(!Number.isInteger(iterations)||iterations<0||iterations>300)throw new Error('Use zero to 300 iterations.');
  if(!Number.isFinite(learningRate)||learningRate<=0)throw new Error('The initial step size must be positive.');
  const start=evaluateDesign({tilt,defocus,target,n});let best=start,trials=0,stopped='iteration limit';
  const row=(value,iteration,step)=>({iteration,tilt:value.tilt,defocus:value.defocus,loss:value.loss,gradient:[...value.gradient],step});
  const history=[row(start,0,0)];
  for(let k=0;k<iterations;k++) {
    const norm2=best.gradient.reduce((sum,v)=>sum+v*v,0);
    if(norm2<1e-22){stopped='small gradient';break;}
    let step=learningRate,accepted=false;
    for(let trial=0;trial<24;trial++) {
      const candidate=evaluateDesign({tilt:best.tilt-step*best.gradient[0],defocus:best.defocus-step*best.gradient[1],target,n});trials++;
      if(candidate.loss<=best.loss-1e-4*step*norm2) {best=candidate;accepted=true;break;}
      step/=2;
    }
    if(!accepted){stopped='line search stalled';break;}
    history.push(row(best,k+1,step));
  }
  return {start,best,history,completed:history.length-1,trials,stopped};
}
