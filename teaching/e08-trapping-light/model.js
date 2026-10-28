import {linspace,line,trace} from '../shared/math.js';
export function slabModes(width,n1=1.5,n2=1.45,wavelength=1.55){const a=width/2,k=2*Math.PI/wavelength,V=k*a*Math.sqrt(n1*n1-n2*n2),modes=[];for(let m=0;m<12;m++){let lo=m*Math.PI/2+1e-9,hi=Math.min((m+1)*Math.PI/2-1e-9,V-1e-9);if(lo>=hi)break;const f=u=>(m%2?-u/Math.tan(u):u*Math.tan(u))-Math.sqrt(Math.max(0,V*V-u*u));if(f(lo)*f(hi)>0)continue;for(let i=0;i<70;i++){const mid=(lo+hi)/2;if(f(mid)>0)hi=mid;else lo=mid;}const u=(lo+hi)/2,w=Math.sqrt(V*V-u*u),neff=Math.sqrt(n1*n1-(u/(k*a))**2);modes.push({u,w,neff,odd:!!(m%2),a});}return modes;}
export function ring(phi,r,a){const den=1+(r*a)**2-2*r*a*Math.cos(phi);return {through:(r*r+a*a-2*r*a*Math.cos(phi))/den,circulating:(1-r*r)/den};}
export function compute(s) {
  const modes=slabModes(s.width),index=s.mode,m=modes[index],x=linspace(-8,8,401);
  // Keep the requested index when a mode crosses cutoff. Substituting a lower
  // mode would make the prescribed disappearance experiment show the wrong state.
  const field=m?x.map(v=>{
    if(Math.abs(v)<=m.a)return m.odd?Math.sin(m.u*v/m.a):Math.cos(m.u*v/m.a);
    const edge=m.odd?Math.sin(m.u)*Math.sign(v):Math.cos(m.u);
    return edge*Math.exp(-m.w*(Math.abs(v)-m.a)/m.a);
  }):null;
  const unavailable=`Requested TE mode ${index} is not guided at this thickness. Only ${modes.length} guided modes exist, indexed from 0. Increase the thickness or select a lower index; no replacement mode is drawn.`;
  const modePlot=line(m?'Discrete guided modes have decaying tails':'Requested mode is below cutoff',
    'Transverse position (μm)','Mode amplitude (arbitrary)',m?[trace('TE mode '+index,x,field)]:[],
    {xRange:[-8,8],...(m?{}:{yRange:[-1,1]}),caption:m?
      'Symmetric slab: ncore = 1.50, ncladding = 1.45, λ₀ = 1.55 μm. Profiles are amplitude-normalized.':unavailable});
  const z=linspace(0,150,301),p=linspace(-Math.PI,Math.PI,401),response=p.map(v=>ring(v,s.coupling,s.loss)),q=s.coupling*s.loss;
  return {plots:[modePlot,
    line('Two-mode coupling conserves total power','Propagation distance (μm)','Power / input',[
      trace('Guide 1',z,z.map(v=>Math.cos(s.kappa*v)**2)),
      trace('Guide 2',z,z.map(v=>Math.sin(s.kappa*v)**2),'#b27931')],{yRange:[0,1.05]}),
    line('Interference sets the through-port dip','Round-trip phase detuning (rad)','Through power / input power',[
      trace('Through port',p,response.map(v=>v.through))],
      {yRange:[0,1.05],caption:'The fixed 0–1 power scale keeps the transmission dip visible. At resonance it reaches zero when r = a.'}),
    line('Repeated round trips build circulating power','Round-trip phase detuning (rad)','Circulating power / input power',[
      trace('Circulating',p,response.map(v=>v.circulating),'#aa5378')],
      {caption:'This separate power scale can exceed one: energy circulates for multiple round trips. It does not represent net output power or gain.'}),
    line('Storage has a finite lifetime','Number of round trips','Remaining stored energy',[
      trace('(r a)²ᴺ',linspace(0,40,161),linspace(0,40,161).map(v=>q**(2*v)))],{yRange:[0,1.05]})],
    metrics:[{label:'Guided modes / selected neff',value:m?modes.length+' / '+m.neff.toFixed(5):modes.length+' / —',
      help:m?'Each guided mode lies between the cladding and core indices.':unavailable},
      {label:'Complete-transfer length',value:(Math.PI/(2*s.kappa)).toFixed(1)+' μm',help:'κ is an independent two-mode model parameter in this panel.'},
      {label:'On-resonance through power',value:(100*ring(0,s.coupling,s.loss).through).toFixed(2)+'%',help:'Critical coupling occurs when amplitude self-coupling r equals round-trip amplitude survival a.'}],
    note:m?'The slab, coupler, and ring are separate reduced models. The notebook derives the coupler splitting from a two-guide eigenproblem.':unavailable,
    numerical:{requested_mode:index,guided_count:modes.length,selected_neff:m?.neff??null}};
}
