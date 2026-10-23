import {line,trace,linspace} from '../shared/math.js';
import {optimizeDesign,traceBundle,paraxial,surfaceSag} from '../shared/ray-optics.js';
import {rayPathPlot} from '../e11-ray-tracing/model.js';

export async function compute(s) {
  const fields=s.objective==='axis'?[0]:[0,3,6];
  const result=optimizeDesign(s.shape,s.aperture,fields,s.iterations),{start,best}=result;
  // The audit has more pupil samples and includes fields unused by the optimizer.
  const auditFields=[0,1.5,3,4.5,6],audit=(design)=>auditFields.map(f=>traceBundle(design.lens,s.aperture,f,design.image_z,10,32));
  const before=audit(start),after=audit(best),centred=b=>b.xy.map(p=>p.map((v,i)=>1000*(v-b.centroid[i])));
  const original=centred(before.at(-1)),optimized=centred(after.at(-1));
  const extent=Math.max(1,...original.flat().map(Math.abs),...optimized.flat().map(Math.abs))*1.1;
  const height=linspace(-12,12,101),profile=lens=>({z:[...height.map(h=>surfaceSag(lens.r1,Math.abs(h))),...height.slice().reverse().map(h=>lens.thickness+surfaceSag(lens.r2,Math.abs(h)))],x:[...height,...height.slice().reverse()]});
  const p0=profile(start.lens),p1=profile(best.lens);
  const lost=after.reduce((sum,b)=>sum+b.requested-b.accepted,0);
  return {plots:[rayPathPlot(best.lens,s.aperture,6,best.image_z),
    line('Start and optimized geometric spots at 6°','Image x minus centroid (μm)','Image y minus centroid (μm)',[
      trace('Starting prescription',original.map(p=>p[0]),original.map(p=>p[1]),'#b27931',{points:true}),
      trace('Optimized prescription',optimized.map(p=>p[0]),optimized.map(p=>p[1]),'#267d85',{points:true})],
      {xRange:[-extent,extent],yRange:[-extent,extent],equalAspect:true,caption:'Both spots use the same aperture and scale. Each prescription has one image plane shared by every field; centroid removal only recentres the drawing.'}),
    line('Field audit on one common plane','Object field angle (degrees)','Geometric RMS radius (μm)',[
      trace('Starting prescription',auditFields,before.map(b=>1000*b.rms_mm),'#b27931'),
      trace('Optimized prescription',auditFields,after.map(b=>1000*b.rms_mm))],
      {caption:'The audit uses 320 pupil rays per field. The 1.5° and 4.5° fields were not used in the design merit.'}),
    line('Best design after each search step','Search iteration','Equal-field mean RMS² (mm²)',[
      trace('Fixed-objective merit',result.history.map((_,i)=>i),result.history.map(r=>r.value_mm2))],
      {caption:'The search changes bending and a common image plane. Aperture, thickness, index, mechanical radius, field weights, and paraxial focal power remain fixed.'}),
    line('Fixed-power lens prescription','Axial z from the first vertex (mm)','Transverse x (mm)',[
      trace('Starting prescription',p0.z,p0.x,'#b27931'),trace('Optimized prescription',p1.z,p1.x)],
      {equalAspect:true,caption:'Changing both curvatures redistributes aberration while preserving the 50 mm paraxial effective focal length.'})],
    metrics:[{label:'Bending q: start → optimized',value:start.shape.toFixed(3)+' → '+best.shape.toFixed(3)},
      {label:'Common image-plane z',value:best.image_z.toFixed(3)+' mm',help:'Every evaluated field uses this same physical detector plane.'},
      {label:'Merit RMS: start → optimized',value:(start.rms_mm*1000).toFixed(2)+' → '+(best.rms_mm*1000).toFixed(2)+' μm',help:'Square root of the selected equal-field mean squared spot radius.'},
      {label:'Effective focal length / edge thickness',value:paraxial(best.lens).focal.toFixed(3)+' / '+best.lens.edge_thickness.toFixed(3)+' mm',help:'The 24 mm mechanical diameter and minimum 0.5 mm edge thickness are fixed constraints.'},
      {label:'Audit lost rays',value:String(lost),help:'Any lost ray invalidates a candidate and receives a large merit penalty.'}],
    warning:!best.valid||lost>0,
    note:s.objective==='axis'?'An on-axis objective can trade away off-axis quality. Inspect the separate field audit before choosing this prescription.':'This bounded search finds a candidate for one stated geometric merit. It does not establish a global optimum or predict diffraction-limited, broadband, or fabricated performance.',
    numerical:{history:result.history.map(r=>r.value_mm2),shape:best.shape,image_z:best.image_z,merit_mm2:best.value_mm2,start_merit_mm2:start.value_mm2,
      fields,edge_thickness:best.lens.edge_thickness,focal:paraxial(best.lens).focal,loss_fraction:best.loss_fraction,
      audit_fields:auditFields,audit_rms_mm:after.map(b=>b.rms_mm),audit_lost:lost}};
}
