// Sequential geometric rays. Lengths mm; fields degrees; vectors [x,y,z].
// R has centre z=vertex+R; null is planar. Entrance stop is fixed at z=-5 mm.
// Only the physical sphere cap is used. No diffraction or Fresnel power weights.
const dot = (a,b) => a.reduce((sum,x,i) => sum+x*b[i],0);
export function unit(v) {
  const length=Math.sqrt(dot(v,v));
  if(!Number.isFinite(length)||length<=0)throw new Error('A ray direction must have a finite positive length.');
  return v.map(x=>x/length);
}
export const curvature = r => r==null||!Number.isFinite(r)?0:1/r;
export function surfaceSag(radius,rho) {
  const c=curvature(radius),q=1-(c*rho)**2;
  return q< -1e-12?NaN:c*rho*rho/(1+Math.sqrt(Math.max(0,q)));
}
export function makeLens({r1=50,r2=-50,thickness=4,index=1.5,clear_radius=12}={}) {
  if(thickness<=0||index<=0||clear_radius<=0)throw new Error('Thickness, refractive index, and clear radius must be positive.');
  const radii=[r1,r2].map(r=>r==null||!Number.isFinite(r)?null:r);
  if(radii.includes(0))throw new Error('Zero radius is not a plane; use null or infinity.');
  return {r1:radii[0],r2:radii[1],thickness,index,clear_radius,edge_thickness:thickness+surfaceSag(radii[1],clear_radius)-surfaceSag(radii[0],clear_radius)};
}
export function paraxial(lens) {
  const n=lens.index,d=lens.thickness,c1=curvature(lens.r1),c2=curvature(lens.r2);
  const power=(n-1)*(c1-c2+(n-1)*d*c1*c2/n),focal=Math.abs(power)>1e-15?1/power:Infinity;
  const bfl=focal*(1-(n-1)*d*c1/n);
  return {power,focal,bfl,image_z:d+bfl};
}
export function prescription(shape=0,{focal=50,index=1.5,thickness=4,clear_radius=12}={}) {
  if(focal<=0||index<=1||Math.abs(shape)>1.2+1e-12)throw new Error('Use positive focal length, n>1, and -1.2 <= shape <= 1.2.');
  const b=1/(focal*(index-1)),a=(index-1)*thickness*(shape*shape-1)/(4*index),discriminant=1+4*a*b;
  if(discriminant<=0)throw new Error('No real prescription on the selected weak-curvature branch.');
  const delta=2*b/(1+Math.sqrt(discriminant)),c1=delta*(shape+1)/2,c2=delta*(shape-1)/2;
  return {...makeLens({r1:Math.abs(c1)<1e-15?null:1/c1,r2:Math.abs(c2)<1e-15?null:1/c2,thickness,index,clear_radius}),shape,target_focal:focal};
}
export function intersectSurface(origin,direction,vertex,radius,clearRadius=12) {
  direction=unit(direction);let roots;
  if(curvature(radius)===0)roots=Math.abs(direction[2])<1e-14?[]:[(vertex-origin[2])/direction[2]];
  else {
    const rel=[origin[0],origin[1],origin[2]-vertex-radius],b=2*dot(rel,direction),c=dot(rel,rel)-radius*radius,disc=b*b-4*c;
    if(disc< -1e-10)return {status:'missed'};
    const q=-.5*(b+(b>=0?1:-1)*Math.sqrt(Math.max(0,disc)));
    roots=Math.abs(q)>1e-15?[q,c/q]:[-b/2];
  }
  for(const distance of roots.sort((a,b)=>a-b)) {
    if(distance<=1e-9)continue;
    const point=origin.map((o,i)=>o+distance*direction[i]),rho=Math.hypot(point[0],point[1]),sag=surfaceSag(radius,rho);
    if(!Number.isFinite(sag)||Math.abs(point[2]-vertex-sag)>1e-7*Math.max(1,Math.abs(vertex),Math.abs(point[2])))continue;
    if(rho>clearRadius+1e-9)return {status:'blocked',point,distance};
    let normal=curvature(radius)===0?[0,0,1]:unit([point[0],point[1],point[2]-vertex-radius]);
    if(dot(direction,normal)<0)normal=normal.map(x=>-x);
    return {status:'ok',point,distance,normal};
  }
  return {status:'missed'};
}
export function refract(direction,normal,nBefore,nAfter) {
  const d=unit(direction);normal=unit(normal);
  if(dot(d,normal)<0)normal=normal.map(x=>-x);
  const cosine=Math.max(0,Math.min(1,dot(d,normal))),eta=nBefore/nAfter,tangent=d.map((v,i)=>v-cosine*normal[i]);
  const radicand=1-eta*eta*dot(tangent,tangent);
  if(radicand< -1e-12)return {status:'tir',normal,cos_incident:cosine};
  const transmitted=Math.sqrt(Math.max(0,radicand));
  return {status:'ok',direction:unit(tangent.map((v,i)=>eta*v+transmitted*normal[i])),normal,cos_incident:cosine,cos_transmitted:transmitted};
}
export function traceRay(origin,direction,lens,imageZ=null) {
  let point=Array.from(origin);direction=unit(direction);const points=[point.slice()],surfaces=[];
  const sequence=[[0,lens.r1,1,lens.index],[lens.thickness,lens.r2,lens.index,1]];
  for(let j=0;j<sequence.length;j++) {
    const [vertex,radius,ni,nt]=sequence[j],hit=intersectSurface(point,direction,vertex,radius,lens.clear_radius);
    hit.surface=j+1;surfaces.push(hit);if(hit.point)points.push(hit.point);
    if(hit.status!=='ok')return {status:hit.status,points,surfaces,failed_surface:j+1,exit:null,direction:null,image:null};
    const bent=refract(direction,hit.normal,ni,nt);
    Object.assign(hit,{n_before:ni,n_after:nt,incoming:direction,outgoing:bent.direction??null});
    if(bent.status!=='ok')return {status:bent.status,points,surfaces,failed_surface:j+1,exit:null,direction:null,image:null};
    point=hit.point;direction=bent.direction;
  }
  let image=null;
  if(imageZ!=null) {
    const t=Math.abs(direction[2])>1e-14?(imageZ-point[2])/direction[2]:-1;
    if(t<0)return {status:'image_behind_lens',points,surfaces,exit:point,direction,image:null};
    image=point.map((p,i)=>p+t*direction[i]);points.push(image);
  }
  return {status:'ok',points,surfaces,exit:point,direction,image};
}
export function pupilRay(x,y,fieldDeg,lens,imageZ=null,launchZ=-10,pupilZ=-5) {
  const theta=fieldDeg*Math.PI/180,direction=[Math.sin(theta),0,Math.cos(theta)];
  const origin=[x+(launchZ-pupilZ)*direction[0]/direction[2],y,launchZ];
  return {...traceRay(origin,direction,lens,imageZ),pupil:[x,y]};
}
export function pupilSamples(aperture,rings=6,spokes=16) {
  if(aperture<=0||rings<1||spokes<4)throw new Error('Use a positive pupil and at least one ring/four spokes.');
  const out=[];
  for(let j=0;j<rings;j++)for(let k=0;k<spokes;k++) {
    const r=aperture*Math.sqrt((j+.5)/rings),theta=2*Math.PI*(k+.5*(j%2))/spokes;
    out.push([r*Math.cos(theta),r*Math.sin(theta)]);
  }
  return out;
}
export function traceBundle(lens,aperture,fieldDeg,imageZ,rings=6,spokes=16) {
  const rays=pupilSamples(aperture,rings,spokes).map(([x,y])=>pupilRay(x,y,fieldDeg,lens,imageZ));
  const accepted=rays.filter(r=>r.status==='ok'),xy=accepted.filter(r=>r.image!=null).map(r=>r.image.slice(0,2));
  const centroid=xy.length?[0,1].map(i=>xy.reduce((sum,p)=>sum+p[i],0)/xy.length):null;
  const rms=xy.length?Math.sqrt(xy.reduce((sum,p)=>sum+(p[0]-centroid[0])**2+(p[1]-centroid[1])**2,0)/xy.length):null;
  return {rays,xy,centroid,rms_mm:rms,requested:rays.length,accepted:accepted.length,throughput:accepted.length/rays.length,field_deg:fieldDeg,image_z:imageZ};
}
export function bestImagePlane(lens,aperture,fields=[0,3,6],rings=6,spokes=16,bounds=[30,90]) {
  let quadratic=0,linear=0;
  for(const field of fields) {
    const accepted=traceBundle(lens,aperture,field,null,rings,spokes).rays.filter(r=>r.status==='ok'&&r.direction[2]>0);
    if(!accepted.length)return Math.max(bounds[0],Math.min(bounds[1],paraxial(lens).image_z));
    for(const axis of [0,1]) {
      const slopes=accepted.map(r=>r.direction[axis]/r.direction[2]),intercepts=accepted.map((r,i)=>r.exit[axis]-r.exit[2]*slopes[i]);
      const bm=slopes.reduce((s,v)=>s+v,0)/slopes.length,am=intercepts.reduce((s,v)=>s+v,0)/intercepts.length;
      quadratic+=slopes.reduce((s,b)=>s+(b-bm)**2,0)/slopes.length;
      linear+=intercepts.reduce((s,a,i)=>s+(a-am)*(slopes[i]-bm),0)/slopes.length;
    }
  }
  return quadratic>1e-20?Math.max(bounds[0],Math.min(bounds[1],-linear/quadratic)):bounds[1];
}
export function designMerit(shape,imageZ,aperture=6,fields=[0,3,6],rings=6,spokes=16) {
  const lens=prescription(shape),bundles=fields.map(f=>traceBundle(lens,aperture,f,imageZ,rings,spokes));
  const loss=bundles.reduce((s,b)=>s+1-b.throughput,0)/bundles.length;
  const valid=Number.isFinite(lens.edge_thickness)&&lens.edge_thickness>=.5&&loss===0;
  const rms2=bundles.reduce((s,b)=>s+(b.rms_mm||0)**2,0)/bundles.length;
  const edgePenalty=Number.isFinite(lens.edge_thickness)?Math.max(0,.5-lens.edge_thickness):1e3;
  return {value_mm2:valid?rms2:1e6+1e3*loss+1e3*edgePenalty,valid,loss_fraction:loss,bundles,lens,shape,image_z:imageZ,rms_mm:Math.sqrt(rms2),aperture,fields:Array.from(fields)};
}
export function optimizeDesign(shape=0,aperture=6,fields=[0,3,6],iterations=24,rings=6,spokes=16) {
  const lens=prescription(shape),start=designMerit(shape,paraxial(lens).image_z,aperture,fields,rings,spokes);
  let best=start,step=.4,trials=0;
  const row=b=>({shape:b.shape,image_z:b.image_z,value_mm2:b.value_mm2}),history=[row(best)];
  for(let iteration=0;iteration<iterations;iteration++) {
    const candidates=iteration===0?[best.shape]:[Math.max(-1.2,best.shape-step),Math.min(1.2,best.shape+step)];
    let improved=false;
    for(const q of candidates) {
      const trialLens=prescription(q),z=bestImagePlane(trialLens,aperture,fields,rings,spokes),result=designMerit(q,z,aperture,fields,rings,spokes);trials++;
      if(result.value_mm2<best.value_mm2-1e-15){best=result;improved=true;}
    }
    if(!improved&&iteration>0)step/=2;
    history.push(row(best));if(step<1e-5)break;
  }
  return {start,best,history,trials,completed:history.length-1};
}
