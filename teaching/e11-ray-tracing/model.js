import {line,trace,linspace} from '../shared/math.js';
import {makeLens,paraxial,pupilRay,traceBundle,bestImagePlane,surfaceSag} from '../shared/ray-optics.js';

export function rayPathPlot(lens,aperture,field,imageZ) {
  const height=linspace(-lens.clear_radius,lens.clear_radius,81);
  const outlineZ=[...height.map(h=>surfaceSag(lens.r1,Math.abs(h))),...height.slice().reverse().map(h=>lens.thickness+surfaceSag(lens.r2,Math.abs(h)))];
  const outlineX=[...height,...height.slice().reverse()];outlineZ.push(outlineZ[0]);outlineX.push(outlineX[0]);
  const series=[trace('Lens surfaces',outlineZ,outlineX,'#66776a')];
  linspace(-aperture,aperture,13).forEach((h,i)=>{
    const r=pupilRay(h,0,field,lens,imageZ);
    series.push(trace('Exact rays',r.points.map(p=>p[2]),r.points.map(p=>p[0]),r.status==='ok'?'#267d85':'#b27931',{showLegend:i===0}));
  });
  series.push(trace('Selected image plane',[imageZ,imageZ],[-12,12],'#b27931',{dash:true}));
  return line('Computed meridional rays','Axial z from the first vertex (mm)','Transverse x (mm)',series,
    {xRange:[-10,Math.max(imageZ,8)],yRange:[-14,14],equalAspect:true,
     caption:'The entrance stop is fixed at z = −5 mm. Curved surfaces and ray paths use the same physical scale.'});
}

export function compute(s) {
  const lens=makeLens({index:s.index}),firstOrder=paraxial(lens),bundle=traceBundle(lens,s.aperture,s.field,s.image);
  const chief=pupilRay(0,0,s.field,lens,s.image),fan=linspace(-s.aperture,s.aperture,41).map(h=>pupilRay(h,0,s.field,lens,s.image)).filter(r=>r.status==='ok');
  const bestZ=bestImagePlane(lens,s.aperture,[s.field]);
  const scan=linspace(Math.max(20,firstOrder.image_z-6),firstOrder.image_z+3,49);
  const good=bundle.rays.filter(r=>r.status==='ok');
  const scanRMS=scan.map(z=>{
    const points=good.map(r=>[0,1].map(i=>r.exit[i]+(z-r.exit[2])*r.direction[i]/r.direction[2]));
    const centre=[0,1].map(i=>points.reduce((sum,p)=>sum+p[i],0)/points.length);
    return 1000*Math.sqrt(points.reduce((sum,p)=>sum+(p[0]-centre[0])**2+(p[1]-centre[1])**2,0)/points.length);
  });
  const xy=bundle.xy.map(p=>p.map((v,i)=>1000*(v-bundle.centroid[i]))),extent=Math.max(.5,...xy.flat().map(Math.abs))*1.1;
  const lost=bundle.requested-bundle.accepted;
  return {plots:[rayPathPlot(lens,s.aperture,s.field,s.image),
    line('Transverse ray fan at the selected plane','Entrance-stop x (mm)','Image x minus chief ray (μm)',[
      trace('Meridional ray error',fan.map(r=>r.pupil[0]),fan.map(r=>1000*(r.image[0]-chief.image[0])))],
      {caption:'The chief ray passes through the centre of the entrance stop. The fan shows geometric aberration and defocus.'}),
    line('Geometric spot relative to its centroid','Image x minus centroid (μm)','Image y minus centroid (μm)',[
      trace('Equal-area pupil rays',xy.map(p=>p[0]),xy.map(p=>p[1]),'#267d85',{points:true})],
      {xRange:[-extent,extent],yRange:[-extent,extent],equalAspect:true,caption:'Each dot is one geometric ray, not an intensity PSF. RMS radius is measured about this field’s centroid.'}),
    line('Focus scan for this field','Common image-plane z (mm)','Geometric RMS radius (μm)',[
      trace('Exact ray bundle',scan,scanRMS),
      trace('Paraxial on-axis plane',[firstOrder.image_z,firstOrder.image_z],[0,Math.max(...scanRMS)],'#b27931',{dash:true}),
      trace('Best geometric plane',[bestZ,bestZ],[0,Math.max(...scanRMS)],'#8a627d',{dash:true})],
      {caption:'The paraxial plane is a small-angle prediction. The best geometric plane depends on aperture and field.'})],
    metrics:[{label:'Paraxial effective focal length',value:firstOrder.focal.toFixed(3)+' mm',help:'Effective focal length is not the distance from the rear vertex.'},
      {label:'Paraxial / best image-plane z',value:firstOrder.image_z.toFixed(3)+' / '+bestZ.toFixed(3)+' mm',help:'Both positions are measured from the first surface vertex.'},
      {label:'Geometric RMS radius',value:(bundle.rms_mm*1000).toFixed(2)+' μm',help:'Centroid-relative radius of accepted rays; it does not include diffraction.'},
      {label:'Image centroid x / y',value:bundle.centroid[0].toFixed(3)+' / '+bundle.centroid[1].toFixed(3)+' mm',help:'Absolute image coordinates before the spot drawing is recentered.'},
      {label:'Accepted / launched rays',value:bundle.accepted+' / '+bundle.requested,help:'Blocked, missed, or totally internally reflected rays are counted as lost.'}],
    warning:lost>0,
    note:lost?'Some launched rays are lost. Inspect their count before interpreting the spot of the surviving rays.':'Exact means geometric intersections and Snell refraction. This monochromatic model omits diffraction, dispersion, Fresnel reflection, and polarization.',
    numerical:{rms_mm:bundle.rms_mm,accepted:bundle.accepted,requested:bundle.requested,paraxial:firstOrder,best_image_z:bestZ,centroid:bundle.centroid}};
}
