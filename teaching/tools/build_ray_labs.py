"""Embed the canonical ray engine into self-contained, staged E11/E12 lab sources.

Run after changing ray_optics.py and before build_course.py. No notebook downloads
or runtime repository imports are needed. The source cells remain ordinary Python.
"""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "tools/ray_optics.py").read_text()
FUNCTIONS = {node.name: ast.get_source_segment(SOURCE, node) for node in ast.parse(SOURCE).body if isinstance(node, ast.FunctionDef)}
IMPORT = "from math import acos, cos, copysign, hypot, inf, isfinite, pi, sin, sqrt\n"


def functions(*names):
    return "\n\n".join(FUNCTIONS[name] for name in names)


E11 = r'''# %% [markdown]
# ## 1 · One ray, one interface
#
# A ray is a position and a unit direction, not a wave amplitude. Throughout this lab,
# vectors are **(x, y, z)**, lengths are **mm**, and z increases to the right in ray drawings.
# A surface normal points into the transmitted medium. Split the incident unit vector
# into normal and tangential parts: Snell's law scales the tangential part by n₁/n₂.
# The new normal component makes the result a unit vector. A negative radicand means
# total internal reflection, not a transmitted ray with a complex plotting coordinate.
#
# **Predict:** from air into glass, does a 30° ray move toward or away from the normal?
# %%
@@SNELL@@

incoming = [sin(pi/6), 0., cos(pi/6)]
answer = refract(incoming, [0.,0.,1.], 1., 1.5)
theta_out = acos(answer["direction"][2])
assert abs(sin(pi/6)-1.5*sin(theta_out)) < 1e-13
assert refract([0.,0.,1.],[0.,0.,1.],1.,1.5)["direction"] == [0.,0.,1.]
assert refract([sin(pi/3),0.,cos(pi/3)],[0.,0.,1.],1.5,1.)["status"] == "tir"
reverse = refract([-v for v in answer["direction"]], [0.,0.,-1.], 1.5, 1.)
assert np.max(abs(np.array(reverse["direction"])+incoming)) < 1e-13
fig, ax = plt.subplots(figsize=(6,4))
ax.axvline(0,color="gray",label="Interface");ax.axhline(0,color="gray",ls="--",label="Normal")
ax.plot([-incoming[2],0],[-incoming[0],0],label="Incident: 30°")
ax.plot([0,answer["direction"][2]],[0,answer["direction"][0]],label=f"Transmitted: {np.rad2deg(theta_out):.2f}°")
ax.set(xlabel="z (drawing units)",ylabel="x (drawing units)",title="Vector Snell law · air to glass")
ax.set_aspect("equal",adjustable="box");ax.legend();show(fig,"01-one-interface")
print("Normal incidence, scalar Snell law, TIR, and ray reversal checks passed.")
# %% [markdown]
# ## 2 · A sphere is not the same as its optical surface
#
# A radius R has its centre at z = vertex + R. The surface is the spherical **cap
# containing the vertex**. A line can meet the complete sphere twice: accepting any
# positive root without checking the cap can send a ray to the far hemisphere.
# At pupil radius ρ, the physical sag is R − sign(R)√(R²−ρ²).
#
# **Predict:** at x = 6 mm on a surface with R = +50 mm, is the first surface point
# near z = 0, or near z = 100 mm? The intersection routine also checks the clear radius.
# %%
@@SURFACE@@

hit = intersect_surface([6.,0.,-10.],[0.,0.,1.],0.,50.,12.)
analytic_sag = 50.-np.sqrt(50.**2-6.**2)
assert hit["status"] == "ok" and abs(hit["point"][2]-analytic_sag) < 1e-12
assert intersect_surface([13.,0.,-10.],[0.,0.,1.],0.,50.,12.)["status"] == "blocked"
assert intersect_surface([0.,0.,-10.],[0.,0.,1.],4.,-50.,12.)["point"][2] == 4.
height = np.linspace(-12,12,121)
fig, ax = plt.subplots(figsize=(7,4))
ax.plot([surface_sag(50.,abs(x)) for x in height],height,label="Physical cap, R = +50 mm")
ax.plot([-10,hit["point"][2]],[6,6],label="Incoming ray")
ax.scatter([hit["point"][2]],[6],color="tab:red",label="Accepted intersection")
ax.set(xlabel="z from vertex (mm)",ylabel="x (mm)",title="Cap selection before refraction",xlim=(-12,12),ylim=(-14,14))
ax.set_aspect("equal",adjustable="box");ax.legend(loc="upper center",bbox_to_anchor=(.5,-.2),fontsize=9);show(fig,"02-physical-cap")
print("Near-cap sag:",analytic_sag,"mm; rejected far-hemisphere root:",50+np.sqrt(50**2-6**2),"mm")
# %% [markdown]
# ## 3 · Trace a pupil through two surfaces
#
# The lens has R₁ = +50 mm, R₂ = −50 mm, thickness 4 mm and refractive index 1.5,
# surrounded by air. Its mechanical clear radius is 12 mm. The entrance stop stays
# at z = −5 mm in air. **Aperture** below is that stop's radius, not the lens diameter.
# Each ray follows intersection → refraction → intersection → refraction → detector.
# Pupil samples use equal-area annuli, so their equal weights approximate uniform illumination.
#
# The paraxial thick-lens power predicts an image plane for an on-axis object at infinity.
# An exact geometric bundle can cross at different z positions. A small RMS radius is
# a statement about geometric rays; it is not an intensity PSF or an MTF.
# **Predict:** what changes if you halve the entrance-stop radius but keep the detector fixed?
# %%
@@BUNDLE@@

lens = make_lens()
aperture, field = 6., 0.
first_order = paraxial(lens)
image_z = first_order["image_z"]
bundle = trace_bundle(lens,aperture,field,image_z)
assert bundle["accepted"] == bundle["requested"]
fig,ax = plt.subplots(1,2,figsize=(12,4.3))
for vertex,radius in [(0.,lens["r1"]),(lens["thickness"],lens["r2"])]:
    ax[0].plot([vertex+surface_sag(radius,abs(h)) for h in height],height,color="gray")
for h in np.linspace(-aperture,aperture,13):
    ray = pupil_ray(h,0.,field,lens,image_z)
    points=np.asarray(ray["points"])
    ax[0].plot(points[:,2],points[:,0],color="teal",alpha=.7)
ax[0].axvline(image_z,color="tab:orange",ls="--",label="Paraxial image plane")
ax[0].set(xlabel="z (mm)",ylabel="x (mm)",title="Computed geometric rays")
ax[0].set_aspect("equal",adjustable="box");ax[0].legend(fontsize=9)
for radius in [aperture,aperture/2]:
    b=trace_bundle(lens,radius,field,image_z)
    points=1000*(np.asarray(b["xy"])-b["centroid"])
    ax[1].scatter(points[:,0],points[:,1],s=10,label=f"Stop radius {radius:g} mm · RMS {1000*b['rms_mm']:.2f} μm")
ax[1].set(xlabel="x minus centroid (μm)",ylabel="y minus centroid (μm)",title="Same detector, same physical scale")
ax[1].set_aspect("equal",adjustable="box");ax[1].legend(fontsize=9)
show(fig,"03-pupil-bundle")
print("Paraxial EFL / back focal length / absolute image z:",first_order)
print("Geometric RMS:",1000*bundle["rms_mm"],"μm; accepted rays:",bundle["accepted"],"/",bundle["requested"])
# %% [markdown]
# ## 4 · Test a ray model against independent predictions
#
# Agreement between two plots is weak evidence. Here a ray close to the axis must
# approach a thick-lens **ABCD matrix** prediction. Then reverse a full two-surface
# ray: it must retrace both intersections and recover the original direction.
# Finally refine the pupil quadrature without moving the lens or detector.
# %%
n,d,R1,R2=lens["index"],lens["thickness"],lens["r1"],lens["r2"]
# Reduced-angle state (height, n*theta): surface power and translation are independent first-order optics.
S1=np.array([[1.,0.],[-(n-1)/R1,1.]])
T=np.array([[1.,d/n],[0.,1.]])
S2=np.array([[1.,0.],[-(1-n)/R2,1.]])
M=S2@T@S1
matrix_efl=-1/M[1,0];matrix_image=d-M[0,0]/M[1,0]
assert abs(matrix_efl-first_order["focal"])<1e-12
heights=np.array([.8,.4,.2,.1]);focus=[]
for h in heights:
    ray=pupil_ray(h,0.,0.,lens)
    focus.append(ray["exit"][2]-ray["exit"][0]*ray["direction"][2]/ray["direction"][0])
errors=abs(np.asarray(focus)-matrix_image)
assert np.all(errors[1:]<errors[:-1]/3.8)
forward=pupil_ray(4.,2.,6.,lens)
origin=np.asarray(forward["exit"])+3*np.asarray(forward["direction"])
direction=-np.asarray(forward["direction"])
for vertex,radius,ni,nt,expected in [(d,R2,1.,n,forward["surfaces"][1]["point"]),(0.,R1,n,1.,forward["surfaces"][0]["point"])]:
    hit=intersect_surface(origin,direction,vertex,radius,lens["clear_radius"])
    assert hit["status"]=="ok" and np.max(abs(np.asarray(hit["point"])-expected))<1e-10
    direction=np.asarray(refract(direction,hit["normal"],ni,nt)["direction"]);origin=hit["point"]
assert np.max(abs(direction+np.asarray(forward["surfaces"][0]["incoming"])))<1e-12
counts=[];rms=[]
for rings,spokes in [(3,12),(6,24),(12,48),(24,96)]:
    refined=trace_bundle(lens,6.,6.,image_z,rings,spokes)
    counts.append(refined["requested"]);rms.append(refined["rms_mm"])
assert abs(rms[-2]/rms[-1]-1)<.01
fig,ax=plt.subplots(1,2,figsize=(11,4))
ax[0].loglog(heights,errors,"o-");ax[0].set(xlabel="Near-axis ray height (mm)",ylabel="Focus error vs ABCD (mm)")
ax[1].semilogx(counts,np.array(rms)*1000,"o-");ax[1].set(xlabel="Pupil ray count",ylabel="RMS at fixed z (μm)")
show(fig,"04-independent-checks")
print("ABCD focus:",matrix_image,"mm; near-axis focus errors:",errors)
print("Pupil convergence:",list(zip(counts,rms)));print("Full-ray reversibility passed.")
# %% [markdown]
# ## Optional extension · A geometric spot is not the diffraction limit
#
# For each aperture, choose its best **single on-axis** geometric image plane.
# Compare the trend with the approximate Airy first-zero radius 0.61 λ/NA, using
# NA ≈ aperture/EFL in air. These are different measures, RMS and first-zero radius:
# do not add them or treat their crossing as an exact wave-optical prediction.
# The ray calculation has no wavelength. A shrinking ray spot cannot prove that
# a sufficiently small aperture will yield unlimited image resolution.
# %%
radii=np.linspace(1.,10.,19);geometric=[];airy=[]
for radius in radii:
    plane=best_image_plane(lens,radius,[0.])
    geometric.append(1000*trace_bundle(lens,radius,0.,plane)["rms_mm"])
    airy.append(1000*.61*.00055/(radius/first_order["focal"]))
fig,ax=plt.subplots(figsize=(8,4))
ax.semilogy(radii,geometric,label="Geometric RMS at its best plane")
ax.semilogy(radii,airy,label="Approximate Airy first-zero radius, 550 nm")
ax.set(xlabel="Entrance-stop radius (mm)",ylabel="Different radius measures (μm)");ax.legend()
show(fig,"05-geometric-versus-wave")
'''

E12 = r'''# %% [markdown]
# ## 1 · Specify a lens family before choosing an optimizer
#
# We design a two-surface spherical singlet in air. The entrance stop is fixed at
# z = −5 mm, and the mechanical clear radius is 12 mm. Index n = 1.5, centre thickness
# d = 4 mm, and paraxial effective focal length f = 50 mm remain fixed. The shape variable
# q=(c₁+c₂)/(c₁−c₂), with c=1/R, changes bending while preserving thick-lens power:
# Φ=(n−1)[c₁−c₂+(n−1)d c₁c₂/n]. The second variable is one detector z for **all fields**.
#
# **Predict:** can two lenses with the same paraxial focal length have different spot sizes?
# The following self-contained engine is the same source used by E11 and the film.
# Run it once; study intersection and vector Snell details in E11. No external files are loaded.
# %%
#@title Self-contained geometric-ray engine from E11
@@ENGINE@@
# %%
for q in [-1.2,-.6,0.,.6,1.2]:
    lens=prescription(q)
    assert abs(paraxial(lens)["focal"]-50.)<1e-11
    assert lens["edge_thickness"]>=.5
    print(f"q={q:+.2f}, R1={lens['r1']:.3f}, R2={lens['r2']:.3f}, EFL={paraxial(lens)['focal']:.6f}, edge={lens['edge_thickness']:.3f} mm")
# %% [markdown]
# ## 2 · Define the merit and the detector before optimizing
#
# Rays uniformly sample area in a fixed 6 mm-radius entrance stop. Within each field,
# RMS² is the mean squared distance from the **spot centroid**. Averaging these squared
# radii with equal weights across 0°, 3°, and 6° defines the merit in mm². Recentering
# a spot removes image height from its size; it does **not** move the detector or correct distortion.
#
# After the rear surface, every ray coordinate is affine in z, x(z)=a+bz. Sum the
# within-field variances over all fields **before** minimizing that one quadratic.
# This gives one common image plane for a candidate q, never separate best focus per field.
# Any lost, blocked, or TIR ray invalidates a candidate and gives a large penalty;
# edge thickness must remain at least 0.5 mm. The aperture cannot shrink during the search.
# **Predict:** will the best detector position for 0° also minimize the 6° spot?
# %%
aperture=6.;fields=[0.,3.,6.];q0=0.
lens0=prescription(q0);z0=paraxial(lens0)["image_z"]
initial=design_merit(q0,z0,aperture,fields)
common=best_image_plane(lens0,aperture,fields)
focus_z=np.linspace(49.,54.,51)
fig,ax=plt.subplots(figsize=(8,4))
for field in fields:
    values=[1000*trace_bundle(lens0,aperture,field,z)["rms_mm"] for z in focus_z]
    ax.plot(focus_z,values,label=f"Field {field:g}°")
ax.axvline(common,color="black",ls="--",label="One joint best plane")
ax.set(xlabel="Detector z from first vertex (mm)",ylabel="Geometric RMS radius (μm)")
ax.legend();show(fig,"01-common-image-plane")
print("Initial paraxial z:",z0,"mm; common least-squares z:",common,"mm")
assert all(b["image_z"]==z0 for b in initial["bundles"])
# %% [markdown]
# ## 3 · Optimize bending and the common plane
#
# A small bounded pattern search changes q. For each candidate q, the preceding
# quadratic solves its one common detector position. This is a two-variable design
# with one variable eliminated analytically. It is deterministic and cheap, but
# does not claim a global optimum. Search bounds are −1.2 ≤ q ≤ 1.2.
# Compare an on-axis objective with the field-weighted objective. A lower number
# for a different objective is not automatically a better lens: judge both on the same audit.
# %%
designs={"On-axis objective":optimize_design(q0,aperture,[0.],24),
         "Three-field objective":optimize_design(q0,aperture,fields,24)}
selected=designs["Three-field objective"]["best"]
for name,result in designs.items():
    best=result["best"]
    history=np.array([row["value_mm2"] for row in result["history"]])
    assert np.all(np.diff(history)<=1e-13) and best["valid"]
    assert best["value_mm2"]<result["start"]["value_mm2"]
    assert abs(paraxial(best["lens"])["focal"]-50)<1e-11
    assert best["aperture"]==aperture and all(b["image_z"]==best["image_z"] for b in best["bundles"])
    print(name,": q =",best["shape"],", common z =",best["image_z"],", merit =",best["value_mm2"],"mm²")
fig,ax=plt.subplots(1,2,figsize=(11,4))
history=designs["Three-field objective"]["history"]
ax[0].semilogy([h["value_mm2"] for h in history],"o-")
ax[0].set(xlabel="Pattern-search iteration",ylabel="Three-field merit (mm²)")
for name,result in designs.items():
    best=result["best"]
    values=[1000*trace_bundle(best["lens"],aperture,f,best["image_z"])["rms_mm"] for f in fields]
    ax[1].plot(fields,values,"o-",label=name)
ax[1].set(xlabel="Field angle (degrees)",ylabel="Same-audit geometric RMS (μm)");ax[1].legend(fontsize=9)
show(fig,"02-design-and-tradeoff")
# %% [markdown]
# ## 4 · Audit the result with checks the optimizer did not use
#
# Use more pupil rays, add the withheld fields 1.5° and 4.5°, and keep the final
# physical detector fixed. Check that the joint-plane derivative vanishes, and
# deliberately try an oversized aperture to confirm that clipping is penalized.
# Sampling convergence and ray accounting are separate from physical-model accuracy.
# **Predict:** does on-axis improvement alone guarantee the best result at every field?
# %%
q,z=selected["shape"],selected["image_z"]
audit_fields=[0.,1.5,3.,4.5,6.]
fig,ax=plt.subplots(1,2,figsize=(12,4.3))
for name,result in designs.items():
    best=result["best"]
    audit=[trace_bundle(best["lens"],aperture,f,best["image_z"],12,48) for f in audit_fields]
    assert all(b["accepted"]==b["requested"] for b in audit)
    ax[0].plot(audit_fields,[1000*b["rms_mm"] for b in audit],"o-",label=name)
    cloud=np.asarray(audit[-1]["xy"])-audit[-1]["centroid"]
    ax[1].scatter(1000*cloud[:,0],1000*cloud[:,1],s=8,label=name,alpha=.6)
ax[0].set(xlabel="Field angle (degrees)",ylabel="Audited geometric RMS (μm)");ax[0].legend(fontsize=9)
ax[1].set(xlabel="x minus centroid (μm)",ylabel="y minus centroid (μm)",title="6° field · separate fixed detector per design")
ax[1].set_aspect("equal",adjustable="box");ax[1].legend(fontsize=9);show(fig,"03-independent-field-audit")
h=1e-3
derivative=(design_merit(q,z+h,aperture,fields)["value_mm2"]-design_merit(q,z-h,aperture,fields)["value_mm2"])/(2*h)
assert abs(derivative)<1e-9
coarse=design_merit(q,z,aperture,fields,6,16)
fine=design_merit(q,z,aperture,fields,12,48)
finer=design_merit(q,z,aperture,fields,24,96)
assert abs(fine["value_mm2"]/finer["value_mm2"]-1)<.01
clipped=design_merit(q,z,13.,fields)
assert not clipped["valid"] and clipped["loss_fraction"]>0 and clipped["value_mm2"]>=1e6
print("Common-plane derivative:",derivative,"mm")
print("Pupil-refined merits (same device and plane):",[v["value_mm2"] for v in [coarse,fine,finer]])
print("Deliberately oversized stop: lost fraction",clipped["loss_fraction"],"penalized merit",clipped["value_mm2"])
# %% [markdown]
# ## Optional extension · Perturb the manufactured prescription
#
# The design is a nominal geometric model. Small radius, thickness, and index changes
# need not preserve its focal power. Draw seeded perturbations and evaluate them at
# the **unchanged** nominal detector with the same aperture and fields. This is a
# sensitivity illustration, not a manufacturing-yield prediction: the distributions
# are chosen here, and diffraction, real glass dispersion, tilt and decentre are absent.
# %%
rng=np.random.default_rng(41);nominal=selected["lens"];perturbed=[]
for _ in range(32):
    lens=make_lens(nominal["r1"]+rng.normal(0,.05),nominal["r2"]+rng.normal(0,.05),
                   nominal["thickness"]+rng.normal(0,.02),nominal["index"]+rng.normal(0,.0005),12.)
    audit=[trace_bundle(lens,aperture,f,z) for f in fields]
    assert lens["edge_thickness"]>.5 and all(b["throughput"]==1 for b in audit)
    perturbed.append(1000*np.sqrt(np.mean([b["rms_mm"]**2 for b in audit])))
fig,ax=plt.subplots(figsize=(8,4))
ax.hist(perturbed,bins=10,color="teal",alpha=.7)
ax.axvline(1000*selected["rms_mm"],color="tab:orange",ls="--",label="Nominal")
ax.set(xlabel="Joint geometric RMS at unchanged detector (μm)",ylabel="Synthetic prescriptions");ax.legend()
show(fig,"04-prescription-sensitivity")
'''


def build():
    e11 = E11.replace("@@SNELL@@", IMPORT + functions("dot", "unit", "refract"))
    e11 = e11.replace("@@SURFACE@@", functions("curvature", "surface_sag", "intersect_surface"))
    e11 = e11.replace("@@BUNDLE@@", functions("make_lens", "paraxial", "trace_ray", "pupil_ray", "pupil_samples", "trace_bundle", "best_image_plane"))
    # Strip only the module docstring; keep the plain import and all canonical definitions.
    engine = IMPORT + "\n\n" + "\n\n".join(FUNCTIONS.values())
    e12 = E12.replace("@@ENGINE@@", engine)
    for slug, body in [("e11-ray-tracing", e11), ("e12-lens-design", e12)]:
        path = ROOT / slug / "lab.py"
        path.parent.mkdir(exist_ok=True)
        path.write_text(body)
    print("Built self-contained E11/E12 lab sources from the canonical geometric-ray engine.")


if __name__ == "__main__":
    build()
