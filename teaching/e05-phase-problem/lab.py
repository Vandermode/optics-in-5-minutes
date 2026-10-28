# %% [markdown]
# ## 1 · Encode the missing phase in a carrier
#
# A camera measures $|U|^2$, which is unchanged if U is multiplied by any local phase.
# Add a known tilted reference $R=A\exp(2\pi i c x/N)$ instead. The intensity now
# contains $UR^*$ and its conjugate, displaced to opposite Fourier sidebands.
# We deliberately band-limit U to B bins so that an isolated sideband can be recovered exactly.
# **Predict:** what fails if c is too small, too close to Nyquist, or the crop is too narrow?
# %%
N=128;B=8;carrier=32;A=2.
x=(np.arange(N)-N//2)/N;X,Y=np.meshgrid(x,x)
phase_object=1.1*np.exp(-((X+.18)**2+(Y-.08)**2)/.012)+.7*np.exp(-((X-.15)**2+(Y+.12)**2)/.006)
amplitude=1-.3*np.exp(-((X+.12)**2+(Y+.19)**2)/.02)
bins=np.fft.fftfreq(N)*N;FX,FY=np.meshgrid(bins,bins)
band=np.hypot(FX,FY)<=B
U=np.fft.ifft2(np.fft.fft2(amplitude*np.exp(1j*phase_object))*band)
def hologram_recover(U,carrier=32,radius=8):
    reference=A*np.exp(2j*np.pi*carrier*np.arange(N)/N)[None,:]
    camera=abs(U+reference)**2
    # UR* lies at -carrier. Move it back to zero, then keep its bandwidth.
    sideband=np.roll(np.fft.fft2(camera),carrier,axis=1)*(np.hypot(FX,FY)<=radius)/A
    return camera,np.fft.ifft2(sideband)
camera,recovered=hologram_recover(U)
error=np.linalg.norm(recovered-U)/np.linalg.norm(U)
assert error<1e-12
fig,ax=plt.subplots(1,3,figsize=(12,3.6),layout="constrained")
camera_image=ax[0].imshow(camera,cmap="magma")
ax[0].set(title="Recorded intensity",xlabel="Pixel",ylabel="Pixel")
fig.colorbar(camera_image,ax=ax[0],label="Intensity (a.u.)")
phase_limit=.2*np.ceil(max(np.max(abs(np.angle(U))),np.max(abs(np.angle(recovered))))/.2)
for a,img,title in zip(ax[1:],[np.angle(U),np.angle(recovered)],["Known phase","Recovered phase"]):
    phase_image=a.imshow(img,cmap="RdBu_r",vmin=-phase_limit,vmax=phase_limit)
    a.set(title=title,xlabel="Pixel",ylabel="Pixel")
fig.colorbar(phase_image,ax=ax[1:],label="Phase (rad) · shared scale",ticks=[-phase_limit,0,phase_limit])
show(fig,"01-holographic-recovery")
print("Relative complex-field error with isolated sideband:",error)
for c,r in [(12,8),(32,3),(60,8)]:
    _,bad=hologram_recover(U,c,r)
    print(f"Carrier {c}, crop {r}: relative field error {np.linalg.norm(bad-U)/np.linalg.norm(U):.3g}")
# %% [markdown]
# ## 2 · A recovered complex field can be refocused
#
# Propagate U with a scalar angular-spectrum transfer function. Here pixels are 1 μm,
# wavelength is 0.55 μm, and the object is band-limited to propagating frequencies.
# Backward propagation is safe for this noiseless propagating example; amplifying
# evanescent components or missing/noisy sidebands is a different inverse problem.
# %%
dx=1.;wavelength=.55
def propagate(field,z):
    f=np.fft.fftfreq(N,dx);fx,fy=np.meshgrid(f,f)
    kz=2*np.pi*np.sqrt((1/wavelength**2-fx*fx-fy*fy).astype(complex))
    return np.fft.ifft2(np.fft.fft2(field)*np.exp(1j*kz*z))
z=150.
sensor=propagate(U,z)
_,sensor_recovered=hologram_recover(sensor)
focused=propagate(sensor_recovered,-z)
# Compare up to a global phase, as an unknown reference phase would require.
alignment=np.vdot(U,focused);focused*=np.exp(-1j*np.angle(alignment))
assert np.linalg.norm(focused-U)/np.linalg.norm(U)<1e-12
fig,ax=plt.subplots(1,3,figsize=(12,3.6))
for a,img,title in zip(ax,[abs(U)**2,abs(sensor)**2,abs(focused)**2],["Object intensity","Defocused sensor","Numerically refocused"]):
    a.imshow(img,cmap="magma",vmin=.35,vmax=1.2);a.set(title=title,xlabel="Pixel")
show(fig,"02-refocusing")
# %% [markdown]
# ## 3 · Recover phase from two magnitude constraints
#
# Without a reference, one intensity plane is insufficient. Gerchberg–Saxton alternates
# between planes: keep the current phase but replace the amplitude with a measured or
# desired amplitude. This enforces one constraint at a time, not both simultaneously.
# A global phase remains undetermined. Measure two different errors on each completed update:
# the Fourier-amplitude residual $r=\|\,|F(u)|-b\,\|_2/\|b\|_2$, where $b=|F(U)|$ is measured,
# and the field error $e=\min_\alpha\|e^{i\alpha}u-U\|_2/\|U\|_2$ against the known synthetic U.
# The second error is available here because we made the object; it is not a measured loss.
# **Predict:** if three initial phases reach small amplitude residuals, must they recover U?
# %%
F=lambda u:np.fft.fft2(u,norm="ortho")
FH=lambda v:np.fft.ifft2(v,norm="ortho")
known_amplitude=abs(U);measured_amplitude=abs(F(U))
def align_global_phase(u):
    return u*np.exp(-1j*np.angle(np.vdot(U,u)))

def error_metrics(u):
    residual=np.linalg.norm(abs(F(u))-measured_amplitude)/np.linalg.norm(measured_amplitude)
    field_error=np.linalg.norm(align_global_phase(u)-U)/np.linalg.norm(U)
    return np.array([residual,field_error])

assert max(error_metrics(U*np.exp(.73j)))<1e-12  # A pure global phase is not an error.

def gs(seed,iterations=150):
    random=np.random.default_rng(seed)
    u=known_amplitude*np.exp(1j*random.uniform(-np.pi,np.pi,(N,N)))
    history=[error_metrics(u)]
    for _ in range(iterations):
        v=F(u)
        u=known_amplitude*np.exp(1j*np.angle(FH(measured_amplitude*np.exp(1j*np.angle(v)))))
        history.append(error_metrics(u))  # Both metrics describe this same post-update field.
    return u,np.asarray(history)

seeds=[3,7,11];iterations=150
results=[gs(seed,iterations) for seed in seeds]
colors=["#0072B2","#D55E00","#009E73"]
fig=plt.figure(figsize=(12,7.4),layout="constrained")
grid=fig.add_gridspec(2,4,height_ratios=[1,1.1])
error_axes=[fig.add_subplot(grid[0,:2]),fig.add_subplot(grid[0,2:])]
for seed,color,(u,history) in zip(seeds,colors,results):
    for column,a in enumerate(error_axes):
        a.semilogy(np.arange(iterations+1),100*history[:,column],color=color,label=f"Seed {seed}")
    assert history.shape==(iterations+1,2)
    assert history[-1,0]<history[0,0]
    assert np.max(abs(history[-1]-error_metrics(u)))<1e-12
    assert np.max(abs(abs(u)-known_amplitude))<1e-12
    # Check the stated default example, while allowing the experiment below to change its budget.
    if iterations==150 and seed in [3,7,11]:
        assert history[-1,0]<.01 and history[-1,1]>.1
for a,title in zip(error_axes,["Agreement with measured Fourier amplitudes","Error against the known complex field"]):
    a.set(title=title,xlabel="Completed GS updates",ylabel="Relative error (%)",ylim=(.1,250))
    a.set_yticks([.1,1,10,100],["0.1","1","10","100"])
    a.legend(fontsize=9)
phase_axes=[fig.add_subplot(grid[1,j]) for j in range(4)]
phase_fields=[U]+[align_global_phase(u) for u,history in results]
phase_limit=.2*np.ceil(max(np.max(abs(np.angle(field))) for field in phase_fields)/.2)
titles=["Known phase"]+[f"Seed {seed}\nField error {100*history[-1,1]:.1f}%" for seed,(u,history) in zip(seeds,results)]
for a,field,title in zip(phase_axes,phase_fields,titles):
    phase_image=a.imshow(np.angle(field),cmap="RdBu_r",vmin=-phase_limit,vmax=phase_limit)
    a.set(title=title,xlabel="x pixel",ylabel="y pixel")
for a,color in zip(phase_axes[1:],colors):a.title.set_color(color)
colorbar=fig.colorbar(phase_image,ax=phase_axes,orientation="horizontal",shrink=.6,pad=.04)
colorbar.set_label("Phase (rad) · common scale · reconstructions aligned to remove global phase")
colorbar.set_ticks([-phase_limit,0,phase_limit])
fig.suptitle(f"Small measurement residuals can coexist with substantial field errors\nSame two magnitude constraints · {iterations} updates · three initial phases",fontsize=13)
show(fig,"03-phase-retrieval")
for seed,(u,history) in zip(seeds,results):
    print(f"Seed {seed}, completed update {iterations}: amplitude residual {100*history[-1,0]:.3f}%; phase-aligned field error {100*history[-1,1]:.3f}%")
# %% [markdown]
# These finite runs fit the amplitudes much better than they recover the known field.
# They do not prove that the inverse problem is nonunique or that longer runs cannot improve.
# Change the iteration count or initial seeds and compare both trajectories. For an unknown
# object, another independent measurement can test predictions beyond the fitted data;
# the ground-truth field error used here is unavailable.
# %% [markdown]
# ## Optional extension · Design, then quantize, a phase-only mask
#
# Replace the measured Fourier amplitude with a target. The pupil has fixed amplitude,
# so the requested target may not be exactly feasible. After optimization, quantizing
# phase generally worsens the objective: that fabrication constraint was absent from the iterations.
# %%
pupil=(X*X+Y*Y<.4**2).astype(float)
target=np.exp(-((X-.18)**2+Y*Y)/.003)+np.exp(-((X+.18)**2+Y*Y)/.003)
target*=np.sum(pupil**2)/target.sum()
u=pupil*np.exp(1j*np.random.default_rng(23).uniform(-np.pi,np.pi,(N,N)))
for _ in range(100):
    v=np.fft.fftshift(F(u))
    u=pupil*np.exp(1j*np.angle(FH(np.fft.ifftshift(np.sqrt(target)*np.exp(1j*np.angle(v))))))
phi=np.angle(u);quantized=np.round(phi/(2*np.pi/4))*(2*np.pi/4)
outputs=[target,abs(np.fft.fftshift(F(u)))**2,abs(np.fft.fftshift(F(pupil*np.exp(1j*quantized))))**2]
fig,ax=plt.subplots(1,3,figsize=(12,3.6))
for a,img,title in zip(ax,outputs,["Requested target","Continuous phase","Four phase levels"]):
    a.imshow(img,cmap="magma",vmin=0,vmax=target.max());a.set(title=title,xlabel="Focal-plane sample")
show(fig,"04-phase-only-design")
assert abs(outputs[1].sum()-pupil.sum())<1e-9
