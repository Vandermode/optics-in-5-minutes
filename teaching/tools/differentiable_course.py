"""E13: chain a physical image model to a real design objective, EN/ZH together."""
from geometric_course import tr, p, h, eq, slider, question, exercise, TRANSLATIONS

MODULES = [dict(
    id='E13', slug='e13-differentiable-optics', local_preview=True, guided_lab=True,
    title=tr('Differentiable optics: from images to design updates', '可微光学：从图像误差到设计更新'),
    headline=tr('Predict the image.', '预测图像。') + '<br><em>' + tr('Differentiate the design.', '求出设计梯度。') + '</em>',
    deck=tr('Connect optical parameters to a measured intensity and a scalar loss. Check the gradient, improve a phase element, and see how the same chain reaches lens design and computational imaging.', '将光学参数、测量强度与标量损失连成一条计算链。检验梯度，改进相位元件，再理解这条链如何延伸到镜头设计与计算成像。'),
    prerequisites=['E02', 'E10', 'E12'],
    goals=[tr('Trace the dependency from real design parameters through a complex field to a real loss.', '追踪实数设计参数如何通过复数光场影响实数损失。'),
           tr('Check a reverse-mode gradient with finite differences before using it to update a design.', '先用有限差分检验反向梯度，再用它更新设计。'),
           tr('Distinguish differentiable optics, automatic differentiation, adjoint methods and end-to-end co-design.', '区分可微光学、自动微分、伴随法和端到端联合设计。')],
    defaults=dict(tilt=1.5, defocus=2, target=0, iterations=40),
    controls=[
        slider('tilt', 'Initial tilt coefficient a', '初始倾斜系数 a', -4, 4, .1, 'rad', 1, 'The linear pupil phase steers the focal pattern.', '瞳面线性相位改变焦面图样的位置。'),
        slider('defocus', 'Initial defocus coefficient b', '初始离焦系数 b', -4, 4, .1, 'rad', 1, 'Quadratic phase broadens the focused pattern at this fixed detector plane.', '在这个固定探测器平面，二次相位使聚焦图样展宽。'),
        slider('target', 'Target tilt a★', '目标倾斜 a★', -2, 2, .5, 'rad', 1, 'The target is generated with this tilt and zero defocus; it is reachable in the stated model.', '目标由此倾斜系数与零离焦生成，在当前模型中可实现。'),
        slider('iterations', 'Maximum gradient steps', '最大梯度更新步数', 0, 80, 5, '', 0, 'Zero shows the initial design. Each accepted step must reduce the same loss.', '零步显示初始设计。每个被接受的更新都必须降低同一个损失。')],
    presets=[dict(label=tr('Inspect before updating', '先检查，再更新'), values=dict(tilt=1.5, defocus=2, target=0, iterations=0)),
             dict(label=tr('Use the gradient', '使用梯度'), values=dict(tilt=1.5, defocus=2, target=0, iterations=40)),
             dict(label=tr('A difficult starting point', '一个困难的初值'), values=dict(tilt=3, defocus=3, target=0, iterations=40)),
             dict(label=tr('Move the target', '移动目标'), values=dict(tilt=1.5, defocus=2, target=-1.5, iterations=60))],
    prediction=tr('Changing a phase mask leaves its local intensity unchanged. Can it still change the focal-plane intensity? Predict what the tilt and defocus controls do, then compare zero updates with forty updates.', '改变相位掩模不会改变掩模处的局部强度，但它能改变焦面强度吗？先预测倾斜和离焦控件的作用，再比较零次与四十次更新。'),
    intro=(p('E12 asks what makes a lens better; E10 shows how an adjoint obtains many sensitivities at once. Differentiable optics connects these ideas: write the optical prediction as a sequence of operations, and carry the loss sensitivity back to the physical parameters you can change.', 'E12 定义什么样的镜头更好，E10 用伴随法一次求出许多灵敏度。可微光学将两者连接起来：把光学预测写成一系列运算，再将损失的灵敏度传回实际可调的物理参数。')
           + eq('θ = (a, b) → φ(x) → u(x) → v(q) → I(q) → L')
           + p('Here x is normalized pupil position, q labels sampled focal-plane spatial frequencies, a and b are real phase coefficients in radians, u is the pupil field, v is the focal field, and I is the intensity. A fixed aperture and incident power keep the task unchanged. The target I★ is the focused pattern of the same aperture with a chosen tilt a★.', '这里 x 是归一化瞳坐标，q 表示采样的焦面空间频率，a、b 是以弧度计的实数相位系数，u 是瞳面场，v 是焦面场，I 是强度。孔径与入射功率保持固定。目标 I★ 是同一孔径在指定倾斜 a★、零离焦时的聚焦图样。')
           + eq('φ(x) = ax + bx²; &nbsp; u = A exp(iφ); &nbsp; v = F u; &nbsp; I = |v|²')
           + p('F is a unitary discrete Fourier transform. The pupil amplitude A is fixed and normalized so ΣA² = 1; consequently ΣI = 1. The camera measures squared field magnitude. It does not directly measure phase, but interference during propagation makes the intensity depend on phase.', 'F 是酉离散傅里叶变换。瞳振幅 A 固定，并归一化为 ΣA² = 1，因此 ΣI = 1。相机测量场模的平方，并不直接测量相位；但传播中的干涉会让强度依赖相位。')),
    steps=[tr('Set the update count to zero. Change tilt alone, then defocus alone. Explain which changes the position and which changes the shape of the detector pattern.', '把更新次数设为零。分别只改变倾斜与离焦，说明哪个主要改变探测器图样的位置，哪个改变形状。'),
           tr('Inspect both gradient components at the initial design. Compare reverse mode with a central finite difference; the sign predicts which small parameter change lowers the loss.', '检查初始设计处的两个梯度分量，比较反向计算与中心有限差分。梯度符号预测哪个方向的小参数变化会降低损失。'),
           tr('Choose Use the gradient and compare initial, target and final intensity on the same scale. Move the target and repeat. Report loss reduction without claiming that every start reaches the unique best design.', '选择“使用梯度”，在同一尺度上比较初始、目标和最终强度。移动目标并重复实验，报告损失下降，但不要据此宣称所有初值都能到达唯一最优设计。')],
    theory=(h('Differentiate the actual measured quantity.', '对实际测量量求导。')
        + eq('S = Σq I★(q)²; &nbsp; L(θ) = Σq [I(q; θ) − I★(q)]² / (2S); &nbsp; r = (I − I★)/S')
        + p('S is the fixed squared norm of the target. Dividing by S makes the dimensionless loss relative to that target; S is held constant when differentiating the design. Its value compares the full sampled pattern, not just the peak. Real cameras also introduce pixel integration, noise and saturation; this core model omits them.', 'S 是目标强度的固定平方范数。除以 S 得到相对该目标的无量纲损失；对设计求导时 S 保持不变。它比较完整采样图样，而不只比较峰值。真实相机还包含像素积分、噪声和饱和；核心模型暂不包含这些因素。')
        + p('At a detector sample, a positive residual means that the prediction is too bright. Since dL = Σ r dI, the field sensitivity contains r ⊙ v. The conjugate transpose F† then combines the contribution of every detector sample at each pupil position.', '在某个探测器采样点，残差为正表示预测强度过亮。由 dL = Σ r dI 可知，场灵敏度包含 r ⊙ v。随后，共轭转置 F† 将所有探测器采样点的贡献汇总到各个瞳面位置。')
        + eq('dL = 2 Re[(r ⊙ v)† dv] = 2 Re[w† du]')
        + eq('dI = 2 Re(v* ⊙ dv); &nbsp; w = F†(r ⊙ v); &nbsp; gφ = 2 Im(u* ⊙ w)')
        + p('The star denotes complex conjugation, ⊙ pointwise multiplication, and † the conjugate transpose. Move the field sensitivity back through F using F†, then use du = i u dφ. This gives the real phase sensitivity gφ. The backward field is a mathematical sensitivity, not light literally travelling backwards through the experiment.', '星号表示复共轭，⊙ 表示逐点相乘，† 表示共轭转置。用 F† 将场的灵敏度传回瞳面，再利用 du = i u dφ，便得到实数相位灵敏度 gφ。反向场是数学上的灵敏度，不是实验中真的有光倒着传播。')
        + eq('∂L/∂a = Σx gφ(x)x; &nbsp; ∂L/∂b = Σx gφ(x)x²')
        + p('This final chain-rule step is the new connection to design. E10 allowed each phase pixel to change independently; here only two physical phase shapes may change. Sum the pixel sensitivities with the derivatives of those shapes. A constant phase offset would have zero intensity sensitivity, so it is not a useful design variable for this task.', '最后这一步链式法则把灵敏度连接到设计。E10 允许每个相位像素独立变化；这里仅允许两个相位形状变化。用形状对参数的导数加权累加各像素灵敏度即可。常量相位偏移对强度的灵敏度为零，因此不适合作为此任务的设计变量。')
        + h('Check the derivative before trusting an update.', '先检验导数，再相信更新。')
        + eq('gⱼ ≈ [L(θ + h eⱼ) − L(θ − h eⱼ)] / (2h); &nbsp; θnew = θ − ηg')
        + p('eⱼ selects one parameter and h is a small perturbation in radians. Too large a step adds truncation error; too small a step subtracts nearly equal floating-point numbers. Sweep several h values in the notebook. The optimizer separately chooses η by backtracking and accepts a step only after recomputing a lower loss. A correct gradient is local information, not a guarantee of a global optimum.', 'eⱼ 选中一个参数，h 是以弧度计的小扰动。h 过大会产生截断误差，过小则会让几乎相等的浮点数相减。在 Notebook 中比较多个 h。优化器另行通过回溯选择更新步长 η，并在重新计算确认损失降低后才接受更新。正确梯度只提供局部信息，不保证全局最优。')
        + p('At the default initial design, the loss gradient initially asks for more defocus while reducing tilt. The two parameters act together on the pattern: descent reduces their combined loss, not necessarily each coefficient separately. Try the difficult starting point to see a nonzero plateau even though a zero-loss target is known.', '在默认初始设计处，损失梯度最初要求减小倾斜、同时增加离焦。两个参数共同影响图样：梯度下降减小的是它们共同决定的损失，不一定分别减小每个系数。尝试困难初值，可以观察到已知存在零损失目标时仍出现非零平台。')
        + h('Three related ideas have different roles.', '三个相关概念，各有不同作用。')
        + p('Differentiable optics is an optical model with usable derivatives with respect to chosen design variables. Automatic differentiation applies local derivative rules to a computation graph. An adjoint method efficiently evaluates the sensitivity of a scalar objective, often through an implicit field solve. They can be combined: an optical solver may supply a custom adjoint derivative inside an automatic-differentiation system. This experiment writes reverse-mode derivatives explicitly; it does not call an autodiff library.', '可微光学指能够对选定设计变量求出可用导数的光学模型。自动微分将局部求导规则应用于计算图。伴随法高效计算标量目标的灵敏度，常用于隐式场求解。三者可以结合：光学求解器可在自动微分系统中提供自定义伴随导数。本实验显式编写反向导数，不调用自动微分库。')
        + h('The same chain can pass through curved lenses.', '同一条链也可以穿过曲面镜头。')
        + p('In E11, an intersection satisfies S(r₀ + td; θ) = 0. If its derivative with respect to t is nonzero and the same intersection branch remains selected, implicit differentiation gives dt/dθ = −(∂S/∂θ)/(∂S/∂t), with the incoming ray held fixed for this local derivative. The complete chain must also differentiate the incoming ray and Snell refraction. Grazing incidence, clipping, total internal reflection and switching intersections can invalidate a smooth local derivative.', 'E11 的交点满足 S(r₀ + td; θ) = 0。若对 t 的导数非零，且始终选取同一个交点分支，隐式求导给出 dt/dθ = −(∂S/∂θ)/(∂S/∂t)；这一局部导数将入射光线保持固定。完整链条还需要对入射光线与斯涅尔折射求导。掠入射、遮挡、全内反射和交点切换可能破坏平滑局部导数的适用条件。')
        + h('End-to-end design changes where the loss is measured.', '端到端设计改变的是损失的评价位置。')
        + eq('y = Hθ(s); &nbsp; ŝ = Rψ(y); &nbsp; Ltask = ℓ(ŝ, s)')
        + p('s is a scene, Hθ the optical and sensor model, y the measurement, and Rψ a reconstruction with digital parameters ψ. Joint design differentiates the final task loss through both Rψ and Hθ, updating optics and reconstruction together. A sharper raw PSF is then not automatically the best task result. The present browser experiment optimizes an optical intensity pattern only; it is not a trained reconstruction system.', 's 是场景，Hθ 是光学与传感器模型，y 是测量，Rψ 是参数为 ψ 的重建算法。联合设计将最终任务损失依次通过 Rψ 和 Hθ 求导，同时更新光学与重建参数。此时，原始 PSF 更锐利并不必然意味着任务结果更好。当前浏览器实验只优化光学强度图样，并未训练重建系统。')
        + p('Hold the scene distribution, illumination, aperture, exposure and sensor model fixed when comparing designs. Evaluate held-out scenes and physical tolerances. Hard phase rounding or changing a ray path is not made differentiable merely by placing it in a computation graph: use a justified smooth model during optimization and check the final discrete or fabricated design separately.', '比较设计时，应固定场景分布、照明、孔径、曝光和传感器模型，并用未参与优化的场景及物理容差评价结果。把相位硬量化或光路切换放进计算图，并不会使其自动可微：优化时应采用有依据的平滑模型，再单独检验最终离散或实际加工设计。')),
    limits=tr('A monochromatic 1D scalar phase pupil with fixed amplitude and power, unitary Fourier propagation, and two continuous real parameters. The detector samples normalized intensity. No polarization, material dispersion, pixel integration, noise, fabrication process or learned reconstruction is simulated. Low loss establishes agreement with this target in this model, not unique parameter recovery or physical performance.', '单色一维标量相位瞳，振幅与功率固定，采用酉傅里叶传播和两个连续实数参数。探测器采样归一化强度。不模拟偏振、材料色散、像素积分、噪声、加工过程或学习式重建。低损失表明当前模型与目标吻合，并不证明参数恢复唯一或实物性能良好。'),
    extension=tr('Quantize the optimized phase and reevaluate its intensity loss. Explain why the continuous design gradient is not the ordinary derivative of hard phase rounding.', '量化优化后的相位，重新评价强度损失。解释为什么连续设计的梯度不是相位硬取整的普通导数。'),
    quiz=[question('Which path must a design gradient include?', '设计梯度必须包含哪条依赖链？', ['Parameters → field → intensity → loss', 'Only the optimizer update', 'Only the displayed peak intensity'], ['参数 → 光场 → 强度 → 损失', '只有优化器更新', '只有显示出的峰值强度'], 0, 'The derivative must match the full objective actually evaluated, including the intensity measurement.', '导数必须与实际评价的完整目标一致，包括强度测量。'),
          question('A finite-difference check agrees with the reverse gradient. What does this establish?', '有限差分与反向梯度吻合，这说明什么？', ['The fabricated optic will work', 'The derivative of the implemented objective is locally consistent', 'The objective has a unique global minimum'], ['实物光学元件一定有效', '实现的目标函数导数在局部相互一致', '目标函数有唯一全局最小值'], 1, 'A gradient check tests a numerical derivative; model fidelity and optimization outcomes require separate evidence.', '梯度检验只检验数值导数；模型保真度和优化结果需要另外的证据。'),
          question('What makes optical and digital design end-to-end?', '什么使光学与数字设计成为端到端设计？', ['Using a neural network somewhere', 'Minimizing raw PSF width only', 'Differentiating a final task loss through reconstruction and optical formation'], ['在某处使用神经网络', '只最小化原始 PSF 宽度', '将最终任务损失通过重建与光学成像共同求导'], 2, 'The placement of the task loss and its dependency on both sets of parameters defines the joint problem.', '任务损失的位置，以及它对两组参数的依赖，定义了联合问题。')],
    exercises=[exercise('Write the forward chain', '写出前向链', 'Name the design variables, fixed quantities, measurement and loss. Show why phase can change the focal intensity without changing pupil power.', '列出设计变量、固定量、测量和损失，说明相位为何能在保持瞳面功率不变时改变焦面强度。', 'Unit-modulus phase preserves local pupil intensity; Fourier interference redistributes that fixed power among detector samples.', '单位模相位保持局部瞳强度；傅里叶干涉将固定功率重新分配到探测器采样点。'),
                exercise('Audit both gradient components', '复核两个梯度分量', 'Compare reverse-mode and finite-difference derivatives over several perturbation sizes at the same initial design.', '在同一初始设计处，对多个扰动步长比较反向与有限差分导数。', 'Look for a range of h where truncation and roundoff errors are both small. Compare absolute errors when a true component is near zero.', '寻找截断误差与舍入误差都较小的 h 范围。真实梯度分量接近零时，应比较绝对误差。'),
                exercise('Interpret the optimized result', '解释优化结果', 'Report the initial and final loss, fixed target and power, and one limitation of the result.', '报告初始与最终损失、固定目标与功率，以及结果的一项局限。', 'A decreasing loss supports improvement for this sampled target. Intensity ambiguities, local stationary points and omitted physics prevent stronger conclusions.', '损失下降支持对当前采样目标的改善。强度歧义、局部驻点和被忽略的物理因素限制了更强的结论。')],
    refs=[(tr('PyTorch · Autograd mechanics and complex-valued operations', 'PyTorch · 自动微分机制与复数运算'), 'https://docs.pytorch.org/docs/stable/notes/autograd.html'),
          (tr('Wang et al. · dO: A Differentiable Engine for Deep Lens Design', 'Wang 等 · dO：深度镜头设计可微引擎'), 'https://doi.org/10.1109/TCI.2022.3212837'),
          (tr('Sitzmann et al. · End-to-end optimization of optics and image processing', 'Sitzmann 等 · 光学与图像处理的端到端优化'), 'https://web.stanford.edu/~boyd/papers/pdf/end_to_end_opt_optics.pdf'),
          (tr('DeepLens · Differentiable optical models and examples', 'DeepLens · 可微光学模型与示例'), 'https://github.com/vccimaging/DeepLens')],
)]

MOTIVATIONS = {'E13': dict(question=tr('Which optical parameter should change to improve the measured image?', '为了改善测量图像，应该改变哪个光学参数？'), problem=tr('Trying every lens curvature or phase coefficient separately becomes expensive. A derivative connects a measured error to its causes in the optical model, so a design update can follow a calculated direction.', '逐一尝试每个透镜曲率或相位系数的代价很高。导数把测量误差与光学模型中的参数联系起来，让设计沿着计算出的方向更新。'), payoff=tr('Build and verify a differentiable optical calculation, understand its relation to adjoints, and identify what is needed before extending it to joint optics and reconstruction design.', '建立并检验可微光学计算，理解它与伴随法的关系，并明确扩展到光学与重建联合设计之前需要补充什么。'))}
PREPARATION = {'E13': dict(skills=[dict(topic='derivatives', need=tr('Apply the chain rule to a real scalar loss.', '对实数标量损失应用链式法则。'), use=tr('The final loss depends on phase through a field and an intensity measurement.', '最终损失通过光场与强度测量依赖相位。')), dict(topic='complex-matrices', need=tr('Use conjugation and a unitary Fourier transform.', '使用复共轭与酉傅里叶变换。'), use=tr('The conjugate transpose carries field sensitivity back to the pupil.', '共轭转置把场的灵敏度传回瞳面。'))], notebook=tr('Manipulate complex arrays and compare scalar function values at nearby parameter settings.', '操作复数数组，并比较相邻参数设置处的标量函数值。'), coding=['numpy', 'fft'], new=tr('Connect parameterized optics, reverse derivatives, gradient checks and a design update. No deep-learning framework is required.', '连接参数化光学、反向导数、梯度检验与设计更新；无需深度学习框架。'))}

# Composed page strings also need complete translations.
for m in MODULES:
    title=m['title']; zh=TRANSLATIONS[title]
    for suffix, translated in [('AN INTERACTIVE OPTICS LESSON', '互动光学课程'), ('LESSON + PYTHON LAB', '课程 + PYTHON 实验'), ('WHY LEARN THIS?', '为什么学这一课？')]:
        tr(f"E13 / {suffix}", f"E13 / {translated}")
    tr('BEFORE E13 / PREREQUISITES', '学习 E13 之前 / 预备知识')
    tr(f'E13 · {title} · Optics in 5 Minutes', f'E13 · {zh} · 五分钟光学')
    tr(f'OPTICS IN 5 MINUTES · E13 / {title.upper()} ·', f'五分钟光学 · E13 / {zh} ·')
    tr(f'Computed {title} illustration.', f'{zh}的计算示意图。')
    for i,q in enumerate(m['quiz'],1): tr(f"{i}. {q['prompt']}", f"{i}. {TRANSLATIONS[q['prompt']]}")
    for skill in PREPARATION['E13']['skills']: tr('Used here: '+skill['use'], '本课用途：'+TRANSLATIONS[skill['use']])
    for label,_ in m['refs']: tr(label+' ↗', TRANSLATIONS[label]+' ↗')

RUNTIME = {
    'Explore a connected map of optical concepts and choose from thirteen interactive lessons with independent Python labs.': '浏览相互关联的光学概念图谱，从十三节互动课程及其独立 Python 实验中选择学习主题。',
    'Focused preparation for thirteen optics lessons, with specific skills and selected primary learning resources.': '十三节光学课的针对性预备知识，包含具体技能要求和精选一手学习资源。',
    'Thirteen self-contained lessons, with optional deeper extensions': '十三节独立课程，附可选深入拓展',
    'Thirteen chapters. Many ways in.': '十三章内容，多种学习入口。',
    'Each lesson opens with an optical problem and why solving it matters. Check the prerequisites, watch the companion film where available, then use controls that change the actual model and connect the result to a few equations. The Python lab then rebuilds the calculation, tests it independently, and asks you to explain what the model leaves out.': '每节课从一个光学问题及其意义出发。检查预备知识，观看已有的配套影片，再通过控件改变实际模型，把结果与公式连接起来。Python 实验随后重建计算，进行独立检验，并要求你说明模型忽略了什么。',
}
for en, zh in RUNTIME.items(): tr(en, zh)

PLOT_TRANSLATIONS = {
'A fixed-power focal pattern':'固定功率下的焦面图样',
'Spatial frequency (cycles / pupil half-width)':'空间频率（周期 / 瞳半宽）',
'Fraction of total power per sample':'每个采样点的总功率占比',
'Target intensity':'目标强度', 'Initial intensity':'初始强度', 'Final intensity':'最终强度',
'The curves share one power scale. The plot shows the central frequencies; the loss uses every detector sample.':'各曲线共用一个功率尺度。图中显示中心频率范围；损失计算使用全部探测器采样点。',
'Phase is the adjustable optical element':'相位是可调的光学元件',
'Normalized pupil position x':'归一化瞳坐标 x', 'Phase (rad)':'相位（rad）',
'Initial phase':'初始相位', 'Final phase':'最终相位',
'Only the illuminated pupil is drawn. The aperture and its amplitude remain fixed.':'仅绘制受照明的瞳面。孔径与振幅保持固定。',
'Two ways to check the same local derivative':'用两种方法检验同一个局部导数',
'Parameter: 0 = tilt a, 1 = defocus b':'参数：0 = 倾斜 a，1 = 离焦 b',
'Loss derivative (per radian)':'损失导数（每弧度）', 'Reverse-mode gradient':'反向梯度', 'Central finite difference':'中心有限差分',
'Both derivatives are evaluated at the initial design, before any update. Agreement checks the derivative of this numerical model.':'两种导数都在更新之前的初始设计处计算。吻合说明此数值模型的导数通过了检验。',
'Recompute the loss after every update':'每次更新后重新计算损失',
'Accepted gradient step':'被接受的梯度更新步数', 'Target-normalized intensity loss L':'按目标归一化的强度损失 L',
'Same target and objective':'相同目标图样与损失函数',
'Backtracking accepts only a decrease in this loss. A plateau may reflect a local stationary point; it is not a global-optimum certificate.':'回溯只接受降低当前损失的更新。平台可能来自局部驻点，并不证明达到全局最优。',
'Loss: initial → final':'损失：初始 → 最终',
'Gradient check: maximum absolute error':'梯度检验：最大绝对误差',
'Central difference with h = 10⁻⁵ rad at the initial design.':'在初始设计处，使用 h = 10⁻⁵ rad 的中心差分。',
'Final tilt a / defocus b':'最终倾斜 a / 离焦 b',
'Total detected / incident power':'总探测功率 / 入射功率',
'The full discrete Fourier plane retains unit total power.':'完整离散傅里叶平面保持单位总功率。',
'Accepted steps':'被接受的步数',
'The model uses explicit reverse differentiation, not an autodiff library. Compare the same target and fixed power; a lower sampled loss does not establish uniqueness or fabricated performance.':'本模型显式计算反向导数，不使用自动微分库。请在相同目标与固定功率下比较；采样损失更低并不证明解的唯一性或实物性能。',
}
for en, zh in PLOT_TRANSLATIONS.items(): tr(en, zh)

for m in MODULES:
    for title, _, _ in m['exercises']:
        tr(title + '.', TRANSLATIONS[title] + '。')
tr('End-to-end design', '端到端设计')

tr('Design parameter', '设计参数')
tr('Tilt a', '倾斜 a')
tr('Defocus b', '离焦 b')
