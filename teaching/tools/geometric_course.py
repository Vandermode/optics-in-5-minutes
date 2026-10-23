"""Authored geometric-optics lessons with paired English/Chinese text.

The numerical implementations live in ray_optics.py and shared/ray-optics.js.
Keep the required experiment, outcomes and assessment on the same learning path.
"""
TRANSLATIONS = {}


def tr(en, zh):
    TRANSLATIONS[" ".join(en.split())] = zh
    return en


def p(en, zh):
    return '<p>' + tr(en, zh) + '</p>'


def h(en, zh):
    return '<h3>' + tr(en, zh) + '</h3>'


def eq(text):
    return '<div class="equation">' + text + '</div>'


def slider(id, label, zh, low, high, step, unit='', digits=0, help='', help_zh=''):
    return dict(id=id, type='range', label=tr(label, zh), min=low, max=high,
                step=step, unit=unit, digits=digits, help=tr(help, help_zh) if help else '')


def question(prompt, zh, choices, answers_zh, answer, explanation, explanation_zh):
    return dict(prompt=tr(prompt, zh), choices=[tr(a, b) for a, b in zip(choices, answers_zh)],
                answer=answer, explanation=tr(explanation, explanation_zh))


def exercise(title, title_zh, task, task_zh, answer, answer_zh):
    return (tr(title, title_zh), tr(task, task_zh), tr(answer, answer_zh))


MIT = 'https://ocw.mit.edu/courses/2-71-optics-spring-2009/pages/lecture-slides/'

MODULES = [dict(
    id='E11', slug='e11-ray-tracing',
    title=tr('Ray tracing through a real lens', '穿过真实曲面的光线追迹'),
    headline=tr('Find the surface.', '找到曲面。') + '<br><em>' + tr('Then bend the ray.', '再让光线转向。') + '</em>',
    deck=tr('Follow light through two curved interfaces, compare a paraxial focus with finite-aperture rays, and separate geometric blur from diffraction.',
            '逐面追踪光线穿过两个曲面，比较近轴焦点与有限孔径光束，并区分几何模糊与衍射。'),
    prerequisites=['E03'], guided_lab=True,
    goals=[
        tr('Find a forward surface intersection and apply Snell’s law using the local normal.', '求出前方曲面交点，并相对局部法线应用 斯涅尔定律。'),
        tr('Use a ray fan to test when a paraxial focal prediction stops being accurate.', '用光线扇检验近轴焦点预测何时开始偏离实际追迹结果。'),
        tr('Explain why a geometric spot diagram does not establish diffraction-limited resolution.', '解释为什么几何点列图不能单独证明成像已经达到衍射极限。'),
    ],
    defaults=dict(aperture=6, field=0, image=53.3, index=1.5),
    controls=[
        slider('aperture', 'Entrance aperture radius', '入口孔径半径', 1, 10, .5, 'mm', 1,
               'Keep this fixed when comparing designs. The lens has a separate mechanical clear radius.', '比较设计时固定此值。透镜曲面另有独立的机械通光半径。'),
        slider('field', 'Object field angle', '物方视场角', 0, 8, .5, '°', 1,
               'Each object point at infinity supplies one parallel bundle; changing angle selects another point.', '无穷远处的每个物点对应一束平行光；改变角度相当于选择另一个物点。'),
        slider('image', 'Image plane z', '像面位置 z', 30, 90, .1, 'mm', 1,
               'Measured from the first lens vertex, not from a principal plane or the rear vertex.', '从透镜第一顶点测量，不是从主平面或后顶点测量。'),
        slider('index', 'Glass refractive index', '玻璃折射率', 1.3, 1.8, .05, '', 2,
               'A real, constant index in a monochromatic model. This control does not model material dispersion.', '单色模型中的实数常量折射率。此控件并不模拟材料色散。'),
    ],
    presets=[
        dict(label=tr('Narrow on-axis bundle', '细轴上光束'), values=dict(aperture=1, field=0, image=53.3)),
        dict(label=tr('Open the aperture', '打开孔径'), values=dict(aperture=8, field=0, image=53.3)),
        dict(label=tr('Look off axis', '观察离轴视场'), values=dict(aperture=6, field=6, image=53.3)),
    ],
    prediction=tr('A thin-lens sketch puts all parallel rays at one point. With the same spherical lens and image plane, will doubling the entrance radius keep that point sharp? Predict what the outer rays will do before opening the aperture.',
                  '薄透镜示意图把所有平行光线汇到一点。保持同一个球面透镜和像面，入口半径加倍后，这一点仍会清晰吗？先预测边缘光线的表现，再打开孔径。'),
    intro=(
        p('Start with the computed rays above. A ray records a position and a direction. In a uniform transparent medium it travels along a straight line; at a curved interface its direction changes according to the normal at the point it actually reaches.',
          '先看上方计算得到的光线路径。每条光线记录一个位置和一个方向。在均匀透明介质中，它沿直线前进；到达曲面时，它根据实际交点处的法线改变方向。')
        + eq('r(t) = r₀ + t d, &nbsp; |d| = 1, &nbsp; t &gt; 0')
        + p('The algorithm has two separate questions: where does this ray meet the next surface, and which way does the transmitted ray leave? Solve the intersection first. A spherical surface has two algebraic intersections, but only the forward intersection on the specified surface cap belongs to this lens.',
            '算法分成两个问题：光线在哪里遇到下一个曲面，以及透射光线朝哪个方向离开。先求交点。球面通常有两个代数交点，但只有位于前方、且属于指定曲面分支的交点才是这片透镜的表面。')
        + eq('n₁ sin θ₁ = n₂ sin θ₂')
        + p('Both angles are measured from the local surface normal, not the optical axis. Refraction preserves the tangential component of the wavevector. The same rule works for a ray away from the meridional plane when expressed with vectors.',
            '两个角都相对局部曲面法线测量，而不是相对光轴。折射保持波矢的切向分量。写成向量形式后，同一规则也适用于不在子午面内的光线。')
        + p('Here a collimated bundle enters a spherical glass singlet in air: R₁ = +50 mm, R₂ = −50 mm and centre thickness 4 mm. The refractive index defaults to n = 1.5. A positive radius places the sphere centre after its vertex along +z. The entrance stop is fixed at z = −5 mm; the image coordinate is always measured from the first vertex.',
            '本例中，平行光束进入空气中的球面单透镜：R₁ = +50 mm、R₂ = −50 mm、中心厚度为 4 mm，默认折射率为 n = 1.5。正半径表示球心位于顶点的 +z 方向。入射光阑固定于 z = −5 mm；像面坐标始终从第一顶点测量。')
    ),
    steps=[
        tr('Choose the narrow on-axis bundle. Keep n = 1.5 and compare its crossing with the paraxial image prediction. Move only the image plane until the geometric spot is smallest.',
           '选择细轴上光束，保持 n = 1.5，比较其交汇位置与近轴像面预测。只移动像面，找到几何光斑最小的位置。'),
        tr('Return to the same image plane and increase the aperture radius. Follow the outer rays and the transverse ray fan. Refocusing can reduce the spread but does not make every spherical-surface ray meet at one point.',
           '回到同一像面并增大孔径半径。观察边缘光线与横向光线扇。重新对焦可以减小散布，但无法让球面折射后的所有光线都汇到同一点。'),
        tr('Keep the aperture fixed and select an off-axis object point. Compare the ray diagram, spot centroid and spread separately. Explain which change is image position and which is aberration.',
           '固定孔径，选择离轴物点。分别比较光路、光斑质心和散布，说明哪些变化属于像的位置，哪些变化属于像差。'),
    ],
    theory=(
        h('A paraxial lens is the small-angle limit of this calculation.', '近轴透镜是这一计算的小角度极限。')
        + p('For small heights and angles, sin θ and tan θ can be replaced by θ. Surface refraction and propagation then become linear operations. A thick-lens calculation gives an effective focal length and a back focal distance; they are measured from different reference planes and need not be equal.',
            '高度和角度足够小时，可以用 θ 近似 sin θ 与 tan θ，曲面折射和传播便成为线性运算。厚透镜计算给出有效焦距与后焦距；它们从不同参考平面测量，因此不一定相等。')
        + eq('1/f = (n − 1)[1/R₁ − 1/R₂ + (n − 1)d/(n R₁R₂)]')
        + p('This lensmaker relation is paraxial, even though it includes centre thickness. The nonparaxial ray trace does not use it to bend each ray. Agreement as the entrance bundle becomes narrow is a useful independent check; disagreement at a wide aperture is often physical spherical aberration rather than a numerical bug.',
            '这个透镜制造者关系包含中心厚度，但仍属于近轴关系。非近轴光线追迹并不靠它来决定每条光线的折射。入口光束变细时两者趋于一致，是一个独立检验；大孔径时的偏离往往是真实球差，而非数值错误。')
        + h('A spot diagram records intersections, not a wave intensity.', '点列图记录的是交点，不是波动强度。')
        + p('The spot diagram places one dot where each sampled ray meets the image plane. Its centroid records image location; its root-mean-square radius records geometric spread about that centroid. The sampling and pupil weights must be stated before comparing numbers.',
            '点列图在每条采样光线与像面的交点处画一个点。质心记录像的位置，均方根半径记录相对质心的几何散布。比较数值之前，必须说明采样方式和瞳面权重。')
        + eq('σ² = Σ wⱼ |xⱼ − x̄|² / Σ wⱼ, &nbsp; x̄ = Σ wⱼ xⱼ / Σ wⱼ')
        + p('A smaller aperture can reduce geometric aberration while increasing diffraction blur. This model has no wavelength-dependent interference, so a zero geometric spot is not a zero-width point-spread function. E03 supplies the wave-optical resolution question that the ray model leaves open.',
            '缩小孔径可以减小几何像差，同时也会增大衍射模糊。本模型没有随波长变化的干涉过程，因此零尺寸几何光斑并不意味着零宽度点扩散函数。光线模型没有回答的波动分辨率问题，由 E03 接续。')
        + h('A missing transmitted ray is a result to report.', '没有透射光线，也是一项需要报告的结果。')
        + p('A ray may miss a surface cap, hit outside the mechanical clear aperture, or undergo total internal reflection. The tracer reports these cases instead of silently including them in the spot. Fresnel power splitting, absorption, scattering and multiple reflected paths are outside this sequential transmitted-ray model.',
            '光线可能没有击中指定曲面、超出机械通光孔径，或者发生全内反射。追迹器会报告这些情况，而不是悄悄把它们混入光斑统计。本模型只追踪顺序透射光线，不包含菲涅耳功率分配、吸收、散射和多次反射路径。')
    ),
    extension=tr('Compare the aperture dependence of the best on-axis geometric RMS radius with an approximate Airy first-zero radius at 550 nm. These are different size measures; do not add them or interpret their crossing as a complete diffraction prediction.',
                 '比较最佳轴上几何 RMS 半径与 550 nm 下近似艾里第一暗环半径随孔径的变化。它们是不同的尺寸度量，不应直接相加，也不能把交点解释为完整的衍射预测。'),
    limits=tr('Sequential transmitted rays through two spherical or plane surfaces, with a real constant glass index and an object at infinity. The mechanical clear radius is 12 mm. No diffraction, polarization, Fresnel power weights, material dispersion or surface roughness is modeled. A geometric RMS spot is not a PSF or an MTF.',
              '对无穷远物体穿过两个球面或平面的顺序透射光线进行追迹，玻璃折射率为实数常量，机械通光半径为 12 mm。不模拟衍射、偏振、菲涅耳功率权重、材料色散或表面粗糙度。几何 RMS 光斑不是 PSF，也不是 MTF。'),
    quiz=[
        question('At a curved refracting surface, the incidence angle is measured from…', '在弯曲折射面上，入射角相对什么测量？',
                 ['The optical axis everywhere', 'The local normal at the ray intersection', 'The line joining the lens vertices'], ['各处都相对光轴', '光线交点处的局部法线', '连接透镜顶点的直线'], 1,
                 'The normal changes across a curved surface. Using the optical axis instead changes the actual refraction problem.', '曲面各处的法线不同。用光轴代替法线，会改变实际的折射问题。'),
        question('A small-angle focus agrees with a narrow ray bundle but not a wide one. The first explanation to test is…', '小角度焦点与细光束吻合，却与宽光束不吻合。首先应检验什么解释？',
                 ['The paraxial approximation no longer describes the outer rays', 'Snell’s law stops applying to the outer surface', 'A ray diagram has already included diffraction'], ['近轴近似已不足以描述边缘光线', '斯涅尔定律在曲面边缘不再适用', '光线图已经包含了衍射'], 0,
                 'The exact geometric surface trace still obeys Snell’s law. Its finite-angle departure from the paraxial prediction is an aberration.', '非近轴几何曲面追迹仍遵守 斯涅尔定律。有限角度结果偏离近轴预测，体现了像差。'),
        question('Reducing the geometric RMS spot proves the image resolution improved only after…', '几何 RMS 光斑变小，要据此判断图像分辨率改善，还需要什么？',
                 ['Comparing diffraction, sampling and the same imaging conditions', 'Rescaling each plot to its own maximum', 'Discarding rays far from the centroid'], ['比较衍射、采样及相同成像条件', '把每张图分别归一到自己的最大值', '丢弃离质心较远的光线'], 0,
                 'A geometric spot omits diffraction, and removing rays changes the experiment. Compare the same pupil and complete model assumptions.', '几何光斑忽略衍射，移除光线则改变了实验。应比较相同瞳面及完整的模型假设。'),
    ],
    exercises=[
        exercise('Follow one ray', '追踪一条光线', 'Report both intersections, the local incidence angles and the two refracted directions. Check Snell’s law numerically.', '报告两个曲面交点、局部入射角和两次折射方向，并数值检验 斯涅尔定律。',
                 'Use the sphere centre to construct each normal. Check forward distance, cap membership and the unit length of every direction.', '用球心构造每个法线，检查前向距离、曲面分支归属，以及每个方向向量的单位长度。'),
        exercise('Test the paraxial limit', '检验近轴极限', 'Reduce the entrance radius at fixed index and compare the traced on-axis focus with the independent thick-lens prediction.', '固定折射率并减小入口半径，比较追迹得到的轴上焦点与独立厚透镜预测。',
                 'The narrow-bundle limit approaches the paraxial back focal distance. Report the image coordinate from the first vertex separately from effective focal length.', '细光束极限趋近近轴后焦距。应将从第一顶点测量的像面坐标与有效焦距分别报告。'),
        exercise('Separate image position and blur', '区分像的位置与模糊', 'At two field angles, report the centroid and RMS radius with the same image plane and pupil sampling.', '在两个视场角下，使用相同像面和瞳面采样，分别报告质心与 RMS 半径。',
                 'Centroid motion locates the image; spread about that centroid is geometric blur. Neither number alone measures diffraction-limited resolution.', '质心移动描述像的位置，相对质心的散布描述几何模糊。两者都不能单独衡量衍射极限分辨率。'),
    ],
    refs=[(tr('MIT OCW · Refraction, paraxial imaging and thick lenses (lectures 2–5)', 'MIT OCW · 折射、近轴成像与厚透镜（第 2–5 讲）'), MIT)],
)]

MODULES.append(dict(
    id='E12', slug='e12-lens-design',
    title=tr('Lens design with a measurable objective', '用可测量的目标设计透镜'),
    headline=tr('Keep the task fixed.', '固定成像任务。') + '<br><em>' + tr('Improve the lens.', '改进透镜。') + '</em>',
    deck=tr('Bend a singlet without changing its paraxial focal power. Choose a common image plane, minimize a stated geometric error, and check what the design sacrifices away from the centre.',
            '在保持近轴光焦度的条件下改变单透镜形状。选择公共像面，减小明确的几何误差，并检查设计在中心视场之外付出了什么代价。'),
    prerequisites=['E11', 'E03'], guided_lab=True, worker=True,
    goals=[
        tr('Separate a lens prescription, a performance metric and the constraints that keep the task unchanged.', '区分透镜结构参数、性能指标，以及保证任务不被改变的约束。'),
        tr('Compare on-axis and multi-field optimization using one common image plane and the same entrance aperture.', '使用同一个公共像面和相同入口孔径，比较轴上优化与多视场优化。'),
        tr('Audit an improved design with denser pupil sampling and explain the image-quality questions its merit function omits.', '用更密的瞳面采样检验改进后的设计，并说明评价函数没有回答哪些成像质量问题。'),
    ],
    defaults=dict(shape=0, aperture=6, objective='field', iterations=24),
    controls=[
        slider('shape', 'Initial lens bending q', '初始透镜弯曲参数 q', -1.2, 1.2, .1, '', 1,
               'Changes the balance of the two curvatures while preserving the specified paraxial focal power.', '改变两个曲率之间的分配，同时保持指定的近轴光焦度。'),
        slider('aperture', 'Entrance aperture radius', '入口孔径半径', 2, 8, .5, 'mm', 1,
               'Fixed during each optimization. Compare runs at the same radius before attributing an improvement to lens shape.', '每次优化过程中保持不变。比较运行结果时，也应使用相同半径，才能把改善归因于透镜形状。'),
        dict(id='objective', type='select', label=tr('Design objective', '设计目标'), help=tr('All fields are evaluated after the search, even when only the central field was optimized.', '即使只优化中心视场，搜索后也会评价所有视场。'),
             options=[dict(value='axis', label=tr('On axis only: 0°', '仅轴上：0°')),
                      dict(value='field', label=tr('Equal weights: 0°, 3°, 6°', '等权重：0°、3°、6°'))]),
        slider('iterations', 'Search refinement steps', '搜索细化步数', 8, 48, 4, '', 0,
               'More steps refine this bounded search; they do not certify a global optimum or a complete optical design.', '更多步数会细化这次有界搜索，并不能证明找到了全局最优或完成了全部光学设计。'),
    ],
    presets=[
        dict(label=tr('Optimize the central field', '优化中心视场'), values=dict(shape=0, aperture=6, objective='axis', iterations=24)),
        dict(label=tr('Include the edge fields', '把边缘视场纳入目标'), values=dict(shape=0, aperture=6, objective='field', iterations=24)),
        dict(label=tr('Try a different starting shape', '尝试另一种初始形状'), values=dict(shape=-.8, aperture=6, objective='field', iterations=24)),
    ],
    prediction=tr('Two lenses have the same paraxial focal length and collect the same entrance bundle. Must they form the same image? Predict whether minimizing the central spot alone will also give the best result at 6°.',
                  '两片透镜具有相同近轴焦距，也接收相同的入口光束，它们一定形成相同的像吗？先预测：只让中心光斑最小，是否也会给出 6° 视场的最佳结果？'),
    intro=(
        p('E11 uses ray tracing to predict where a chosen lens sends each ray. Inverse design turns that calculation into a decision: which permitted prescription makes a specified image metric better? A search algorithm cannot decide what better means until the objective and design constraints are written down.',
          'E11 用光线追迹预测选定透镜会把每条光线送到哪里。逆向设计则把这个计算变成决策：在允许的结构参数中，哪一组能改善指定的成像指标？在写清目标与设计约束之前，搜索算法并不知道什么叫更好。')
        + p('This lesson keeps the model deliberately small. A singlet in air has index n = 1.5, centre thickness d = 4 mm and paraxial effective focal length f = 50 mm. We vary the two surface curvatures together and choose one image plane for the whole field. The entrance stop stays at z = −5 mm. Glass type, element count and aperture are not optimized.',
            '本课使用一个规模明确的小模型：空气中的单透镜，折射率 n = 1.5、中心厚度 d = 4 mm、近轴有效焦距 f = 50 mm。我们联动调整两个表面曲率，并为整个视场选择同一个像面。入射光阑保持在 z = −5 mm。玻璃类型、镜片数量和孔径不参与优化。')
        + eq('c₁ = 1/R₁, &nbsp; c₂ = 1/R₂, &nbsp; q = (c₁ + c₂)/(c₁ − c₂)')
        + p('The bending parameter q changes how the refracting power is shared between the front and rear surfaces. The curvatures are solved together so the thick-lens focal power remains fixed. The back focal distance can still move: a fixed effective focal length does not fix the principal planes.',
            '弯曲参数 q 改变前后两个表面对折射能力的分担。两个曲率一起求解，使厚透镜光焦度保持不变。后焦距仍可移动，因为固定有效焦距并不等于固定主平面位置。')
        + p('For each object field angle we trace a weighted pupil bundle to the same image plane. The centroid gives the image location. The mean squared distance about that centroid is the geometric spot variance; its square root is the RMS spot radius. The multi-field objective averages the variance over 0°, 3° and 6° with equal field weights.',
            '对每个物方视场角，我们将加权瞳面光束追迹到同一个像面。质心给出像的位置，相对质心的平均平方距离给出几何光斑方差；其平方根就是 RMS 光斑半径。多视场目标对 0°、3° 和 6° 的光斑方差作等权平均。')
        + eq('J = Σₖ Wₖ σₖ² / Σₖ Wₖ, &nbsp; σₖ = RMS radius about field k’s centroid')
    ),
    steps=[
        tr('Run the central-field design. Compare the initial and optimized prescriptions, then inspect all three field RMS values, not only the loss used by the search.',
           '运行中心视场设计。比较初始与优化后的结构参数，然后检查三个视场的 RMS 数值，而不只看搜索使用的损失值。'),
        tr('Keep the aperture and starting shape unchanged, switch to equal field weights, and run again. Identify the central-field and edge-field tradeoff at the single common image plane.',
           '保持孔径与初始形状不变，切换到等权视场目标后再次运行。观察在唯一公共像面上，中心视场与边缘视场之间的取舍。'),
        tr('Try a different starting shape and increase the search refinement. Inspect transmission and edge clearance as well as spot size, then repeat the result with denser pupil sampling in Python.',
           '更换初始形状并增加搜索细化步数。除了光斑尺寸，还要检查光线通过率与镜片边缘间隔，再在 Python 中用更密的瞳面采样复核结果。'),
    ],
    theory=(
        h('A prescription says what may change.', '结构参数说明哪些量可以改变。')
        + p('The ray tracer reads surface radii, axial positions, index and clear apertures. The design parameterization must turn every trial into a valid prescription. Here the focal-power constraint is built into that parameterization; positive edge thickness and complete tracing are checked for every trial.',
            '追迹器读取曲面半径、轴向位置、折射率和通光孔径。设计参数化必须把每次试探转换成有效结构。本例将光焦度约束直接写入参数化，并为每次试探检查正的边缘厚度与完整光线追迹。')
        + eq('(50 mm)⁻¹ = (n − 1)[c₁ − c₂ + (n − 1)d c₁c₂/n]')
        + p('For a fixed prescription, moving the image plane changes the transverse intercepts of all rays. Refocusing is therefore part of this design calculation. Every evaluated field must use the same plane; independently refocusing each field would hide field curvature and evaluate a different instrument.',
            '对固定结构，移动像面会改变所有光线的横向交点，因此重新对焦是本设计计算的一部分。每个被评价的视场必须使用同一像面；若各视场独立对焦，就会掩盖场曲，并相当于评价了另一种仪器。')
        + h('The objective decides which compromise the search prefers.', '目标函数决定搜索偏好哪种取舍。')
        + p('An on-axis objective gives no direct reward for an improved edge field. Including several fields changes the question the search answers. Averaging squared spot radii also differs from minimizing the worst field; changing these weights is a design decision, not a plotting option.',
            '轴上目标不会直接奖励边缘视场的改善。加入多个视场，会改变搜索所回答的问题。平均光斑半径的平方，也不同于最小化最差视场；改变权重是设计决策，不只是绘图选项。')
        + p('This RMS metric subtracts each field’s own centroid. It therefore measures geometric spread without penalizing the image’s displacement from an ideal height. Distortion, relative illumination, chromatic performance and modulation transfer require additional measurements or objectives.',
            '此 RMS 指标减去了各视场自己的质心，所以它衡量几何散布，却不会惩罚像的位置偏离理想像高。畸变、相对照度、色差和调制传递，需要额外的测量或目标。')
        + h('A search must not improve the score by changing the task.', '搜索不能通过改变任务来改善分数。')
        + p('The entrance aperture stays fixed throughout a run. Rays that miss an allowed surface or fail to transmit are counted as failures and penalized, not silently removed before computing a smaller spot. The lens surfaces must retain physical clearance over the mechanical aperture.',
            '每次运行中，入口孔径保持不变。未击中允许曲面或无法透射的光线会计入失败并受到惩罚，而不是在计算更小光斑前悄悄移除。透镜在机械孔径内还必须保持真实的曲面间隔。')
        + p('The browser uses a bounded numerical search in this two-parameter family, with one common refocus. E10 instead demonstrates adjoint derivatives for a phase-pupil wave model. The shared idea is the forward-model–objective–update loop; more complex optical systems may need different derivatives and search methods.',
            '浏览器在这个两参数族中进行有界数值搜索，并统一重新对焦。E10 则用相位瞳面的波动模型演示伴随导数。两者共享的是前向模型—目标函数—更新这一流程；更复杂的光学系统可能需要不同的导数与搜索方法。')
        + h('Validate beyond the samples that chose the design.', '用选择设计时未覆盖的采样来验证。')
        + p('Re-evaluate the final prescription with more pupil rays and intermediate field angles. Report the common image plane, field weights, ray count and failed-ray fraction. A lower sampled merit establishes improvement for that model and objective; it does not establish a global optimum, a manufacturable camera or a diffraction PSF.',
            '用更多瞳面光线和中间视场角重新评价最终结构。报告公共像面、视场权重、光线数与失败比例。采样评价函数下降，说明该模型和目标下取得了改善；它并不证明全局最优、可制造相机或某种衍射 PSF。')
    ),
    extension=tr('Apply seeded perturbations to the surface radii, centre thickness and refractive index, then re-evaluate the same fields at the unchanged nominal image plane. This tests sensitivity under assumed variations, not manufacturing yield.',
                 '对曲面半径、中心厚度和折射率施加固定随机种子的扰动，再在不变的标称像面上重新评价相同视场。这检验的是假定变化下的敏感性，并非制造良率。'),
    limits=tr('Monochromatic sequential geometric design of a single spherical lens in air: n = 1.5, centre thickness 4 mm and paraxial EFL 50 mm. Only bending and a common image position are optimized at fixed entrance aperture. RMS is measured about each field centroid. Diffraction, chromatic aberration, distortion targets, polarization, Fresnel power weights and manufacturing tolerances are not optimized.',
              '空气中单片球面透镜的单色顺序几何设计：n = 1.5、中心厚度 4 mm、近轴有效焦距 50 mm。固定入口孔径，仅优化弯曲参数和公共像面位置。RMS 相对各视场质心计算。优化不包含衍射、色差、畸变目标、偏振、菲涅耳功率权重或制造公差。'),
    quiz=[
        question('Two prescriptions have the same effective focal length. Their finite-aperture spots…', '两组结构具有相同有效焦距，它们的有限孔径光斑会怎样？',
                 ['Must be identical', 'Can differ because bending changes finite-angle refraction', 'Must be diffraction limited'], ['必然相同', '可能不同，因为弯曲改变了有限角度折射', '必然达到衍射极限'], 1,
                 'Paraxial power constrains the small-angle response. It does not fix all higher-order ray deviations or the back focal distance.', '近轴光焦度约束小角度响应，并不固定所有高阶光线偏差或后焦距。'),
        question('Why must every field use the same image plane in this comparison?', '为什么这次比较中的所有视场必须使用同一个像面？',
                 ['Because Snell’s law only works on axis', 'Because the modeled camera has one sensor plane', 'Because each field has the same spot centroid'], ['因为 斯涅尔定律只在轴上成立', '因为所模拟的相机只有一个传感器平面', '因为每个视场的光斑质心都相同'], 1,
                 'Separately refocusing every field hides a tradeoff that one fixed sensor would still see.', '为每个视场单独对焦，会隐藏一个固定传感器仍然需要面对的取舍。'),
        question('A design has lower centroid-based RMS at 0°, 3° and 6°. What is still untested?', '一个设计在 0°、3° 和 6° 的质心 RMS 更低，什么仍未被检验？',
                 ['Whether its sampled geometric spots became smaller', 'Its distortion, diffraction and behaviour between the sampled fields', 'Whether the objective was a scalar'], ['采样几何光斑是否变小', '它的畸变、衍射及采样视场之间的表现', '目标函数是否为标量'], 1,
                 'The metric rewards only the quantities and configurations it contains. Validate omitted conditions explicitly.', '指标只奖励其中包含的量与配置。应明确验证被省略的条件。'),
    ],
    exercises=[
        exercise('State the design contract', '写清设计约定', 'List the variables, fixed focal power, entrance aperture, field weights, common image plane and failure penalties.', '列出变量、固定光焦度、入口孔径、视场权重、公共像面和失败惩罚。',
                 'A reproducible design includes both the prescription and the evaluation conditions. Fixed aperture and ray-failure checks prevent a misleading improvement.', '可复现的设计同时包含结构参数与评价条件。固定孔径和光线失败检查，可以避免误导性的改善。'),
        exercise('Compare two objectives', '比较两个目标', 'Optimize the on-axis and multi-field objectives from the same start. Report all field RMS values and the common image location for each result.', '从相同起点分别优化轴上与多视场目标，报告每个结果的全部视场 RMS 和公共像面位置。',
                 'Changing field weights changes the design problem. Judge each result against the same full field table rather than comparing two unlike scalar losses.', '改变视场权重就改变了设计问题。应依据同一张完整视场表评价结果，而不是直接比较两个定义不同的标量损失。'),
        exercise('Validate the final prescription', '验证最终结构', 'Re-trace the chosen lens with denser pupil sampling and an intermediate field. Report any change in the ranking and the transmitted-ray fraction.', '用更密瞳面采样和一个中间视场重新追迹选定透镜，报告设计排名是否变化以及透射光线比例。',
                 'Independent sampling checks whether the selected solution exploits the evaluation grid. It complements, rather than replaces, a more complete wave and tolerance model.', '独立采样用于检验所选方案是否利用了评价网格的局限。它能补充更完整的波动和公差模型，但不能替代后者。'),
    ],
    refs=[(tr('MIT OCW · Aberrations and optical design (lectures 8–9)', 'MIT OCW · 像差与光学设计（第 8–9 讲）'), MIT)],
))

MOTIVATIONS = {
    'E11': dict(
        question=tr('Why do the edge rays miss the focus?', '为什么边缘光线会错过焦点？'),
        problem=tr('A thin-lens sketch predicts one focus, but a real spherical lens has thickness and accepts rays over a finite aperture. Before changing its shape, you need a model that follows where those rays actually go and explains which approximation caused the mismatch.',
                   '薄透镜示意图预测一个焦点，但真实球面透镜有厚度，也会接收有限孔径内的光线。改变形状之前，需要一个能追踪这些光线实际去向的模型，并弄清偏差来自哪种近似。'),
        payoff=tr('Build a surface-by-surface ray tracer, locate a practical image plane and identify the geometric image errors that motivate lens design.',
                  '建立逐面光线追迹器，找到实际像面，并识别促使我们进一步设计透镜的几何成像误差。')),
    'E12': dict(
        question=tr('Which lens is better for the image you need?', '哪一片透镜更适合你需要的图像？'),
        problem=tr('Two lenses can have the same focal length yet form different finite-aperture spots. Improving the centre may worsen the edge. A useful design process must state the required field, pupil and image plane before it changes the surface shapes.',
                   '两片透镜可以具有相同焦距，却形成不同的有限孔径光斑。改善中心可能使边缘变差。一个有用的设计过程必须先明确所需视场、瞳面和像面，再改变曲面形状。'),
        payoff=tr('Turn an imaging requirement into a reproducible objective, optimize within physical constraints and explain the tradeoffs that remain after the score improves.',
                  '把成像要求转化为可复现的目标，在物理约束下优化，并解释分数改善后仍然存在的取舍。')),
}

TOPICS = {
    'ray-vectors': dict(
        title=tr('Vectors, angles and surface normals', '向量、角度与曲面法线'),
        ready=tr('Normalize a direction vector, use a dot product to find an angle, and identify the normal to a sphere at a point.', '归一化方向向量，用点积求角度，并确定球面上一点的法线。'),
        source=tr('MIT OCW · Reflection and refraction, lecture 2', 'MIT OCW · 反射与折射，第 2 讲'), url=MIT,
        focus=tr('Use lecture 2 to identify incident rays and interface normals. Review vector lengths and dot products before translating the picture into code.', '阅读第 2 讲，识别入射光线与界面法线。把图像写成代码之前，复习向量长度与点积。')),
    'ray-geometry': dict(
        title=tr('Ray states and lens reference planes', '光线状态与透镜参考平面'),
        ready=tr('Follow a position and direction through successive surfaces, and distinguish effective focal length from a vertex-to-focus distance.', '跟踪位置与方向如何依次通过曲面，并区分有效焦距和从顶点到焦点的距离。'),
        source=tr('MIT OCW · Thick lenses, lecture 5', 'MIT OCW · 厚透镜，第 5 讲'), url=MIT,
        focus=tr('Read the thick-lens and principal-plane diagrams. E11 develops the actual surface-intersection and refraction algorithm.', '阅读厚透镜与主平面示意图。E11 会具体展开曲面求交与折射算法。')),
    'squared-error': dict(
        title=tr('Weighted means and squared error', '加权平均与平方误差'),
        ready=tr('Compute a weighted centroid and the mean squared distance about it; explain how changing weights changes an objective.', '计算加权质心及相对质心的平均平方距离，并解释改变权重如何改变目标函数。'),
        source=tr('MIT OCW · Aberrations and optical design, lectures 8–9', 'MIT OCW · 像差与光学设计，第 8–9 讲'), url=MIT,
        focus=tr('Use the spot diagrams to separate image position from image spread. The lesson defines its own RMS metric and field weights explicitly.', '通过点列图区分像的位置与像的散布。本课会明确给出采用的 RMS 指标和视场权重。')),
}

PREPARATION = {
    'E11': dict(
        skills=[dict(topic='ray-vectors',
                     need=tr('Use a unit direction, a dot product and a quadratic surface equation.', '使用单位方向、点积与二次曲面方程。'),
                     use=tr('These identify the next intersection and the angle to its local normal.', '这些运算用于确定下一个交点及光线与局部法线的夹角。'))],
        notebook=tr('Use small real vectors and arrays of ray positions; keep millimetres and degrees separate from array indices.', '使用小型实向量和光线位置数组，将毫米、角度与数组下标分开。'),
        coding=['numpy'],
        new=tr('Build the sequential ray-tracing algorithm and compare nonparaxial geometric rays with the small-angle lens prediction.', '建立顺序光线追迹算法，比较非近轴几何光线与小角度透镜预测。')),
    'E12': dict(
        skills=[
            dict(topic='ray-geometry',
                 need=tr('Trace a prescribed lens and interpret an image-plane ray intersection.', '追迹一片已知结构的透镜，并解释像面上的光线交点。'),
                 use=tr('Every trial design is evaluated by the same forward ray model.', '每个试探设计都由同一个前向光线模型评价。')),
            dict(topic='squared-error',
                 need=tr('Distinguish the centroid of a spot from its spread and combine weighted errors.', '区分光斑质心与散布，并组合加权误差。'),
                 use=tr('The objective compares geometric spot variances across specified fields.', '目标函数比较指定视场中的几何光斑方差。')),
        ],
        notebook=tr('Evaluate a function on a parameter grid, record the best valid trial and recheck it with denser sampling.', '在参数网格上评价函数，记录最佳有效试探，并用更密采样复核。'),
        coding=['numpy'],
        new=tr('Define and optimize a constrained lens-design objective. Calculus-based optimization and an adjoint solver are not required for the core experiment.', '定义并优化受约束的透镜设计目标。核心实验不要求微积分优化方法或伴随求解器。')),
}
