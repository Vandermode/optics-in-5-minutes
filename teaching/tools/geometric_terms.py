"""Paired definitions and runtime labels for the geometric-optics lessons."""
import json
from geometric_course import tr, TRANSLATIONS, MODULES, PREPARATION

RUNTIME = {
    'Download the notebook and open it in Jupyter, or upload it to Colab using File → Upload notebook. Then select Runtime → Run all.': '下载 Notebook 后在 Jupyter 中打开，或在 Colab 中通过“文件 → 上传笔记本”导入，再选择“运行时 → 全部运行”。',
    'Lens surfaces': '透镜表面', 'Exact rays': '精确几何光线', 'Selected image plane': '选定像面',
    'Computed meridional rays': '计算所得的子午光线', 'Axial z from the first vertex (mm)': '相对第一顶点的轴向 z（mm）',
    'Transverse x (mm)': '横向 x（mm）',
    'The entrance stop is fixed at z = −5 mm. Curved surfaces and ray paths use the same physical scale.': '入射光阑固定于 z = −5 mm。曲面与光线路径采用相同的实际尺度。',
    'Transverse ray fan at the selected plane': '选定像面上的横向光线像差', 'Entrance-stop x (mm)': '入射光阑处 x（mm）',
    'Image x minus chief ray (μm)': '像面 x 相对主光线（μm）', 'Meridional ray error': '子午光线像差',
    'The chief ray passes through the centre of the entrance stop. The fan shows geometric aberration and defocus.': '主光线穿过入射光阑中心。光线像差曲线显示几何像差和离焦。',
    'Geometric spot relative to its centroid': '相对质心的几何光斑', 'Image x minus centroid (μm)': '像面 x 相对质心（μm）',
    'Image y minus centroid (μm)': '像面 y 相对质心（μm）', 'Equal-area pupil rays': '等面积瞳采样光线',
    'Each dot is one geometric ray, not an intensity PSF. RMS radius is measured about this field’s centroid.': '每个点代表一条几何光线，并非强度 PSF。RMS 半径相对当前视场质心计算。',
    'Focus scan for this field': '当前视场的调焦扫描', 'Common image-plane z (mm)': '共用像面 z（mm）',
    'Geometric RMS radius (μm)': '几何 RMS 半径（μm）', 'Exact ray bundle': '精确几何光束',
    'Paraxial on-axis plane': '近轴轴上像面', 'Best geometric plane': '最佳几何像面',
    'The paraxial plane is a small-angle prediction. The best geometric plane depends on aperture and field.': '近轴像面是小角度预测。最佳几何像面取决于孔径和视场。',
    'Paraxial effective focal length': '近轴有效焦距',
    'Effective focal length is not the distance from the rear vertex.': '有效焦距并非从后顶点量起的距离。',
    'Paraxial / best image-plane z': '近轴 / 最佳像面 z', 'Both positions are measured from the first surface vertex.': '两个位置均相对第一曲面顶点测量。',
    'Geometric RMS radius': '几何 RMS 半径', 'Centroid-relative radius of accepted rays; it does not include diffraction.': '被接受光线相对质心的半径，不包含衍射。',
    'Accepted / launched rays': '接受 / 发射光线数',
    'Image centroid x / y': '像面质心 x / y',
    'Absolute image coordinates before the spot drawing is recentered.': '点列图居中显示之前的绝对像面坐标。',
    'Blocked, missed, or totally internally reflected rays are counted as lost.': '被遮挡、未命中或发生全内反射的光线均计为丢失。',
    'Some launched rays are lost. Inspect their count before interpreting the spot of the surviving rays.': '部分发射光线已丢失。解释剩余光线形成的光斑前，请先检查丢失数量。',
    'Exact means geometric intersections and Snell refraction. This monochromatic model omits diffraction, dispersion, Fresnel reflection, and polarization.': '“精确”指几何求交和斯涅尔折射。本单色模型不包含衍射、色散、菲涅耳反射和偏振。',
    'Start and optimized geometric spots at 6°': '6° 视场下的初始与优化几何光斑', 'Starting prescription': '初始结构', 'Optimized prescription': '优化结构',
    'Both spots use the same aperture and scale. Each prescription has one image plane shared by every field; centroid removal only recentres the drawing.': '两个光斑采用相同孔径和显示尺度。每种结构的全部视场共用一个像面；扣除质心仅用于居中绘图。',
    'Field audit on one common plane': '共用像面上的视场复核', 'Object field angle (degrees)': '物方视场角（度）',
    'The audit uses 320 pupil rays per field. The 1.5° and 4.5° fields were not used in the design merit.': '复核对每个视场使用 320 条瞳采样光线。1.5° 与 4.5° 视场未参与设计评价。',
    'Accepted design steps': '被接受的设计步骤', 'Search iteration': '搜索迭代', 'Equal-field mean RMS² (mm²)': '等权视场平均 RMS²（mm²）',
    'Best design after each search step': '每一步搜索后的最佳设计',
    'Fixed-objective merit': '固定目标下的评价值',
    'The search changes bending and a common image plane. Aperture, thickness, index, mechanical radius, field weights, and paraxial focal power remain fixed.': '搜索改变弯曲形状和共用像面。孔径、厚度、折射率、机械半径、视场权重和近轴焦度保持固定。',
    'Fixed-power lens prescription': '固定焦度下的透镜结构',
    'Changing both curvatures redistributes aberration while preserving the 50 mm paraxial effective focal length.': '同时改变两个曲率可以重新分配像差，同时保持 50 mm 近轴有效焦距。',
    'Bending q: start → optimized': '弯曲因子 q：初始 → 优化', 'Common image-plane z': '共用像面 z',
    'Every evaluated field uses this same physical detector plane.': '每个被评价视场均使用同一个实际探测器平面。',
    'Merit RMS: start → optimized': '评价 RMS：初始 → 优化',
    'Square root of the selected equal-field mean squared spot radius.': '所选等权视场的光斑均方半径取平均后开方。',
    'Effective focal length / edge thickness': '有效焦距 / 边缘厚度',
    'The 24 mm mechanical diameter and minimum 0.5 mm edge thickness are fixed constraints.': '24 mm 机械直径和至少 0.5 mm 边缘厚度是固定约束。',
    'Audit lost rays': '复核丢失光线数', 'Any lost ray invalidates a candidate and receives a large merit penalty.': '丢失任意光线都会使候选设计无效，并受到较大评价惩罚。',
    'An on-axis objective can trade away off-axis quality. Inspect the separate field audit before choosing this prescription.': '轴上目标可能牺牲轴外质量。选择该结构前，请检查独立视场复核。',
    'This bounded search finds a candidate for one stated geometric merit. It does not establish a global optimum or predict diffraction-limited, broadband, or fabricated performance.': '有界搜索针对所定义的几何评价指标找到一个候选设计，并不能证明全局最优，也不能预测衍射极限、宽带或实际加工后的性能。',
    'Explore a connected map of optical concepts and choose from twelve interactive lessons with independent Python labs.': '浏览相互关联的光学概念图谱，从十二节互动课程及其独立 Python 实验中选择学习主题。',
    'Focused preparation for twelve optics lessons, with specific skills and selected primary learning resources.': '十二节光学课的针对性预备知识，包含具体技能要求和精选一手学习资源。',
    'Twelve self-contained lessons, with optional deeper extensions': '十二节独立课程，附可选深入拓展',
    'Twelve episodes. Many ways in.': '十二集内容，多种学习入口。',
    'Open the film ↗': '打开影片 ↗',
    'Ray Tracing Through a Real Lens | Optics in 5 Minutes': '光线穿过真实透镜 | 五分钟光学',
    'Designing a Lens | Optics in 5 Minutes': '设计一片透镜 | 五分钟光学',
    'Tracing Light Through a Lens | Optics in 5 Minutes': '追踪光线穿过透镜 | 五分钟光学',
    'Lens Design: Choose What Better Means | Optics in 5 Minutes': '透镜设计：定义何谓更好 | 五分钟光学',
    'Follow one ray.': '跟随一条光线。', 'Test the paraxial limit.': '检验近轴极限。',
    'Separate image position and blur.': '区分像的位置与模糊。', 'State the design contract.': '明确设计条件。',
    'Compare two objectives.': '比较两个目标。', 'Validate the final prescription.': '验证最终结构。',
    'Vectors, angles and surface normals.': '向量、角度与曲面法线。',
    'Ray states and lens reference planes.': '光线状态与透镜参考平面。',
    'Weighted means and squared error.': '加权平均与平方误差。',
}
for en, zh in RUNTIME.items():
    tr(en, zh)

# These strings are composed by the page renderer from already paired content.
for module in MODULES:
    episode, title = module['id'], module['title']
    translated = TRANSLATIONS[title]
    for suffix, zh in [('AN INTERACTIVE OPTICS LESSON', '互动光学课程'), ('FILM + LESSON + PYTHON LAB', '影片 + 课程 + PYTHON 实验'), ('THE FIVE-MINUTE FILM', '五分钟影片'), ('WHY LEARN THIS?', '为什么学这一课？')]:
        tr(f'{episode} / {suffix}', f'{episode} / {zh}')
    tr(f'BEFORE {episode} / PREREQUISITES', f'学习 {episode} 之前 / 预备知识')
    tr(f'{episode} · {title} · Optics in 5 Minutes', f'{episode} · {translated} · 五分钟光学')
    tr(f'OPTICS IN 5 MINUTES · {episode} / {title.upper()} ·', f'五分钟光学 · {episode} / {translated} ·')
    tr(f'Computed {title} illustration.', f'{translated}的计算示意图。')
    tr(f'watch the {episode} film above', f'观看上方 {episode} 影片')
    for i, quiz in enumerate(module['quiz'], 1):
        question = quiz['prompt']
        tr(f'{i}. {question}', f'{i}. {TRANSLATIONS[question]}')
    for skill in PREPARATION[episode]['skills']:
        used = skill['use']
        tr(f'Used here: {used}', f'本课用途：{TRANSLATIONS[used]}')
for duration, episode, title in [('5:02', 'E11', MODULES[0]['title']), ('5:08', 'E12', MODULES[1]['title'])]:
    tr(f'{duration} · {episode} · {title}', f'{duration} · {episode} · {TRANSLATIONS[title]}')
for label in list(TRANSLATIONS):
    if label.startswith('MIT OCW ·'):
        tr(label + ' ↗', TRANSLATIONS[label] + ' ↗')


def term(name, zh, aliases, definition, definition_zh, formula='', notation='', notation_zh='', episodes=None):
    return dict(name=tr(name, zh), aliases=aliases, definition=tr(definition, definition_zh),
                formula=formula, notation=tr(notation, notation_zh) if notation else '', episodes=episodes or ['E11', 'E12'])


TERMS = {
    'rayTracing': term('Ray tracing', '光线追迹', ['ray tracing', 'ray-tracing', 'geometric rays'],
        'Following a ray by intersecting each physical surface and updating its direction with a reflection or refraction law. This course traces the transmitted path through two surfaces; a ray has no diffraction width.',
        '逐面计算光线与实际曲面的交点，再按反射或折射定律更新方向。本课程追迹穿过两个曲面的透射路径；几何光线本身没有衍射宽度。',
        'r(t) = r₀ + t d̂;   t > 0', 'r₀ is the current position and d̂ is a unit direction. Select the next valid intersection on the physical surface cap.', 'r₀ 是当前位置，d̂ 是单位方向。选择实际曲面分支上的下一个有效交点。'),
    'paraxial': term('Paraxial approximation', '近轴近似', ['paraxial approximation', 'paraxial prediction', 'paraxial rays'],
        'A first-order description for rays close to the optical axis at small angles. It predicts focal power and image position but omits higher-order geometric aberrations.',
        '描述靠近光轴、角度较小的光线的一阶近似。它预测焦度和像面位置，但不包含高阶几何像差。',
        'sin θ ≈ θ;   tan θ ≈ θ', 'Angles are in radians. A finite-aperture bundle need not focus at the paraxial image plane.', '角度以弧度计。有限孔径光束的最佳聚焦平面不一定等于近轴像面。'),
    'sphericalAberration': term('Spherical aberration', '球差', ['spherical aberration'],
        'The on-axis geometric aberration in which rays at different pupil radii cross the optical axis at different axial positions. Moving the detector can reduce the spot but cannot make all those rays meet.',
        '不同瞳半径处的轴上光线在不同轴向位置与光轴相交的几何像差。移动探测器可以缩小光斑，但不能让所有光线交于一点。'),
    'spotDiagram': term('Geometric spot diagram', '几何点列图', ['geometric spot diagram', 'spot diagram', 'spot diagrams'],
        'A plot of ray intersections with a chosen detector plane. Its shape and spread reveal geometric aberration and defocus; equal-weight ray dots are not a diffraction intensity pattern.',
        '光线与选定探测器平面交点的图。形状与散布可揭示几何像差和离焦；等权重的光线点不等同于衍射强度图样。'),
    'lensPrescription': term('Lens prescription', '透镜结构参数', ['lens prescription', 'prescription', 'prescriptions'],
        'The numerical specification of optical surfaces, spacings, materials, apertures, and an image plane. The singlet here uses two signed radii, a centre thickness, and a real refractive index.',
        '对光学曲面、间隔、材料、孔径和像面的数值描述。本课单透镜采用两个带符号半径、一个中心厚度及一个实折射率。',
        'z_center = z_vertex + R', 'Positive R places the centre of curvature to the right of the vertex along the propagation axis. R = ∞ represents a plane.', 'R 为正时，曲率中心位于顶点沿传播轴的右侧。R = ∞ 表示平面。'),
    'lensBending': term('Lens bending', '透镜弯曲', ['lens bending', 'bending', 'shape factor'],
        'Redistributing optical power between the two surfaces while preserving the prescribed total paraxial power. Different bendings can have the same focal length but different aberrations.',
        '在保持指定总近轴焦度的同时，重新分配两个曲面的焦度。不同弯曲形状可以具有相同焦距，却产生不同像差。',
        'q = (c₁ + c₂) / (c₁ − c₂);   cᵢ = 1/Rᵢ', 'The model jointly solves both curvatures at fixed index, thickness and effective focal length; it does not hold one radius fixed.', '模型在固定折射率、厚度和有效焦距的条件下联合求解两个曲率，并非固定其中一个半径。', ['E12']),
    'meritFunction': term('Merit function', '评价函数', ['merit function', 'design merit'],
        'A scalar score specifying what counts as a better design. Here it averages centroid-relative squared spot radii at the selected fields, with fixed weights and a penalty for invalid geometry or lost rays.',
        '规定何种设计更好的标量评分。本课以固定视场权重平均相对质心的光斑均方半径，并惩罚无效几何结构或丢失光线。',
        'M = Σⱼ wⱼ RMSⱼ²;   Σⱼ wⱼ = 1', 'Every field uses one common image plane. Removing each centroid means this merit does not constrain distortion or absolute image position.', '每个视场共用一个像面。逐视场扣除质心意味着该指标不约束畸变或绝对像位置。', ['E12']),
    'rmsSpot': term('RMS spot radius', 'RMS 光斑半径', ['RMS spot radius', 'RMS radius', 'geometric RMS radius', 'root-mean-square radius'],
        'The root mean square distance of ray intersections from their centroid. It measures geometric spread for the chosen pupil samples, field and image plane; it is not a resolution threshold.',
        '光线交点到其质心距离的均方根。它度量给定瞳采样、视场和像面下的几何散布，并非分辨率阈值。',
        'RMS = √[Σᵢ |rᵢ − r̄|² / N]', 'N is the number of accepted equal-area pupil samples and r̄ is their centroid. Always report lost rays separately.', 'N 为被接受的等面积瞳采样数，r̄ 为其质心。应同时单独报告丢失光线数。'),
    'designConstraints': term('Design constraints', '设计约束', ['design constraints', 'fixed constraints'],
        'Conditions a candidate must satisfy throughout the search. This example fixes focal length, material index, centre thickness, aperture and stop location, and rejects insufficient edge thickness or lost rays.',
        '候选设计在搜索全过程中必须满足的条件。本例固定焦距、材料折射率、中心厚度、孔径和光阑位置，并拒绝边缘厚度不足或光线丢失的设计。', episodes=['E12']),
    'apertureStop': term('Entrance stop', '入射光阑', ['entrance stop', 'entrance-stop', 'stop radius'],
        'A physical opening limiting which incoming rays enter the system. This model places the opening in air before the first surface, so its pupil coordinates refer to one fixed plane as the lens shape changes.',
        '限制哪些入射光线能够进入系统的实际开口。本模型把开口放在第一曲面前的空气中，因此改变透镜形状时瞳坐标仍对应同一个固定平面。',
        'x² + y² ≤ a²;   z_stop = −5 mm', 'a is the stop radius. The lens mechanical radius remains 12 mm and is checked independently for clipping.', 'a 为光阑半径。透镜机械半径固定为 12 mm，其遮挡情况单独检查。'),
}

ALIASES = {
    'rayTracing': ['光线追迹', '光线追踪', '几何光线'], 'paraxial': ['近轴近似', '近轴光线'],
    'sphericalAberration': ['球差'], 'spotDiagram': ['几何点列图', '点列图'],
    'lensPrescription': ['透镜结构参数', '镜头处方'], 'lensBending': ['透镜弯曲', '形状因子'],
    'meritFunction': ['评价函数', '优值函数'], 'rmsSpot': ['RMS 光斑半径', '均方根光斑半径', 'RMS 半径', '均方根半径'],
    'designConstraints': ['设计约束'], 'apertureStop': ['入射光阑', '光阑半径'],
}

LESSON_TERMS = {
    'E11': {
        'rayTracing': tr('Intersect a physical surface, orient its normal, refract, and repeat before measuring the image-plane intersections.', '先与实际曲面求交，确定法线方向，折射后重复，再测量像面交点。'),
        'snell': tr('The vector form changes direction at both curved surfaces and explicitly detects total internal reflection.', '矢量形式在两个弯曲表面更新方向，并显式检测全内反射。'),
        'refractiveIndex': tr('Changing the glass index changes both refractions, focal power and aberrations.', '改变玻璃折射率会同时改变两次折射、焦度和像差。'),
        'paraxial': tr('Compare the small-angle image plane with the best plane found from the finite-aperture bundle.', '比较小角度像面与有限孔径光束算出的最佳像面。'),
        'sphericalAberration': tr('Open the stop on axis and compare marginal and near-axis ray crossings; then try refocusing.', '在轴上打开光阑，比较边缘光线与近轴光线的交轴位置，再尝试重新调焦。'),
        'spotDiagram': tr('Each plotted dot is a computed ray intersection, centred only to show the spot shape.', '每个点都是计算所得的光线交点；居中显示仅用于观察光斑形状。'),
        'rmsSpot': tr('The reported radius is relative to this field’s centroid and excludes diffraction.', '报告的半径相对于当前视场的质心计算，不含衍射。'),
        'apertureStop': tr('The stop stays at z = −5 mm while its radius changes; the glass mechanical radius stays fixed.', '光阑半径改变时，其位置保持在 z = −5 mm，玻璃机械半径也固定。'),
        'lensPrescription': tr('Two signed 50 mm radii, a 4 mm centre thickness and a chosen index define the traced lens.', '两个带符号的 50 mm 半径、4 mm 中心厚度和选定折射率定义被追迹的透镜。'),
    },
    'E12': {
        'rayTracing': tr('Every trial prescription is evaluated with the same sequential surface model used in E11.', '每个试探结构均由 E11 中同一套顺序曲面模型评价。'),
        'lensPrescription': tr('Compare the initial and final curvatures and common image plane under the fixed lens constraints.', '在固定透镜约束下比较初始与最终曲率，以及共用像面。'),
        'lensBending': tr('Changing q changes both radii while preserving a 50 mm paraxial effective focal length.', '改变 q 会同时改变两个半径，同时保持 50 mm 近轴有效焦距。'),
        'meritFunction': tr('Compare an on-axis score with equal weights at 0°, 3° and 6°; both use a single detector plane.', '比较轴上评分与 0°、3°、6° 等权评分；两者都使用单一探测器平面。'),
        'rmsSpot': tr('The displayed aggregate RMS is √J, not the mean of the field RMS values. The optimized merit J remains the mean squared spot radius.', '显示的综合 RMS 是 √J，而非各视场 RMS 值的平均。实际优化的评价函数 J 仍然是光斑均方半径的平均。'),
        'designConstraints': tr('The optimizer cannot improve the score by closing the stop, changing focal length, or silently dropping difficult rays.', '优化器不能通过缩小光阑、改变焦距或悄悄丢弃难以追迹的光线来改善评分。'),
        'apertureStop': tr('The entrance stop remains fixed during bending, so compared designs receive the same incident ray bundle.', '弯曲变化过程中入射光阑保持固定，确保比较的设计接收同一组入射光线。'),
        'spotDiagram': tr('Initial and optimized spots share a scale, while the independent field audit checks for off-axis tradeoffs.', '初始与优化点列图使用相同尺度，独立视场复核用于检查轴外性能的取舍。'),
        'objective': tr('Changing the selected fields changes the design question; a lower score answers only that question.', '改变所选视场就改变了设计问题；更低评分只回答所定义的问题。'),
        'inverseDesign': tr('This two-variable example uses a bounded direct search; many-variable wave design in E10 uses a different derivative strategy.', '这个双变量示例采用有界直接搜索；E10 的多变量波动设计使用另一种导数计算策略。'),
    },
}


def write(root):
    (root / 'shared/geometric-glossary.js').write_text('// Generated by tools/geometric_terms.py.\n'
        + 'export const geometricTerms = ' + json.dumps(TERMS, ensure_ascii=False, indent=2) + ';\n'
        + 'export const geometricLessonTerms = ' + json.dumps(LESSON_TERMS, ensure_ascii=False, indent=2) + ';\n')
