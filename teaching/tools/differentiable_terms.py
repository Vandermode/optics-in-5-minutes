"""Paired glossary definitions and chapter contexts for differentiable optics."""
import json
from geometric_course import tr, TRANSLATIONS


def term(name, zh, aliases, definition, definition_zh, formula='', notation='', notation_zh=''):
    return dict(name=tr(name, zh), aliases=aliases, definition=tr(definition, definition_zh),
                formula=formula, notation=tr(notation, notation_zh) if notation else '', episodes=['E13'])


TERMS = {
    'differentiableOptics': term('Differentiable optics', '可微光学',
        ['differentiable optics', 'differentiable optical model', 'differentiable optical calculation'],
        'An optical model whose output has usable derivatives with respect to chosen design parameters. Combining those derivatives with a measurement and loss connects an image error to an optical design update; the derivatives describe the stated model.',
        '输出对选定设计参数具有可用导数的光学模型。将这些导数与测量及损失连接，可以把图像误差传回光学设计参数；导数描述的是所采用的模型。',
        'θ → u → I → L;   g = ∇θ L',
        'θ denotes real design parameters, u the complex field, I the measured intensity and L a real scalar loss. Differentiability can fail when the selected optical path changes.',
        'θ 是实数设计参数，u 是复数场，I 是测量强度，L 是实数标量损失。选定光路发生切换时，可微性可能失效。'),
    'endToEnd': term('End-to-end optical design', '端到端光学设计',
        ['end-to-end optical design', 'end-to-end design', 'end-to-end co-design', 'joint design'],
        'Optimizing optical and digital parameters against a final task loss evaluated after measurement and reconstruction. The loss is differentiated through both parts of the model; optimizing the raw optical intensity alone is a different objective.',
        '用测量与重建之后的最终任务损失，同时优化光学参数和数字处理参数。损失需要通过模型的这两部分求导；仅优化原始光学强度对应另一种目标。',
        'y = Hθ(s);   ŝ = Rψ(y);   Ltask = ℓ(ŝ, s)',
        's is the scene, Hθ the optical and sensor model, y the measurement and Rψ the reconstruction. Compare designs with fixed acquisition conditions and evaluate scenes not used during optimization.',
        's 是场景，Hθ 是光学与传感器模型，y 是测量，Rψ 是重建算法。比较时固定采集条件，并评价未用于优化的场景。'),
    'reverseMode': term('Reverse-mode differentiation', '反向模式微分',
        ['reverse-mode differentiation', 'reverse-mode derivatives', 'reverse-mode gradient', 'reverse gradient', 'reverse mode', 'reverse-mode'],
        'Applying the chain rule from a scalar loss back through a sequence of operations to its inputs. A backward quantity carries loss sensitivity; it is not generally an inverse of the physical forward process.',
        '从标量损失出发，沿一系列运算反向应用链式法则，求出损失对输入的导数。反向量传递的是损失灵敏度，通常不等于物理前向过程的逆。',
        'y = f(x);   ∇x L = Jf(x)ᵀ ∇y L',
        'This expression uses real coordinates. Jf is the Jacobian of one operation; complex fields can be represented by real and imaginary parts. E13 writes the equivalent complex adjoint formulas explicitly.',
        '此式采用实数坐标。Jf 是一个运算的雅可比矩阵；复数场可以分解为实部和虚部。E13 显式写出等价的复数伴随公式。'),
    'implicitDifferentiation': term('Implicit differentiation', '隐式求导',
        ['implicit differentiation', 'implicit derivative'],
        'Differentiating a solved constraint instead of every step used to solve it. For a ray–surface intersection, it gives the local change of the selected intersection distance when a surface parameter changes.',
        '直接对求解结果满足的约束求导，无需逐步微分整个求解过程。对于光线与曲面的交点，它给出曲面参数改变时，所选交点距离的局部变化。',
        'S(t, θ) = 0;   dt/dθ = −(∂S/∂θ)/(∂S/∂t)',
        'The denominator must be nonzero and the same solution branch must remain selected. Incoming-ray changes require their own chain-rule terms; clipping and switching intersections need separate treatment.',
        '分母必须非零，且必须保持同一个解分支。入射光线的变化需要另加链式法则项；遮挡与交点切换需要单独处理。'),
}

ALIASES = {
    'differentiableOptics': ['可微光学', '可微光学模型', '可微光学计算'],
    'endToEnd': ['端到端光学设计', '端到端设计', '端到端联合设计', '联合设计'],
    'reverseMode': ['反向模式微分', '反向梯度', '反向导数', '反向计算'],
    'implicitDifferentiation': ['隐式求导', '隐式微分'],
}

LESSON_TERMS = {'E13': {
    'differentiableOptics': tr('Follow how two real phase coefficients change the pupil field, the measured intensity and the loss.', '追踪两个实数相位系数如何改变瞳面场、测量强度和损失。'),
    'endToEnd': tr('The final section extends the dependency chain through reconstruction; the browser experiment itself stops at an optical intensity loss.', '最后一节将依赖链延伸到重建；浏览器实验本身仍以光学强度损失为终点。'),
    'reverseMode': tr('The experiment explicitly carries sensitivity back from the intensity loss through a Fourier transform and the phase parameterization.', '实验显式地将强度损失的灵敏度通过傅里叶变换和相位参数化反向传递。'),
    'implicitDifferentiation': tr('The ray-intersection example explains how a smooth local sensitivity can pass through a solved surface equation.', '光线求交示例解释如何让平滑的局部灵敏度通过已求解的曲面方程。'),
    'automaticDifferentiation': tr('An autodiff system could assemble these local derivative rules; this small experiment writes them explicitly and checks them independently.', '自动微分系统可以组合这些局部求导规则；本小实验显式编写规则，并进行独立检验。'),
    'adjointMethod': tr('A solver may provide an adjoint derivative as one operation inside a larger differentiable image-formation model.', '求解器可以提供伴随导数，作为更大可微成像模型中的一个运算。'),
    'adjoint': tr('The Fourier conjugate transpose carries the field sensitivity back to the pupil. Its equality to the inverse depends on the unitary normalization.', '傅里叶共轭转置将场灵敏度传回瞳面。它与逆变换相等依赖于酉归一化。'),
    'objective': tr('The same full-pattern intensity error defines both the derivative check and every accepted design update.', '导数检验和每个被接受的设计更新都使用同一个完整图样强度误差。'),
    'gradient': tr('Weight the pixelwise phase sensitivities by x and x² to obtain the derivatives of the tilt and defocus coefficients.', '分别用 x 与 x² 对逐像素相位灵敏度加权求和，得到倾斜和离焦系数的导数。'),
    'finiteDifference': tr('Perturb each coefficient at the same initial design and compare several step sizes with the reverse-mode result.', '在同一个初始设计处扰动各系数，以多个扰动步长与反向计算结果比较。'),
    'unitary': tr('The chosen Fourier normalization preserves total power and makes its conjugate transpose equal its inverse.', '选定的傅里叶归一化保持总功率，并使其共轭转置等于逆变换。'),
    'lineSearch': tr('The optimizer recomputes the loss at each trial and reduces the step until it finds an accepted decrease.', '优化器在每个试探点重新计算损失，并缩小步长，直到找到可接受的下降。'),
    'pupil': tr('Its amplitude and total power stay fixed while the linear and quadratic phase coefficients change.', '线性和二次相位系数改变时，瞳振幅与总功率保持固定。'),
    'phase': tr('Changing pupil phase preserves local pupil intensity but changes interference at the detector.', '改变瞳面相位保持局部瞳强度不变，却会改变探测器上的干涉。'),
    'quantization': tr('Reevaluate the optimized phase after rounding it to fabrication levels; its hard rounding is not the smooth derivative used during design.', '将优化相位取整到加工等级后重新评价；硬取整的导数并不等于设计中使用的平滑导数。'),
}}


def write(root):
    (root / 'shared/differentiable-glossary.js').write_text('// Generated by tools/differentiable_terms.py.\n'
        + 'export const differentiableTerms = ' + json.dumps(TERMS, ensure_ascii=False, indent=2) + ';\n'
        + 'export const differentiableLessonTerms = ' + json.dumps(LESSON_TERMS, ensure_ascii=False, indent=2) + ';\n')
