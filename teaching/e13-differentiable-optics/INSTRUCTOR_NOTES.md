# Instructor notes · E13 · Differentiable optics: from images to design updates

<!-- course-motivation:start -->
## Why learn this?

**Which optical parameter should change to improve the measured image?**

Trying every lens curvature or phase coefficient separately becomes expensive. A derivative connects a measured error to its causes in the optical model, so a design update can follow a calculated direction.

**What you gain:** Build and verify a differentiable optical calculation, understand its relation to adjoints, and identify what is needed before extending it to joint optics and reconstruction design.
<!-- course-motivation:end -->

For learners with undergraduate calculus, linear algebra, and basic Python. Each lesson names the specific ideas it uses and links to focused refreshers. Review unfamiliar topics as needed; the lesson develops the optical method. Use the checklist below before class. During the lesson, give a short notation reminder or point back to a resource when needed, then keep the explanation focused on the current optical question.

<!-- course-preparation:start -->
## Before this lesson

Use these focused refreshers if an idea is unfamiliar; you do not need to complete each linked course.

- **Chain rule and gradients:** Apply the chain rule to a real scalar loss. Used here: The final loss depends on phase through a field and an intensity measurement. [Review: MIT OCW · Chain Rule, Gradient and Directional Derivatives](https://ocw.mit.edu/courses/18-02sc-multivariable-calculus-fall-2010/pages/2.-partial-derivatives/part-b-chain-rule-gradient-and-directional-derivatives/). Use Sessions 32 and 34 for the multivariable chain rule, Session 35 for the gradient, and Session 38 for directional derivatives.
- **Complex inner products:** Use conjugation and a unitary Fourier transform. Used here: The conjugate transpose carries field sensitivity back to the pupil. [Review: MIT OCW · Lecture 26: Complex matrices; fast Fourier transform](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/resources/lecture-26-complex-matrices-fast-fourier-transform/). Focus on complex vectors, conjugate transpose, Hermitian matrices, and unitary matrices. The FFT algorithm at the end is optional here.

**Python preparation:** Manipulate complex arrays and compare scalar function values at nearby parameter settings.
- [NumPy · The absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) — Use the sections on array shape, indexing and slicing, basic operations, and broadcasting. Practice changing one parameter and rerunning a cell.
- [NumPy · Discrete Fourier Transform](https://numpy.org/doc/stable/reference/routines.fft.html) — Read Implementation details and Normalization; look up fftfreq, fftshift, and ifftshift as they appear in the notebook.

**Developed here:** Connect parameterized optics, reverse derivatives, gradient checks and a design update. No deep-learning framework is required.

[All prerequisites and learning resources](https://vandermode.github.io/optics-in-5-minutes-course/preparation/)
<!-- course-preparation:end -->

## Outcomes

- Trace the dependency from real design parameters through a complex field to a real loss.
- Check a reverse-mode gradient with finite differences before using it to update a design.
- Distinguish differentiable optics, automatic differentiation, adjoint methods and end-to-end co-design.

## Suggested 45–50-minute sequence

1. Predict (3 minutes): Changing a phase mask leaves its local intensity unchanged. Can it still change the focal-plane intensity? Predict what the tilt and defocus controls do, then compare zero updates with forty updates.
2. Explore (8 minutes): Set the update count to zero. Change tilt alone, then defocus alone. Explain which changes the position and which changes the shape of the detector pattern. Inspect both gradient components at the initial design. Compare reverse mode with a central finite difference; the sign predicts which small parameter change lowers the loss. Choose Use the gradient and compare initial, target and final intensity on the same scale. Move the target and repeat. Report loss reduction without claiming that every start reaches the unique best design.
3. Explain (6 minutes): use the equation and measurement or boundary assumptions to connect the observed changes.
4. Check (3 minutes): ask the three page questions before showing feedback; have learners justify an answer.
5. Compute (20–25 minutes): run the core notebook, alter one parameter, and inspect its numerical checks.
6. Reflect (5 minutes): state one changed prediction, the evidence, and a remaining model limitation.

Use the optional extension for a longer lab rather than rushing the core. The quiz is local and does not collect student responses.

## Assessment and worked reasoning

### Write the forward chain

Name the design variables, fixed quantities, measurement and loss. Show why phase can change the focal intensity without changing pupil power.

**Worked reasoning:** Unit-modulus phase preserves local pupil intensity; Fourier interference redistributes that fixed power among detector samples.

### Audit both gradient components

Compare reverse-mode and finite-difference derivatives over several perturbation sizes at the same initial design.

**Worked reasoning:** Look for a range of h where truncation and roundoff errors are both small. Compare absolute errors when a true component is near zero.

### Interpret the optimized result

Report the initial and final loss, fixed target and power, and one limitation of the result.

**Worked reasoning:** A decreasing loss supports improvement for this sampled target. Intensity ambiguities, local stationary points and omitted physics prevent stronger conclusions.

## Ten-point rubric

- Prediction and physical reasoning: 2 points.
- Reproducible parameter choices, units, and labelled plots: 3 points.
- Quantitative comparison or numerical validation: 3 points.
- Model limitations and interpretation of disagreement: 2 points.

Grade reasoning and evidence, not an attractive plot alone. Results may vary with a deliberately changed seed or model; require the learner to record it.



## Common misconceptions

- Which path must a design gradient include? The derivative must match the full objective actually evaluated, including the intensity measurement.

- A finite-difference check agrees with the reverse gradient. What does this establish? A gradient check tests a numerical derivative; model fidelity and optimization outcomes require separate evidence.

- What makes optical and digital design end-to-end? The placement of the task loss and its dependency on both sets of parameters defines the joint problem.

## Scope

A monochromatic 1D scalar phase pupil with fixed amplitude and power, unitary Fourier propagation, and two continuous real parameters. The detector samples normalized intensity. No polarization, material dispersion, pixel integration, noise, fabrication process or learned reconstruction is simulated. Low loss establishes agreement with this target in this model, not unique parameter recovery or physical performance.

Optional extension: Quantize the optimized phase and reevaluate its intensity loss. Explain why the continuous design gradient is not the ordinary derivative of hard phase rounding.
