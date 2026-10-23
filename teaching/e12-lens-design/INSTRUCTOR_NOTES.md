# Instructor notes · E12 · Lens design with a measurable objective

<!-- course-motivation:start -->
## Why learn this?

**Which lens is better for the image you need?**

Two lenses can have the same focal length yet form different finite-aperture spots. Improving the centre may worsen the edge. A useful design process must state the required field, pupil and image plane before it changes the surface shapes.

**What you gain:** Turn an imaging requirement into a reproducible objective, optimize within physical constraints and explain the tradeoffs that remain after the score improves.
<!-- course-motivation:end -->

For learners with undergraduate calculus, linear algebra, and basic Python. Each lesson names the specific ideas it uses and links to focused refreshers. Review unfamiliar topics as needed; the lesson develops the optical method. Use the checklist below before class. During the lesson, give a short notation reminder or point back to a resource when needed, then keep the explanation focused on the current optical question.

<!-- course-preparation:start -->
## Before this lesson

Use these focused refreshers if an idea is unfamiliar; you do not need to complete each linked course.

- **Ray states and lens reference planes:** Trace a prescribed lens and interpret an image-plane ray intersection. Used here: Every trial design is evaluated by the same forward ray model. [Review: MIT OCW · Thick lenses, lecture 5](https://ocw.mit.edu/courses/2-71-optics-spring-2009/pages/lecture-slides/). Read the thick-lens and principal-plane diagrams. E11 develops the actual surface-intersection and refraction algorithm.
- **Weighted means and squared error:** Distinguish the centroid of a spot from its spread and combine weighted errors. Used here: The objective compares geometric spot variances across specified fields. [Review: MIT OCW · Aberrations and optical design, lectures 8–9](https://ocw.mit.edu/courses/2-71-optics-spring-2009/pages/lecture-slides/). Use the spot diagrams to separate image position from image spread. The lesson defines its own RMS metric and field weights explicitly.

**Python preparation:** Evaluate a function on a parameter grid, record the best valid trial and recheck it with denser sampling.
- [NumPy · The absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) — Use the sections on array shape, indexing and slicing, basic operations, and broadcasting. Practice changing one parameter and rerunning a cell.

**Developed here:** Define and optimize a constrained lens-design objective. Calculus-based optimization and an adjoint solver are not required for the core experiment.

[All prerequisites and learning resources](https://vandermode.github.io/optics-in-5-minutes-course/preparation/)
<!-- course-preparation:end -->

## Outcomes

- Separate a lens prescription, a performance metric and the constraints that keep the task unchanged.
- Compare on-axis and multi-field optimization using one common image plane and the same entrance aperture.
- Audit an improved design with denser pupil sampling and explain the image-quality questions its merit function omits.

## Suggested 45–50-minute sequence

1. Predict (3 minutes): Two lenses have the same paraxial focal length and collect the same entrance bundle. Must they form the same image? Predict whether minimizing the central spot alone will also give the best result at 6°.
2. Explore (8 minutes): Run the central-field design. Compare the initial and optimized prescriptions, then inspect all three field RMS values, not only the loss used by the search. Keep the aperture and starting shape unchanged, switch to equal field weights, and run again. Identify the central-field and edge-field tradeoff at the single common image plane. Try a different starting shape and increase the search refinement. Inspect transmission and edge clearance as well as spot size, then repeat the result with denser pupil sampling in Python.
3. Explain (6 minutes): use the equation and measurement or boundary assumptions to connect the observed changes.
4. Check (3 minutes): ask the three page questions before showing feedback; have learners justify an answer.
5. Compute (20–25 minutes): run the core notebook, alter one parameter, and inspect its numerical checks.
6. Reflect (5 minutes): state one changed prediction, the evidence, and a remaining model limitation.

Use the optional extension for a longer lab rather than rushing the core. The quiz is local and does not collect student responses.

## Assessment and worked reasoning

### State the design contract

List the variables, fixed focal power, entrance aperture, field weights, common image plane and failure penalties.

**Worked reasoning:** A reproducible design includes both the prescription and the evaluation conditions. Fixed aperture and ray-failure checks prevent a misleading improvement.

### Compare two objectives

Optimize the on-axis and multi-field objectives from the same start. Report all field RMS values and the common image location for each result.

**Worked reasoning:** Changing field weights changes the design problem. Judge each result against the same full field table rather than comparing two unlike scalar losses.

### Validate the final prescription

Re-trace the chosen lens with denser pupil sampling and an intermediate field. Report any change in the ranking and the transmitted-ray fraction.

**Worked reasoning:** Independent sampling checks whether the selected solution exploits the evaluation grid. It complements, rather than replaces, a more complete wave and tolerance model.

## Ten-point rubric

- Prediction and physical reasoning: 2 points.
- Reproducible parameter choices, units, and labelled plots: 3 points.
- Quantitative comparison or numerical validation: 3 points.
- Model limitations and interpretation of disagreement: 2 points.

Grade reasoning and evidence, not an attractive plot alone. Results may vary with a deliberately changed seed or model; require the learner to record it.

## Common misconceptions

- Two prescriptions have the same effective focal length. Their finite-aperture spots… Paraxial power constrains the small-angle response. It does not fix all higher-order ray deviations or the back focal distance.

- Why must every field use the same image plane in this comparison? Separately refocusing every field hides a tradeoff that one fixed sensor would still see.

- A design has lower centroid-based RMS at 0°, 3° and 6°. What is still untested? The metric rewards only the quantities and configurations it contains. Validate omitted conditions explicitly.

## Scope

Monochromatic sequential geometric design of a single spherical lens in air: n = 1.5, centre thickness 4 mm and paraxial EFL 50 mm. Only bending and a common image position are optimized at fixed entrance aperture. RMS is measured about each field centroid. Diffraction, chromatic aberration, distortion targets, polarization, Fresnel power weights and manufacturing tolerances are not optimized.

Optional extension: Apply seeded perturbations to the surface radii, centre thickness and refractive index, then re-evaluate the same fields at the unchanged nominal image plane. This tests sensitivity under assumed variations, not manufacturing yield.
