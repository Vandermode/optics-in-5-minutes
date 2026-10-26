# Instructor notes · E11 · Ray tracing through a real lens

<!-- course-motivation:start -->
## Why learn this?

**Why do the edge rays miss the focus?**

A thin-lens sketch predicts one focus, but a real spherical lens has thickness and accepts rays over a finite aperture. Before changing its shape, you need a model that follows where those rays actually go and explains which approximation caused the mismatch.

**What you gain:** Build a surface-by-surface ray tracer, locate a practical image plane and identify the geometric image errors that motivate lens design.
<!-- course-motivation:end -->

For learners with undergraduate calculus, linear algebra, and basic Python. Each lesson names the specific ideas it uses and links to focused refreshers. Review unfamiliar topics as needed; the lesson develops the optical method. Use the checklist below before class. During the lesson, give a short notation reminder or point back to a resource when needed, then keep the explanation focused on the current optical question.

<!-- course-preparation:start -->
## Before this lesson

Use these focused refreshers if an idea is unfamiliar; you do not need to complete each linked course.

- **Vectors, angles and surface normals:** Use a unit direction, a dot product and a quadratic surface equation. Used here: These identify the next intersection and the angle to its local normal. [Review: MIT OCW · Reflection and refraction, lecture 2](https://ocw.mit.edu/courses/2-71-optics-spring-2009/pages/lecture-slides/). Use lecture 2 to identify incident rays and interface normals. Review vector lengths and dot products before translating the picture into code.

**Python preparation:** Use small real vectors and arrays of ray positions; keep millimetres and degrees separate from array indices.
- [NumPy · The absolute basics for beginners](https://numpy.org/doc/stable/user/absolute_beginners.html) — Use the sections on array shape, indexing and slicing, basic operations, and broadcasting. Practice changing one parameter and rerunning a cell.

**Developed here:** Build the sequential ray-tracing algorithm and compare nonparaxial geometric rays with the small-angle lens prediction.

[All prerequisites and learning resources](https://vandermode.github.io/optics-in-5-minutes-course/preparation/)
<!-- course-preparation:end -->

## Outcomes

- Find a forward surface intersection and apply Snell’s law using the local normal.
- Use a ray fan to test when a paraxial focal prediction stops being accurate.
- Explain why a geometric spot diagram does not establish diffraction-limited resolution.

## Suggested 45–50-minute sequence

1. Predict (3 minutes): A thin-lens sketch puts all parallel rays at one point. With the same spherical lens and image plane, will doubling the entrance radius keep that point sharp? Predict what the outer rays will do before opening the aperture.
2. Explore (8 minutes): Choose the narrow on-axis bundle. Keep n = 1.5 and compare its crossing with the paraxial image prediction. Move only the image plane until the geometric spot is smallest. Return to the same image plane and increase the aperture radius. Follow the outer rays and the transverse ray fan. Refocusing can reduce the spread but does not make every spherical-surface ray meet at one point. Keep the aperture fixed and select an off-axis object point. Compare the ray diagram, spot centroid and spread separately. Explain which change is image position and which is aberration.
3. Explain (6 minutes): use the equation and measurement or boundary assumptions to connect the observed changes.
4. Check (3 minutes): ask the three page questions before showing feedback; have learners justify an answer.
5. Compute (20–25 minutes): run the core notebook, alter one parameter, and inspect its numerical checks.
6. Reflect (5 minutes): state one changed prediction, the evidence, and a remaining model limitation.

Use the optional extension for a longer lab rather than rushing the core. The quiz is local and does not collect student responses.

## Assessment and worked reasoning

### Follow one ray

Report both intersections, the local incidence angles and the two refracted directions. Check Snell’s law numerically.

**Worked reasoning:** Use the sphere centre to construct each normal. Check forward distance, cap membership and the unit length of every direction.

### Test the paraxial limit

Reduce the entrance radius at fixed index and compare the traced on-axis focus with the independent thick-lens prediction.

**Worked reasoning:** The narrow-bundle limit approaches the paraxial back focal distance. Report the image coordinate from the first vertex separately from effective focal length.

### Separate image position and blur

At two field angles, report the centroid and RMS radius with the same image plane and pupil sampling.

**Worked reasoning:** Centroid motion locates the image; spread about that centroid is geometric blur. Neither number alone measures diffraction-limited resolution.

## Ten-point rubric

- Prediction and physical reasoning: 2 points.
- Reproducible parameter choices, units, and labelled plots: 3 points.
- Quantitative comparison or numerical validation: 3 points.
- Model limitations and interpretation of disagreement: 2 points.

Grade reasoning and evidence, not an attractive plot alone. Results may vary with a deliberately changed seed or model; require the learner to record it.



## Common misconceptions

- At a curved refracting surface, the incidence angle is measured from… The normal changes across a curved surface. Using the optical axis instead changes the actual refraction problem.

- A small-angle focus agrees with a narrow ray bundle but not a wide one. The first explanation to test is… The exact geometric surface trace still obeys Snell’s law. Its finite-angle departure from the paraxial prediction is an aberration.

- Reducing the geometric RMS spot proves the image resolution improved only after… A geometric spot omits diffraction, and removing rays changes the experiment. Compare the same pupil and complete model assumptions.

## Scope

Sequential transmitted rays through two spherical or plane surfaces, with a real constant glass index and an object at infinity. The mechanical clear radius is 12 mm. No diffraction, polarization, Fresnel power weights, material dispersion or surface roughness is modeled. A geometric RMS spot is not a PSF or an MTF.

Optional extension: Compare the aperture dependence of the best on-axis geometric RMS radius with an approximate Airy first-zero radius at 550 nm. These are different size measures; do not add them or interpret their crossing as a complete diffraction prediction.
