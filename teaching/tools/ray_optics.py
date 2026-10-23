"""Sequential geometric rays for E11/E12; all lengths mm and fields degrees.

Coordinates are (x,y,z). A signed spherical radius has centre z=vertex+R;
only the cap containing the vertex belongs to the optical surface. None is a
plane. Surface normals point into the transmitted side, along the incident ray.
The circular entrance stop is fixed at z=-5 mm in air, independently of field.
No diffraction, Fresnel transmission weights, polarization, dispersion, or ghosts.
Pure Python keeps this source usable by both standalone notebooks and films.
"""
from math import acos, cos, copysign, hypot, inf, isfinite, pi, sin, sqrt


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def unit(v):
    length = sqrt(dot(v, v))
    if not isfinite(length) or length <= 0:
        raise ValueError("A ray direction must have a finite positive length.")
    return [x / length for x in v]


def curvature(radius):
    return 0.0 if radius is None or not isfinite(radius) else 1.0 / radius


def surface_sag(radius, rho):
    c = curvature(radius)
    radicand = 1 - (c * rho) ** 2
    if radicand < -1e-12:
        return float("nan")
    return c * rho * rho / (1 + sqrt(max(0.0, radicand)))


def make_lens(r1=50.0, r2=-50.0, thickness=4.0, index=1.5, clear_radius=12.0):
    if thickness <= 0 or index <= 0 or clear_radius <= 0:
        raise ValueError("Thickness, refractive index, and clear radius must be positive.")
    radii = [None if r is None or not isfinite(r) else float(r) for r in (r1, r2)]
    if any(r == 0 for r in radii):
        raise ValueError("Zero radius is not a plane; use None or infinity.")
    edge = thickness + surface_sag(radii[1], clear_radius) - surface_sag(radii[0], clear_radius)
    return dict(r1=radii[0], r2=radii[1], thickness=float(thickness), index=float(index),
                clear_radius=float(clear_radius), edge_thickness=edge)


def paraxial(lens):
    """Independent first-order thick-lens prediction, object at infinity in air."""
    n, d = lens["index"], lens["thickness"]
    c1, c2 = curvature(lens["r1"]), curvature(lens["r2"])
    power = (n - 1) * (c1 - c2 + (n - 1) * d * c1 * c2 / n)
    focal = 1 / power if abs(power) > 1e-15 else inf
    bfl = focal * (1 - (n - 1) * d * c1 / n)
    return dict(power=power, focal=focal, bfl=bfl, image_z=d + bfl)


def prescription(shape=0.0, focal=50.0, index=1.5, thickness=4.0, clear_radius=12.0):
    """Vary q=(c1+c2)/(c1-c2), keeping the paraxial thick-lens power fixed."""
    if focal <= 0 or index <= 1 or abs(shape) > 1.2 + 1e-12:
        raise ValueError("Use positive focal length, n>1, and -1.2 <= shape <= 1.2.")
    b = 1 / (focal * (index - 1))
    a = (index - 1) * thickness * (shape * shape - 1) / (4 * index)
    discriminant = 1 + 4 * a * b
    if discriminant <= 0:
        raise ValueError("No real prescription on the selected weak-curvature branch.")
    delta = 2 * b / (1 + sqrt(discriminant))
    c1, c2 = delta * (shape + 1) / 2, delta * (shape - 1) / 2
    lens = make_lens(None if abs(c1) < 1e-15 else 1 / c1,
                     None if abs(c2) < 1e-15 else 1 / c2, thickness, index, clear_radius)
    lens.update(shape=float(shape), target_focal=float(focal))
    return lens


def intersect_surface(origin, direction, vertex, radius, clear_radius=12.0):
    """Nearest forward intersection with the physical cap, never the far hemisphere."""
    direction = unit(direction)
    if curvature(radius) == 0:
        roots = [] if abs(direction[2]) < 1e-14 else [(vertex - origin[2]) / direction[2]]
    else:
        relative = [origin[0], origin[1], origin[2] - vertex - radius]
        b, c = 2 * dot(relative, direction), dot(relative, relative) - radius * radius
        disc = b * b - 4 * c
        if disc < -1e-10:
            return dict(status="missed")
        q = -.5 * (b + copysign(sqrt(max(0.0, disc)), b))
        roots = [q, c / q] if abs(q) > 1e-15 else [-b / 2]
    for t in sorted(roots):
        if t <= 1e-9:
            continue
        point = [o + t * v for o, v in zip(origin, direction)]
        rho = hypot(point[0], point[1])
        sag = surface_sag(radius, rho)
        if not isfinite(sag) or abs(point[2] - vertex - sag) > 1e-7 * max(1, abs(vertex), abs(point[2])):
            continue
        if rho > clear_radius + 1e-9:
            return dict(status="blocked", point=point, distance=t)
        normal = [0., 0., 1.] if curvature(radius) == 0 else unit([point[0], point[1], point[2] - vertex - radius])
        if dot(direction, normal) < 0:
            normal = [-x for x in normal]
        return dict(status="ok", point=point, distance=t, normal=normal)
    return dict(status="missed")


def refract(direction, normal, n_before, n_after):
    """Vector Snell law; normal is oriented into the transmitted medium."""
    d, normal = unit(direction), unit(normal)
    if dot(d, normal) < 0:
        normal = [-x for x in normal]
    cosine = max(0., min(1., dot(d, normal)))
    eta = n_before / n_after
    tangential = [d[i] - cosine * normal[i] for i in range(3)]
    radicand = 1 - eta * eta * dot(tangential, tangential)
    if radicand < -1e-12:
        return dict(status="tir", normal=normal, cos_incident=cosine)
    transmitted = sqrt(max(0., radicand))
    return dict(status="ok", direction=unit([eta * tangential[i] + transmitted * normal[i] for i in range(3)]),
                normal=normal, cos_incident=cosine, cos_transmitted=transmitted)


def trace_ray(origin, direction, lens, image_z=None):
    point, direction = list(origin), unit(direction)
    points, surfaces = [point[:]], []
    for number, (vertex, radius, ni, nt) in enumerate([(0., lens["r1"], 1., lens["index"]),
            (lens["thickness"], lens["r2"], lens["index"], 1.)], 1):
        hit = intersect_surface(point, direction, vertex, radius, lens["clear_radius"])
        hit["surface"] = number
        surfaces.append(hit)
        if "point" in hit:
            points.append(hit["point"])
        if hit["status"] != "ok":
            return dict(status=hit["status"], points=points, surfaces=surfaces, failed_surface=number,
                        exit=None, direction=None, image=None)
        bent = refract(direction, hit["normal"], ni, nt)
        hit.update(n_before=ni, n_after=nt, incoming=direction, outgoing=bent.get("direction"))
        if bent["status"] != "ok":
            return dict(status=bent["status"], points=points, surfaces=surfaces, failed_surface=number,
                        exit=None, direction=None, image=None)
        point, direction = hit["point"], bent["direction"]
    image = None
    if image_z is not None:
        t = (image_z - point[2]) / direction[2] if abs(direction[2]) > 1e-14 else -1
        if t < 0:
            return dict(status="image_behind_lens", points=points, surfaces=surfaces,
                        exit=point, direction=direction, image=None)
        image = [p + t * v for p, v in zip(point, direction)]
        points.append(image)
    return dict(status="ok", points=points, surfaces=surfaces, exit=point, direction=direction, image=image)


def pupil_ray(x, y, field_deg, lens, image_z=None, launch_z=-10.0, pupil_z=-5.0):
    theta = field_deg * pi / 180
    direction = [sin(theta), 0., cos(theta)]
    origin = [x + (launch_z - pupil_z) * direction[0] / direction[2], y, launch_z]
    result = trace_ray(origin, direction, lens, image_z)
    result["pupil"] = [x, y]
    return result


def pupil_samples(aperture, rings=6, spokes=16):
    """Equal-area annuli, midpoint radial quadrature, equal weights per ray."""
    if aperture <= 0 or rings < 1 or spokes < 4:
        raise ValueError("Use a positive pupil and at least one ring/four spokes.")
    return [[aperture * sqrt((j + .5) / rings) * cos(2 * pi * (k + .5 * (j % 2)) / spokes),
             aperture * sqrt((j + .5) / rings) * sin(2 * pi * (k + .5 * (j % 2)) / spokes)]
            for j in range(rings) for k in range(spokes)]


def trace_bundle(lens, aperture, field_deg, image_z, rings=6, spokes=16):
    rays = [pupil_ray(x, y, field_deg, lens, image_z) for x, y in pupil_samples(aperture, rings, spokes)]
    accepted = [r for r in rays if r["status"] == "ok"]
    xy = [r["image"][:2] for r in accepted if r["image"] is not None]
    centroid = [sum(p[i] for p in xy) / len(xy) for i in range(2)] if xy else None
    rms = sqrt(sum(sum((p[i] - centroid[i]) ** 2 for i in range(2)) for p in xy) / len(xy)) if xy else None
    return dict(rays=rays, xy=xy, centroid=centroid, rms_mm=rms, requested=len(rays),
                accepted=len(accepted), throughput=len(accepted) / len(rays), field_deg=field_deg, image_z=image_z)


def best_image_plane(lens, aperture, fields=(0., 3., 6.), rings=6, spokes=16, bounds=(30., 90.)):
    """Minimize equal-field-weighted centroid RMS² over ONE common axial plane.

    In air after the second surface, x(z)=a+bz. Subtract the centroid separately
    within each field. Sum all field variances before solving the quadratic.
    This removes image displacement from the spot size, but never refocuses fields.
    """
    quadratic, linear = 0., 0.
    for field in fields:
        bundle = trace_bundle(lens, aperture, field, None, rings, spokes)
        accepted = [r for r in bundle["rays"] if r["status"] == "ok" and r["direction"][2] > 0]
        if not accepted:
            return max(bounds[0], min(bounds[1], paraxial(lens)["image_z"]))
        for axis in range(2):
            slopes = [r["direction"][axis] / r["direction"][2] for r in accepted]
            intercepts = [r["exit"][axis] - r["exit"][2] * b for r, b in zip(accepted, slopes)]
            bm, am = sum(slopes) / len(slopes), sum(intercepts) / len(intercepts)
            quadratic += sum((b - bm) ** 2 for b in slopes) / len(slopes)
            linear += sum((a - am) * (b - bm) for a, b in zip(intercepts, slopes)) / len(slopes)
    return max(bounds[0], min(bounds[1], -linear / quadratic)) if quadratic > 1e-20 else bounds[1]


def design_merit(shape, image_z, aperture=6., fields=(0., 3., 6.), rings=6, spokes=16):
    lens = prescription(shape)
    bundles = [trace_bundle(lens, aperture, field, image_z, rings, spokes) for field in fields]
    loss = sum(1 - b["throughput"] for b in bundles) / len(bundles)
    valid = isfinite(lens["edge_thickness"]) and lens["edge_thickness"] >= .5 and loss == 0
    rms2 = sum((b["rms_mm"] or 0.) ** 2 for b in bundles) / len(bundles)
    # Any clipped/lost/TIR ray invalidates the design. A missing ray never helps.
    edge_penalty = max(0., .5 - lens["edge_thickness"]) if isfinite(lens["edge_thickness"]) else 1e3
    value = rms2 if valid else 1e6 + 1e3 * loss + 1e3 * edge_penalty
    return dict(value_mm2=value, valid=valid, loss_fraction=loss, bundles=bundles, lens=lens,
                shape=float(shape), image_z=float(image_z), rms_mm=sqrt(rms2), aperture=aperture,
                fields=list(fields))


def optimize_design(shape=0., aperture=6., fields=(0., 3., 6.), iterations=24, rings=6, spokes=16):
    """Bounded deterministic pattern search; common image plane solved analytically.

    This small educational search is not a claim of global optimality. Aperture,
    field weights, material, thickness, mechanical diameter and paraxial power stay
    fixed. All merit values retain the same physical mm² normalization.
    """
    lens = prescription(shape)
    start = design_merit(shape, paraxial(lens)["image_z"], aperture, fields, rings, spokes)
    best, step, trials = start, .4, 0
    history = [dict(shape=best["shape"], image_z=best["image_z"], value_mm2=best["value_mm2"])]
    for iteration in range(iterations):
        candidates = [best["shape"]] if iteration == 0 else [max(-1.2, best["shape"] - step), min(1.2, best["shape"] + step)]
        improved = False
        for q in candidates:
            trial_lens = prescription(q)
            z = best_image_plane(trial_lens, aperture, fields, rings, spokes)
            result = design_merit(q, z, aperture, fields, rings, spokes)
            trials += 1
            if result["value_mm2"] < best["value_mm2"] - 1e-15:
                best, improved = result, True
        if not improved and iteration > 0:
            step /= 2
        history.append(dict(shape=best["shape"], image_z=best["image_z"], value_mm2=best["value_mm2"]))
        if step < 1e-5:
            break
    return dict(start=start, best=best, history=history, trials=trials, completed=len(history) - 1)
