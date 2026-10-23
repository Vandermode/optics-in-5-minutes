"""Independent scalar-Snell / bisection references and Python/JS ray fixtures."""
from pathlib import Path
import json
import math
import ray_optics as ray

ROOT = Path(__file__).resolve().parents[1]


def scalar_meridional(lens, pupil_height, field_deg, image_z):
    """Independent implementation: bracket sag equation, refract scalar angles.

    It does not call the production intersection, normal, vector-refraction, or
    tracing routines. This is intentionally limited to a 2D meridional slice.
    """
    angle = math.radians(field_deg)
    x, z = pupil_height - 5 * math.tan(angle), -10.
    points = [[x, 0., z]]
    for vertex, radius, ni, nt in [(0., lens["r1"], 1., lens["index"]),
                                    (lens["thickness"], lens["r2"], lens["index"], 1.)]:
        if radius is None:
            hit_z = vertex
        else:
            edge_sag = radius - math.copysign(math.sqrt(radius * radius - lens["clear_radius"] ** 2), radius)
            low, high = sorted([vertex, vertex + edge_sag])
            def residual(zz):
                xx = x + (zz - z) * math.tan(angle)
                return zz - vertex - radius + math.copysign(math.sqrt(radius * radius - xx * xx), radius)
            assert residual(low) <= 1e-12 and residual(high) >= -1e-12
            for _ in range(70):
                middle = (low + high) / 2
                if residual(middle) > 0:
                    high = middle
                else:
                    low = middle
            hit_z = (low + high) / 2
        x, z = x + (hit_z - z) * math.tan(angle), hit_z
        assert abs(x) <= lens["clear_radius"]
        normal_angle = 0. if radius is None else math.asin(-x / radius)
        angle = normal_angle + math.asin(ni / nt * math.sin(angle - normal_angle))
        points.append([x, 0., z])
    points.append([x + (image_z - z) * math.tan(angle), 0., image_z])
    return dict(points=points, direction=[math.sin(angle), 0., math.cos(angle)])


def build():
    scalar = []
    for n in [1.3, 1.5, 1.8]:
        lens = ray.make_lens(index=n)
        for field in [0., 3., 8.]:
            for height in [-6., -2., 0., 4., 8.]:
                expected = scalar_meridional(lens, height, field, 55.)
                result = ray.pupil_ray(height, 0., field, lens, 55.)
                assert result["status"] == "ok"
                error = max(abs(a - b) for pa, pb in zip(result["points"], expected["points"]) for a, b in zip(pa, pb))
                assert error < 2e-10, error
                scalar.append(dict(index=n, field=field, height=height, image_z=55., **expected))
    bundles = []
    for shape in [-1.2, -1., 0., .7, 1., 1.2]:
        lens = ray.prescription(shape)
        assert abs(ray.paraxial(lens)["power"] - .02) < 1e-14
        for field in [0., 6.]:
            result = ray.trace_bundle(lens, 6., field, 52., 4, 12)
            bundles.append(dict(shape=shape, field=field, aperture=6., image_z=52., rings=4, spokes=12,
                                centroid=result["centroid"], rms_mm=result["rms_mm"], accepted=result["accepted"]))
    designs = []
    for fields in [[0.], [0., 3., 6.]]:
        result = ray.optimize_design(fields=fields)
        best = result["best"]
        assert best["valid"] and best["value_mm2"] < result["start"]["value_mm2"] / 4
        designs.append(dict(fields=fields, shape=best["shape"], image_z=best["image_z"], value_mm2=best["value_mm2"],
                            start_mm2=result["start"]["value_mm2"], history=[row["value_mm2"] for row in result["history"]]))
    payload = dict(convention="mm; degrees; entrance stop z=-5 mm; physical spherical caps; scalar Snell reference uses independent bisection",
                   scalar=scalar, bundles=bundles, designs=designs)
    path = ROOT / "tests/ray-references.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(f"Validated {len(scalar)} independent scalar-Snell/cap rays; wrote {len(bundles)} 3D bundle and {len(designs)} design references.")


if __name__ == "__main__":
    build()
