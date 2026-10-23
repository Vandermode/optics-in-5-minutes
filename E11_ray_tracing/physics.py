"""Precompute E11's geometric rays using the course's canonical 3-D tracer.

All lengths are millimetres. The film shows meridional (z,x) slices of a
spherical singlet and equal-area circular-pupil spot samples. No image is a
wave-optical PSF. Run once before rendering; scenes only load the arrays.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(name, "1")

import numpy as np
from scipy.special import j1

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "teaching" / "tools"))
from ray_optics import (best_image_plane, make_lens, paraxial, pupil_ray,
                        refract, surface_sag, trace_bundle)

DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
LENS = make_lens(r1=50., r2=-50., thickness=4., index=1.5, clear_radius=12.)
PARAXIAL = paraxial(LENS)
IMAGE_Z = PARAXIAL["image_z"]
APERTURE = 10.
RINGS, SPOKES = 8, 24


def fan(aperture, angle=0.):
    rays = [pupil_ray(float(x), 0., float(angle), LENS, 70.)
            for x in np.linspace(-aperture, aperture, 19)]
    assert all(r["status"] == "ok" for r in rays), (aperture, angle)
    return np.asarray([r["points"] for r in rays], dtype=float)


def bundle(aperture, angle, image_z):
    result = trace_bundle(LENS, float(aperture), float(angle), float(image_z), RINGS, SPOKES)
    assert result["accepted"] == RINGS * SPOKES, (aperture, angle, result["accepted"])
    return np.asarray(result["xy"], dtype=float), result


def main():
    apertures = np.linspace(1., APERTURE, 181)
    fans = np.asarray([fan(a) for a in apertures])
    aperture_rms = np.asarray([bundle(a, 0., IMAGE_Z)[1]["rms_mm"] for a in apertures])
    best_z = best_image_plane(LENS, APERTURE, fields=(0.,), rings=RINGS, spokes=SPOKES)
    focus_zs = np.linspace(best_z - 1.5, IMAGE_Z + .65, 241)
    focus_results = [bundle(APERTURE, 0., z) for z in focus_zs]
    focus_xy = np.asarray([r[0] for r in focus_results])
    focus_rms = np.asarray([r[1]["rms_mm"] for r in focus_results])
    best_xy, best_bundle = bundle(APERTURE, 0., best_z)
    paraxial_xy, paraxial_bundle = bundle(APERTURE, 0., IMAGE_Z)
    aperture_best_xy = np.asarray([bundle(a, 0., best_z)[0] for a in apertures])
    fields = np.linspace(0., 8., 161)
    field_fans = np.asarray([fan(APERTURE, f) for f in fields])
    field_results = [bundle(APERTURE, f, best_z) for f in fields]
    field_xy = np.asarray([r[0] for r in field_results])
    field_rms = np.asarray([r[1]["rms_mm"] for r in field_results])
    field_centroids = np.asarray([r[1]["centroid"] for r in field_results])

    heights = np.linspace(-LENS["clear_radius"], LENS["clear_radius"], 401)
    front = np.asarray([surface_sag(LENS["r1"], abs(x)) for x in heights])
    back = LENS["thickness"] + np.asarray([surface_sag(LENS["r2"], abs(x)) for x in heights])
    selected = pupil_ray(8., 0., 0., LENS, 70.)
    selected_normals = np.asarray([s["normal"] for s in selected["surfaces"]])
    selected_directions = np.asarray([selected["surfaces"][0]["incoming"],
                                     selected["surfaces"][0]["outgoing"], selected["direction"]])
    roots = np.asarray([50. - np.sqrt(50.**2 - 8.**2), 50. + np.sqrt(50.**2 - 8.**2)])
    blocked = pupil_ray(14., 0., 0., LENS, IMAGE_Z)
    assert blocked["status"] == "blocked"

    # A separate, ideal wave-optical reference illustrates the model boundary.
    # These are not a PSF calculated from this singlet's geometric spot.
    psf_x_um = np.linspace(-14., 14., 1001)
    psf = []
    for aperture in apertures:
        v = 2*np.pi*aperture*(psf_x_um*1e-3)/(.00055*PARAXIAL["focal"])
        amplitude = np.ones_like(v)
        np.divide(2*j1(v), v, out=amplitude, where=abs(v)>1e-12)
        psf.append(amplitude*amplitude)

    theta = np.linspace(0., 65., 261)
    incident = np.column_stack([np.sin(np.deg2rad(theta)), np.zeros_like(theta),
                                np.cos(np.deg2rad(theta))])
    snell_in, snell_out, tir = [], [], []
    for direction in incident:
        air = refract(direction.tolist(), [0., 0., 1.], 1., 1.5)
        glass = refract(direction.tolist(), [0., 0., 1.], 1.5, 1.)
        snell_in.append(air["direction"])
        snell_out.append(glass.get("direction", [np.nan, np.nan, np.nan]))
        tir.append(glass["status"] == "tir")

    ray = selected
    marginal = pupil_ray(APERTURE, 0., 0., LENS, IMAGE_Z)
    axis_cross = marginal["exit"][2] - marginal["exit"][0] * marginal["direction"][2] / marginal["direction"][0]
    assert axis_cross < best_z < IMAGE_Z
    assert best_bundle["rms_mm"] < paraxial_bundle["rms_mm"]
    assert aperture_rms[-1] > aperture_rms[0]
    assert np.all(np.isfinite(fans)) and np.all(np.isfinite(field_xy))
    assert np.max(abs(np.linalg.norm(selected_directions, axis=1) - 1)) < 1e-12

    np.savez_compressed(DATA / "rays.npz", apertures=apertures, fan_paths=fans,
        aperture_rms=aperture_rms, image_z=IMAGE_Z, focal=PARAXIAL["focal"], bfl=PARAXIAL["bfl"],
        best_z=best_z, focus_zs=focus_zs, focus_xy=focus_xy, focus_rms=focus_rms,
        best_xy=best_xy, paraxial_xy=paraxial_xy, aperture_best_xy=aperture_best_xy,
        fields=fields, field_fans=field_fans,
        field_xy=field_xy, field_rms=field_rms, field_centroids=field_centroids,
        heights=heights, front=front, back=back, selected_path=np.asarray(ray["points"]),
        selected_normals=selected_normals, selected_directions=selected_directions,
        sphere_roots=roots, theta=theta, incident=incident, air_glass=np.asarray(snell_in),
        glass_air=np.asarray(snell_out), tir=np.asarray(tir), marginal_cross=axis_cross,
        blocked_path=np.asarray(blocked["points"]), psf_x_um=psf_x_um, ideal_psf=np.asarray(psf))
    summary = dict(lens=LENS, paraxial=PARAXIAL, best_image_z_mm=best_z,
        marginal_axis_crossing_z_mm=axis_cross, aperture_radius_mm=APERTURE,
        spot_samples=RINGS*SPOKES, paraxial_plane_rms_mm=paraxial_bundle["rms_mm"],
        best_plane_rms_mm=best_bundle["rms_mm"], field_8deg_rms_mm=float(field_rms[-1]),
        maxwell_or_diffraction_psf=False)
    (DATA / "metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print("E11 precomputation passed: finite traces, unit directions, focus ordering, common pupil samples.")


if __name__ == "__main__":
    main()
