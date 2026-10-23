"""Precompute E12's geometric lens-design film from the canonical course tracer.

Run ``python physics.py``. Everything physical shown by scenes.py is saved here;
the renderer never optimizes or traces rays. Lengths are millimetres, angles degrees.
The entrance stop is fixed at z=-5 mm, radius 6 mm; the glass has n=1.5, centre
thickness 4 mm and mechanical radius 12 mm. Shape q preserves thick-lens EFL=50 mm.
One optimized detector plane is shared by all fields in each objective.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE.parent / "teaching" / "tools"))
from ray_optics import (best_image_plane, design_merit, make_lens, optimize_design,
                        paraxial, prescription, pupil_ray, pupil_samples, surface_sag, trace_bundle)

APERTURE = 6.0
FIELDS = np.array([0.0, 3.0, 6.0])
RINGS, SPOKES = 6, 16


def profile(lens, y):
    return np.array([[surface_sag(lens["r1"], abs(v)) for v in y],
                     [lens["thickness"] + surface_sag(lens["r2"], abs(v)) for v in y]])


def bundle_arrays(lens, image_z, fields=FIELDS, rings=RINGS, spokes=SPOKES):
    bundles = [trace_bundle(lens, APERTURE, float(field), image_z, rings, spokes) for field in fields]
    assert all(b["accepted"] == rings * spokes for b in bundles)
    xy = np.array([b["xy"] for b in bundles])
    centroids = np.array([b["centroid"] for b in bundles])
    rms = np.array([b["rms_mm"] for b in bundles])
    return xy, centroids, rms


def meridians(lens, image_z, field=0.0, count=17):
    rays = [pupil_ray(float(x), 0.0, float(field), lens, image_z)
            for x in np.linspace(-APERTURE, APERTURE, count)]
    assert all(ray["status"] == "ok" and len(ray["points"]) == 4 for ray in rays)
    return np.array([ray["points"] for ray in rays])


def main():
    print("Computing E12's fixed-power singlet design experiment...", flush=True)
    axis = optimize_design(fields=(0.0,), aperture=APERTURE, iterations=28, rings=RINGS, spokes=SPOKES)
    balanced = optimize_design(fields=tuple(FIELDS), aperture=APERTURE, iterations=28, rings=RINGS, spokes=SPOKES)
    q_design = np.array([0.0, axis["best"]["shape"], balanced["best"]["shape"]])
    z_design = np.array([paraxial(prescription(0.0))["image_z"],
                         axis["best"]["image_z"], balanced["best"]["image_z"]])
    names = ["symmetric start", "on-axis objective", "three-field objective"]
    y_profile = np.linspace(-12.0, 12.0, 161)
    design_profiles, design_paths, design_spots, design_centroids, design_rms = [], [], [], [], []
    for q, z in zip(q_design, z_design):
        lens = prescription(float(q))
        xy, centroids, rms = bundle_arrays(lens, float(z))
        design_profiles.append(profile(lens, y_profile))
        design_paths.append([meridians(lens, float(z), float(field)) for field in FIELDS])
        design_spots.append(xy)
        design_centroids.append(centroids)
        design_rms.append(rms)
    design_rms = np.array(design_rms)
    assert design_rms[1, 0] < design_rms[0, 0]
    assert np.mean(design_rms[2] ** 2) < np.mean(design_rms[1] ** 2)

    q_scan = np.linspace(-1.2, 1.2, 121)
    scan_z, scan_merit, scan_rms, scan_valid, scan_profiles = [], [], [], [], []
    scan_spots, scan_centroids, scan_paths = [], [], []
    for i, q in enumerate(q_scan):
        lens = prescription(float(q))
        assert abs(paraxial(lens)["focal"] - 50.0) < 1e-10
        zs, merits, rms_fields, valid, spots, centres, paths = [], [], [], [], [], [], []
        for fields in ((0.0,), tuple(FIELDS)):
            z = best_image_plane(lens, APERTURE, fields, RINGS, SPOKES)
            result = design_merit(float(q), z, APERTURE, fields, RINGS, SPOKES)
            xy, centroid, rms = bundle_arrays(lens, z)
            zs.append(z); merits.append(result["value_mm2"]); rms_fields.append(rms)
            valid.append(result["valid"]); spots.append(xy); centres.append(centroid)
            paths.append(meridians(lens, z, 0.0))
        scan_z.append(zs); scan_merit.append(merits); scan_rms.append(rms_fields)
        scan_valid.append(valid); scan_spots.append(spots); scan_centroids.append(centres)
        scan_paths.append(paths); scan_profiles.append(profile(lens, y_profile))
        if i % 30 == 0:
            print(f"  scan {i + 1}/{len(q_scan)}", flush=True)
    assert np.all(scan_valid), "The selected fixed-aperture scan must contain no lost rays."

    # Hold each selected prescription and detector fixed during independent checks.
    validation_fields = np.array([0.0, 1.5, 3.0, 4.5, 6.0])
    validation_rms, validation_spots, validation_centroids = [], [], []
    convergence_counts = np.array([96, 384, 1536])
    convergence_rms = []
    for q, z in zip(q_design, z_design):
        lens = prescription(float(q))
        xy, centroid, rms = bundle_arrays(lens, float(z), validation_fields, 24, 64)
        validation_rms.append(rms); validation_spots.append(xy); validation_centroids.append(centroid)
        convergence_rms.append([bundle_arrays(lens, float(z), FIELDS, r, s)[2]
                                for r, s in [(6, 16), (12, 32), (24, 64)]])
    validation_rms = np.asarray(validation_rms)
    assert np.mean(validation_rms[2, [0, 2, 4]] ** 2) < np.mean(validation_rms[1, [0, 2, 4]] ** 2)
    # Full dense-field curve for the verification scene, not an optimized training target.
    field_curve = np.linspace(0.0, 6.0, 25)
    field_rms = [bundle_arrays(prescription(float(q)), float(z), field_curve, 12, 32)[2]
                 for q, z in zip(q_design, z_design)]

    # Two explicitly forbidden examples illustrate constraints, not optimization results.
    collision = make_lens(r1=18., r2=-18., thickness=4., index=1.5, clear_radius=12.)
    assert collision["edge_thickness"] < 0
    clipped = dict(prescription(float(q_design[2])))
    clipped["clear_radius"] = 5.0
    bad_bundle = trace_bundle(clipped, APERTURE, 6.0, float(z_design[2]), RINGS, SPOKES)
    assert bad_bundle["accepted"] < bad_bundle["requested"]
    bad_paths = []
    bad_status = []
    for x in np.linspace(-APERTURE, APERTURE, 17):
        ray = pupil_ray(float(x), 0., 6., clipped, float(z_design[2]))
        pts = np.full((4, 3), np.nan)
        pts[:len(ray["points"])] = ray["points"]
        bad_paths.append(pts); bad_status.append(ray["status"] == "ok")

    # Ideal diffraction reference, not the aberrated lens's PSF: scalar uniform
    # circular pupil in the paraxial focal plane, lambda=0.55 micrometres.
    from scipy.special import j1
    airy_axis_um = np.linspace(-10., 10., 384)
    airy_x, airy_y = np.meshgrid(airy_axis_um, airy_axis_um)
    airy_argument = 2 * np.pi * APERTURE * np.hypot(airy_x, airy_y) * 1e-3 / (.00055 * 50.)
    airy = (2 * j1(airy_argument) / airy_argument) ** 2
    airy_first_null_um = 3.8317059702075125 * (.00055 * 50.) / (2 * np.pi * APERTURE) * 1000

    np.savez_compressed(DATA / "design.npz",
        fields=FIELDS, aperture=APERTURE, y_profile=y_profile,
        q_design=q_design, z_design=z_design, design_profiles=design_profiles,
        design_paths=design_paths, design_spots=design_spots, design_centroids=design_centroids,
        design_rms=design_rms, q_scan=q_scan, scan_z=scan_z, scan_merit=scan_merit,
        scan_rms=scan_rms, scan_valid=scan_valid, scan_profiles=scan_profiles,
        scan_spots=scan_spots, scan_centroids=scan_centroids, scan_paths=scan_paths,
        validation_fields=validation_fields, validation_rms=validation_rms,
        validation_spots=validation_spots, validation_centroids=validation_centroids,
        convergence_counts=convergence_counts, convergence_rms=convergence_rms,
        field_curve=field_curve, field_rms=field_rms,
        collision_profile=profile(collision, y_profile), collision_edge=collision["edge_thickness"],
        clipped_y_profile=np.linspace(-5., 5., 161),
        clipped_profile=profile(clipped, np.linspace(-5., 5., 161)),
        clipped_paths=bad_paths, clipped_valid=bad_status,
        clipped_accepted=bad_bundle["accepted"], clipped_requested=bad_bundle["requested"],
        pupil_sparse=pupil_samples(APERTURE, 6, 16), pupil_dense=pupil_samples(APERTURE, 24, 64),
        airy=airy.astype(np.float32), airy_axis_um=airy_axis_um, airy_first_null_um=airy_first_null_um,
        axis_history=np.array([[v["shape"], v["image_z"], v["value_mm2"]] for v in axis["history"]]),
        balanced_history=np.array([[v["shape"], v["image_z"], v["value_mm2"]] for v in balanced["history"]]))
    report = dict(model="monochromatic exact sequential spherical ray tracing",
        assumptions=dict(n=1.5, thickness_mm=4, efl_mm=50, stop_z_mm=-5,
                         entrance_radius_mm=6, mechanical_radius_mm=12,
                         minimum_edge_mm=.5, q_bounds=[-1.2, 1.2], equal_field_weights=True),
        training_samples_per_field=RINGS*SPOKES, validation_samples_per_field=24*64,
        designs=[dict(name=name, q=float(q), image_z_mm=float(z),
                      radii_mm=[prescription(float(q))["r1"], prescription(float(q))["r2"]],
                      rms_um=(1000*rms).tolist(), combined_rms_um=float(1000*np.sqrt(np.mean(rms**2))))
                 for name, q, z, rms in zip(names, q_design, z_design, design_rms)],
        validation_fields_deg=validation_fields.tolist(), validation_rms_um=(validation_rms*1000).tolist(),
        scan_all_rays_retained=True,
        rejected_examples=dict(collision_edge_mm=collision["edge_thickness"],
                               clipped_rays=bad_bundle["requested"]-bad_bundle["accepted"]))
    (DATA / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    print("All physical checks passed; wrote data/design.npz and data/report.json.", flush=True)


if __name__ == "__main__":
    main()
