"""Precompute E13's fixed-power differentiable Fourier-optics experiment.

Run ``python physics.py``. Scenes only load the resulting arrays; they never run
the optimizer. The model and defaults match teaching/shared/differentiable-optics.js
and the independent Python lab. The aperture has |x| < 1 and fixed sum(A**2)=1.
The phase is a*x+b*x**2, with a and b in radians. The unitary Fourier transform
retains every detector sample, so the total detected power is always one.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

for _key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_key, "1")

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
N = 128
START = np.array([1.5, 2.0])
DIFFICULT_START = np.array([3.0, 3.0])


def fourier(pupil):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(pupil), norm="ortho"))


def adjoint(field):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(field), norm="ortho"))


def grid(n=N):
    if n < 16 or n % 8:
        raise ValueError("Use n divisible by eight, with n >= 16.")
    x = 8 * (np.arange(n) - (n - 1) / 2) / n
    amplitude = (np.abs(x) < 1).astype(float)
    amplitude /= np.linalg.norm(amplitude)
    frequency = (np.arange(n) - n // 2) / 8
    return x, amplitude, frequency


def forward(a=1.5, b=2.0, n=N, levels=None, piston=0.0):
    x, amplitude, frequency = grid(n)
    phase = a * x + b * x**2 + piston
    if levels is not None:
        if not isinstance(levels, (int, np.integer)) or levels < 2:
            raise ValueError("At least two phase levels are required.")
        step = 2 * np.pi / levels
        phase = step * np.floor(phase / step + 0.5)
    pupil = amplitude * np.exp(1j * phase)
    field = fourier(pupil)
    return dict(x=x, amplitude=amplitude, frequency=frequency, phase=phase,
                pupil=pupil, field=field, intensity=np.abs(field)**2)


def evaluate(a=1.5, b=2.0, target=0.0, n=N):
    result = forward(a, b, n)
    target_intensity = forward(target, 0.0, n)["intensity"]
    scale = np.sum(target_intensity**2)
    residual = result["intensity"] - target_intensity
    sensor_sensitivity = residual * result["field"] / scale
    back = adjoint(sensor_sensitivity)
    phase_gradient = 2 * np.imag(np.conj(result["pupil"]) * back)
    gradient = np.array([np.sum(phase_gradient * result["x"]),
                         np.sum(phase_gradient * result["x"]**2)])
    result.update(a=a, b=b, target=target, n=n, target_intensity=target_intensity,
                  scale=scale, residual=residual, sensor_sensitivity=sensor_sensitivity,
                  back=back, phase_gradient=phase_gradient, gradient=gradient,
                  loss=0.5 * np.sum(residual**2) / scale,
                  energy=np.sum(result["intensity"]))
    return result


def finite_difference(a=1.5, b=2.0, target=0.0, n=N, h=1e-5):
    theta = np.array([a, b])
    return np.array([(evaluate(*(theta + h * direction), target, n)["loss"] -
                      evaluate(*(theta - h * direction), target, n)["loss"]) / (2 * h)
                     for direction in np.eye(2)])


def tangent_gradient(result):
    """An independent forward derivative, without using the adjoint field."""
    return np.array([
        2 * np.real(np.sum(result["residual"] * np.conj(result["field"]) *
                           fourier(1j * result["x"]**power * result["pupil"]))) /
        result["scale"] for power in [1, 2]])


def optimize(a=1.5, b=2.0, target=0.0, n=N, iterations=40, learning_rate=12.0):
    best = evaluate(a, b, target, n)
    history = [[0, a, b, best["loss"], 0.0]]
    states = [best]
    for iteration in range(iterations):
        gradient = best["gradient"]
        norm2 = np.sum(gradient**2)
        if norm2 < 1e-22:
            break
        step = learning_rate
        for _ in range(24):
            candidate = evaluate(best["a"] - step * gradient[0],
                                 best["b"] - step * gradient[1], target, n)
            if candidate["loss"] <= best["loss"] - 1e-4 * step * norm2:
                best = candidate
                break
            step *= 0.5
        else:
            break
        history.append([iteration + 1, best["a"], best["b"], best["loss"], step])
        states.append(best)
    return best, np.array(history), states


def validation():
    initial = evaluate()
    fd = finite_difference()
    tangent = tangent_gradient(initial)
    assert abs(initial["energy"] - 1.0) < 1e-13
    assert evaluate(0.0, 0.0)["loss"] == 0.0
    assert np.max(np.abs(tangent - initial["gradient"])) < 1e-13
    assert np.max(np.abs(fd - initial["gradient"])) < 1e-9
    assert abs(np.sum(initial["phase_gradient"])) < 1e-13
    assert np.max(np.abs(forward(piston=0.71)["intensity"] - initial["intensity"])) < 1e-14
    assert abs(forward(0.0, 0.0)["intensity"][N // 2] - 0.25) < 1e-14
    finer = evaluate(n=256)
    assert abs(finer["loss"] - initial["loss"]) < 5e-4
    assert np.max(np.abs(finer["gradient"] - initial["gradient"])) < 5e-4
    gradient_checks = []
    for a, b, target in [(1.5, 2.0, 0.0), (3.0, 3.0, 0.0),
                         (-0.8, 1.3, -1.5), (0.4, -0.9, 1.0)]:
        result = evaluate(a, b, target)
        error = float(np.max(np.abs(finite_difference(a, b, target) - result["gradient"])))
        assert error < 1e-9
        assert np.max(np.abs(tangent_gradient(result) - result["gradient"])) < 1e-13
        gradient_checks.append(dict(a=a, b=b, target=target, max_absolute_error=error))
    return initial, dict(
        finite_difference_h_rad=1e-5, finite_difference_gradient=fd.tolist(),
        tangent_gradient=tangent.tolist(), gradient_checks=gradient_checks,
        piston_gradient=float(np.sum(initial["phase_gradient"])),
        grid_refinement_loss_change=float(finer["loss"] - initial["loss"]),
        grid_refinement_gradient_change=(finer["gradient"] - initial["gradient"]).tolist(),
        zero_loss_reference=float(evaluate(0.0, 0.0)["loss"]),
        uniform_pupil_center_power=float(forward(0.0, 0.0)["intensity"][N // 2]))


def main():
    initial, checks = validation()
    best, history, states = optimize()
    difficult, difficult_history, difficult_states = optimize(*DIFFICULT_START)
    assert np.all(np.diff(history[:, 3]) <= 1e-14)
    assert np.all(np.diff(difficult_history[:, 3]) <= 1e-14)
    assert best["loss"] < initial["loss"] / 1000
    assert difficult["loss"] > 0.1
    assert history[1, 2] > history[0, 2], "First accepted update increases defocus."
    energy_errors = np.array([abs(state["energy"] - 1) for state in states + difficult_states])
    assert np.max(energy_errors) < 1e-13
    fd_h = np.logspace(-1, -10, 37)
    fd_gradients = np.array([finite_difference(h=h) for h in fd_h])
    fd_errors = np.max(np.abs(fd_gradients - initial["gradient"]), axis=1)

    # A loss landscape and exact parameter slices support gradient visualizations.
    landscape_a = np.linspace(-1.5, 4.0, 111)
    landscape_b = np.linspace(-1.0, 6.0, 113)
    landscape_loss = np.array([[evaluate(a, b)["loss"] for a in landscape_a]
                               for b in landscape_b])
    slice_delta = np.linspace(-0.5, 0.5, 121)
    slice_loss = np.array([[evaluate(*(START + delta * direction))["loss"]
                            for delta in slice_delta] for direction in np.eye(2)])
    sweep_a = np.linspace(-2.5, 2.5, 81)
    sweep_b = np.linspace(0.0, 3.5, 81)
    sweep_a_states = [forward(a, 0.0) for a in sweep_a]
    sweep_b_states = [forward(0.0, b) for b in sweep_b]

    # A shifted target makes quantization nontrivial: a flat phase is not optimal.
    shifted, shifted_history, _ = optimize(target=1.0, iterations=80)
    levels = np.array([2, 4, 8, 16])
    quantized = [forward(shifted["a"], shifted["b"], levels=int(level)) for level in levels]
    quantized_loss = np.array([0.5 * np.sum((state["intensity"] - shifted["target_intensity"])**2) /
                              shifted["scale"] for state in quantized])
    assert all(abs(np.sum(state["intensity"]) - 1) < 1e-13 for state in quantized)
    q0 = forward(0.83, 0.71, levels=4)["intensity"]
    assert np.array_equal(q0, forward(0.83 + 1e-7, 0.71, levels=4)["intensity"])
    assert np.array_equal(q0, forward(0.83 - 1e-7, 0.71, levels=4)["intensity"])

    arrays = dict(
        x=initial["x"], frequency=initial["frequency"], amplitude=initial["amplitude"],
        target_intensity=initial["target_intensity"], target_scale=initial["scale"],
        initial_parameters=START, initial_phase=initial["phase"],
        initial_pupil=initial["pupil"], initial_field=initial["field"],
        initial_intensity=initial["intensity"], initial_residual=initial["residual"],
        initial_sensor_sensitivity=initial["sensor_sensitivity"], initial_back=initial["back"],
        initial_phase_gradient=initial["phase_gradient"], initial_gradient=initial["gradient"],
        initial_loss=initial["loss"], history=history,
        history_phase=np.array([state["phase"] for state in states]),
        history_intensity=np.array([state["intensity"] for state in states]),
        history_gradient=np.array([state["gradient"] for state in states]),
        history_phase_gradient=np.array([state["phase_gradient"] for state in states]),
        history_energy=np.array([state["energy"] for state in states]),
        final_parameters=np.array([best["a"], best["b"]]), final_phase=best["phase"],
        final_intensity=best["intensity"], final_loss=best["loss"],
        difficult_history=difficult_history,
        difficult_history_intensity=np.array([state["intensity"] for state in difficult_states]),
        difficult_history_gradient=np.array([state["gradient"] for state in difficult_states]),
        difficult_initial_intensity=difficult_states[0]["intensity"],
        difficult_intensity=difficult["intensity"], difficult_loss=difficult["loss"],
        fd_h=fd_h, fd_gradient=fd_gradients, fd_error=fd_errors,
        check_fd_gradient=finite_difference(), check_tangent_gradient=tangent_gradient(initial),
        landscape_a=landscape_a, landscape_b=landscape_b, landscape_loss=landscape_loss,
        slice_delta=slice_delta, slice_loss=slice_loss,
        sweep_a=sweep_a, sweep_a_phase=np.array([state["phase"] for state in sweep_a_states]),
        sweep_a_intensity=np.array([state["intensity"] for state in sweep_a_states]),
        sweep_b=sweep_b, sweep_b_phase=np.array([state["phase"] for state in sweep_b_states]),
        sweep_b_intensity=np.array([state["intensity"] for state in sweep_b_states]),
        quantization_levels=levels,
        quantization_phase=np.array([state["phase"] for state in quantized]),
        quantization_intensity=np.array([state["intensity"] for state in quantized]),
        quantization_loss=quantized_loss, quantization_target=shifted["target_intensity"],
        quantization_smooth_phase=shifted["phase"], quantization_smooth_intensity=shifted["intensity"],
        quantization_smooth_loss=shifted["loss"], quantization_parameters=np.array([shifted["a"], shifted["b"]]),
        shifted_history=shifted_history)
    np.savez_compressed(DATA / "differentiable.npz", **arrays)
    report = dict(
        model="monochromatic one-dimensional scalar phase pupil with unitary Fourier propagation",
        assumptions=dict(n=N, aperture_half_width=1, computational_width=8,
                         phase="a*x + b*x**2", parameter_units="rad",
                         frequency_units="cycles per pupil half-width", fixed_pupil_power=1,
                         target_parameters=[0, 0], loss="sum((I-T)**2)/(2*sum(T**2))",
                         retains_all_detector_samples=True,
                         sensor_noise=False, reconstruction=False),
        history_columns=["accepted_step", "a_rad", "b_rad", "dimensionless_loss", "step_size"],
        initial=dict(parameters=START.tolist(), loss=float(initial["loss"]),
                     gradient=initial["gradient"].tolist(), power=float(initial["energy"])),
        first_update=dict(parameters=history[1, 1:3].tolist(), loss=float(history[1, 3]),
                          step_size=float(history[1, 4])),
        final=dict(parameters=[float(best["a"]), float(best["b"])], loss=float(best["loss"]),
                   accepted_steps=len(history)-1,
                   loss_reduction_factor=float(initial["loss"] / best["loss"])),
        difficult=dict(initial_parameters=DIFFICULT_START.tolist(),
                       final_parameters=[float(difficult["a"]), float(difficult["b"])],
                       initial_loss=float(difficult_history[0, 3]), final_loss=float(difficult["loss"]),
                       final_gradient=difficult["gradient"].tolist(), accepted_steps=len(difficult_history)-1),
        quantization=dict(target_parameters=[1, 0],
                          smooth_parameters=[float(shifted["a"]), float(shifted["b"])],
                          smooth_loss=float(shifted["loss"]), levels=levels.tolist(),
                          losses=quantized_loss.tolist(), zero_local_difference_verified=True),
        checks=dict(**checks, max_power_error=float(np.max(energy_errors)),
                    monotone_accepted_losses=True, first_update_increases_defocus=True),
        cache_keys={key: list(np.shape(value)) for key, value in arrays.items()})
    (DATA / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in ["initial", "first_update", "final", "difficult", "quantization"]}, indent=2))
    print("All power, gradient, refinement, optimizer, and quantization checks passed.")
    print("Wrote data/differentiable.npz and data/report.json.")


if __name__ == "__main__":
    main()
