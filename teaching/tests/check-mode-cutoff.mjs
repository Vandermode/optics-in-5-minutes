import assert from 'node:assert/strict';
import {compute,slabModes} from '../e08-trapping-light/model.js';

const state={width:5,mode:2,kappa:.025,coupling:.85,loss:.85};
const supported=compute(state);
assert.equal(supported.numerical.guided_count,3);
assert.equal(supported.plots[0].series.length,1);
assert.equal(supported.plots[0].series[0].label,'TE mode 2');
assert.ok(supported.numerical.selected_neff>1.45);

// The analytic symmetric-slab TE_m cutoff occurs at V = m*pi/2.
// Follow the same requested mode across that physical threshold, keeping the
// unrelated coupler and ring settings fixed.
const cutoff=state.mode*1.55/(2*Math.sqrt(1.5**2-1.45**2));
assert.equal(slabModes(cutoff+.01).length,3);
assert.equal(slabModes(cutoff-.01).length,2);
const missing=compute({...state,width:cutoff-.01});
assert.equal(missing.numerical.requested_mode,2);
assert.equal(missing.numerical.selected_neff,null);
assert.deepEqual(missing.plots[0].series,[],'A disappeared mode must not be replaced by a lower mode.');
assert.ok(missing.plots[0].xRange.every(Number.isFinite));
assert.ok(missing.plots[0].yRange.every(Number.isFinite));
assert.match(missing.plots[0].caption,/mode 2 is not guided/);
assert.match(missing.metrics[0].value,/—/);
assert.deepEqual(missing.plots.slice(1),supported.plots.slice(1),'The coupler and ring are separate reduced models.');

const recovered=compute({...state,width:cutoff+.01});
assert.equal(recovered.plots[0].series[0].label,'TE mode 2');
assert.ok(recovered.numerical.selected_neff>1.45);
assert.equal(compute({...state,width:.6,mode:0}).plots[0].series.length,1,'The fundamental symmetric-slab mode has no positive-thickness cutoff.');
console.log('PASS E08: a requested mode disappears at its analytic cutoff and returns without changing its index.');
