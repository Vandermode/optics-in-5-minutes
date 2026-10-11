# Optics in 5 Minutes: build commands. Incremental and fastest-route by default: each step reruns only when its
# inputs changed; physics runs on a GPU (or downloads the precomputed bundle) when it would be slow on the CPU.
#   make check-env                       is this machine ready?
#   make models                          download the speech models into .models/
#   make episode E=E03                   data -> narration -> render -> verify (skips steps that are up to date)
#   make physics|audio|render|verify E=E03      one step (FORCE=1 reruns it even if up to date)
#   make preview E=E03 [CH="Ch2Foo Ch3Bar"]
#   make release [E=E03]                 the shareable 1080p copy in dist/ (all episodes without E)
#   make pack-data [E=E03]               physics data bundles in dist/data/ (attach to a release; see O5_DATA_URL)
#   make all                             every episode
#   make new NAME=E11_my_topic TITLE="My Topic"
# E accepts a folder name or its prefix (E03 -> E03_diffraction_limit). WORKERS = render processes (default 6).
SHELL := /bin/bash
# PY sources common/env.sh (config.local.sh) and runs $O5_PY
PY := $(abspath common/py)
WORKERS ?= 6
EPISODES := $(sort $(wildcard E[0-9][0-9]_*))
EP = $(or $(firstword $(filter $(E) $(E)_%,$(EPISODES))),$(error E=$(E) matches no episode folder; have: $(EPISODES)))
lc = $(shell echo $(1) | tr 'A-Z' 'a-z')
MP4 = $(EP)/$(call lc,$(EP))_1080p60.mp4
# renderer and series code: a change re-renders every episode
RENDER_DEPS := common/o5common.py common/o5env.py common/fastrender.py common/fast_render.py

.PHONY: help list check-env models physics audio render verify episode preview release pack-data all new

help:
	@sed -n '1,13p' Makefile | sed 's/^# \{0,1\}//'

list:
	@printf '%s\n' $(EPISODES)

# ---- per-episode file targets: data/.done -> audio/narration.wav -> <slug>_1080p60.mp4 -> .verify/report.txt ----
define EPISODE_RULES
$(1)/data/.done: $(1)/physics.py
	$(PY) common/run_physics.py $(1)
$(1)/audio/narration.wav: $(1)/narration.json
	common/audio.sh $(1)
$(1)/$(call lc,$(1))_1080p60.mp4: $(1)/scenes.py $(1)/episode.json $(1)/audio/narration.wav $(1)/data/.done \
                                  $(1)/subtitles.ass $(RENDER_DEPS)
	WORKERS=$$(WORKERS) common/render.sh $(1) --skip-audio
$(1)/.verify/report.txt: $(1)/$(call lc,$(1))_1080p60.mp4
	mkdir -p $(1)/.verify && $(PY) common/verify.py $(1) | tee $(1)/.verify/report.tmp && mv $(1)/.verify/report.tmp $$@
endef
$(foreach e,$(EPISODES),$(eval $(call EPISODE_RULES,$(e))))

physics:
	@$(MAKE) --no-print-directory $(if $(FORCE),-W $(EP)/physics.py) $(EP)/data/.done
audio:
	@$(MAKE) --no-print-directory $(if $(FORCE),-W $(EP)/narration.json) $(EP)/audio/narration.wav
render:
	@$(MAKE) --no-print-directory $(if $(FORCE),-W $(EP)/scenes.py) $(MP4)
verify:
	@$(MAKE) --no-print-directory $(if $(FORCE),-W $(MP4)) $(EP)/.verify/report.txt
episode:
	@$(MAKE) --no-print-directory $(EP)/.verify/report.txt
	@tail -1 $(EP)/.verify/report.txt

check-env:
	$(PY) common/check_env.py

models:
	$(PY) common/fetch_models.py $(if $(FROM),--from $(FROM))

preview:
	common/preview.sh $(EP) $(CH)

release:
	@for e in $(if $(E),$(EP),$(EPISODES)); do common/release_encode.sh $$e; done

pack-data:
	@for e in $(if $(E),$(EP),$(EPISODES)); do $(PY) common/data_bundle.py pack $$e; done

all:
	@for e in $(EPISODES); do $(MAKE) --no-print-directory episode E=$$e || exit 1; done

new:
	common/new_episode.sh $(NAME) "$(TITLE)"
