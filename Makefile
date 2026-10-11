# Optics in 5 Minutes: build commands. Activate the environment first (conda activate optics5min), or set
# O5_PY in config.local.sh.
#   make check-env                       is this machine ready?
#   make models                          download the speech models into .models/
#   make episode E=E03                   physics -> narration -> render -> verify (one episode)
#   make physics|audio|render|verify E=E03
#   make preview E=E03 [CH="Ch2Foo Ch3Bar"]
#   make release E=E03                   compact copies in dist/ for sharing
#   make all                             build every episode (slow: about 1 h)
#   make new NAME=E11_my_topic TITLE="My Topic"
# E accepts a folder name or its prefix (E03 -> E03_diffraction_limit). WORKERS sets render parallelism (default 3).
SHELL := /bin/bash
# PY sources common/env.sh (config.local.sh) and runs $O5_PY
PY := $(abspath common/py)
WORKERS ?= 3
EPISODES := $(sort $(wildcard E[0-9][0-9]_*))
EP = $(or $(firstword $(filter $(E) $(E)_%,$(EPISODES))),$(error E=$(E) matches no episode folder; have: $(EPISODES)))

.PHONY: help check-env models physics audio preview render verify episode release all new list

help:
	@sed -n '1,11p' Makefile | sed 's/^# \{0,1\}//'

list:
	@printf '%s\n' $(EPISODES)

check-env:
	$(PY) common/check_env.py

models:
	$(PY) common/fetch_models.py $(if $(FROM),--from $(FROM))

physics:
	cd $(EP) && nice -n 5 $(PY) physics.py

audio:
	common/audio.sh $(EP)

preview:
	common/preview.sh $(EP) $(CH)

render:
	WORKERS=$(WORKERS) common/render.sh $(EP) --skip-audio

verify:
	$(PY) common/verify.py $(EP)

episode: physics audio render verify

release:
	common/release_encode.sh $(EP)

all:
	@for e in $(EPISODES); do $(MAKE) --no-print-directory episode E=$$e || exit 1; done

new:
	common/new_episode.sh $(NAME) "$(TITLE)"
