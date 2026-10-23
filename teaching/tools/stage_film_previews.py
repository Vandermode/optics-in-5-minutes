"""Copy finished release films and WebVTT captions into the ignored local preview directory."""
from pathlib import Path
import argparse
import json
import re
import shutil

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = ROOT.parent


def stage(episodes):
    videos = json.loads((ROOT / 'videos.json').read_text())
    catalog = {entry['id']: entry for entry in json.loads((ROOT / 'course.json').read_text())}
    selected = episodes or [episode for episode, video in videos.items() if video.get('file')]
    for episode in selected:
        video, entry = videos[episode], catalog[episode]
        folders = list(REPOSITORY.glob(episode + '_*'))
        if len(folders) != 1:
            raise ValueError(f'Expected one production source folder for {episode}')
        folder = folders[0]
        slug = json.loads((folder / 'episode.json').read_text())['slug']
        movie = REPOSITORY / 'dist' / (slug + '_1080p.mp4')
        if not movie.is_file():
            raise FileNotFoundError(f'Build the release first: make release E={episode}')
        target = (ROOT / entry['slug'] / video['file']).resolve()
        captions = (ROOT / entry['slug'] / video['captions']).resolve()
        if ROOT not in target.parents or ROOT not in captions.parents:
            raise ValueError('Local film and caption paths must stay inside teaching/')
        target.parent.mkdir(parents=True, exist_ok=True)
        captions.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(movie, target)
        srt = (folder / 'subtitles.srt').read_text(encoding='utf-8-sig').replace('\r\n', '\n')
        vtt = re.sub(r'(?m)^(\d{2}:\d{2}:\d{2}),(\d{3}) --> (\d{2}:\d{2}:\d{2}),(\d{3})', r'\1.\2 --> \3.\4', srt)
        captions.write_text('WEBVTT\n\n' + vtt, encoding='utf-8')
        print(f'Staged {episode}: {target.relative_to(ROOT)} and English captions')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('episodes', nargs='*', help='Defaults to films with local preview metadata')
    stage(parser.parse_args().episodes)
