"""Package the complete buildless course without development environments or tests."""
from pathlib import Path
import argparse
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
ALLOWED = {".html", ".css", ".js", ".json", ".md", ".ipynb", ".png", ".svg", ".ttf", ".txt", ".zip"}
SKIP = {".qa", "node_modules", "test-results", "tests", "tools", "__pycache__"}


def export(destination, include_local_media=False):
    destination = Path(destination).resolve()
    if destination == ROOT or ROOT in destination.parents:
        raise ValueError("Export outside teaching/ to avoid recursively copying the output.")
    videos = json.loads((ROOT / 'videos.json').read_text())
    local_videos = [episode for episode, video in videos.items() if not video.get('youtubeId')]
    if local_videos and not include_local_media:
        raise ValueError('Local film previews remain for ' + ', '.join(local_videos)
                         + '. Supply published film metadata before exporting for the website, or use --include-local-media for an offline review package.')
    allowed = ALLOWED | ({'.mp4', '.vtt'} if include_local_media else set())
    if include_local_media:
        slugs = {entry['id']: entry['slug'] for entry in json.loads((ROOT / 'course.json').read_text())}
        for episode in local_videos:
            for key in ('file', 'captions'):
                source = (ROOT / slugs[episode] / videos[episode][key]).resolve()
                if ROOT not in source.parents or not source.is_file():
                    raise ValueError(f'Missing local review media: {episode} {key}')
    destination.mkdir(parents=True, exist_ok=True)
    written = []
    for source in sorted(ROOT.rglob("*")):
        relative = source.relative_to(ROOT)
        if not source.is_file() or any(part.startswith(".") or part in SKIP for part in relative.parts):
            continue
        if source.suffix not in allowed or source.name.startswith("package"):
            continue
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        written.append(str(relative))
    if include_local_media:
        # The review folder must be usable independently of the source checkout.
        server = Path('tools/serve_course.py')
        (destination / server).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / server, destination / server)
        written.append(str(server))
        readme = destination / 'README.md'
        readme.write_text(
            '# Open this local review package\n\n'
            'From this folder, run `python tools/serve_course.py --port 8787`, '
            'then open http://127.0.0.1:8787/ in your browser. The server supports '
            'film seeking. Python 3 is the only server requirement.\n\n'
            'The build and QA instructions below refer to the production source '
            'checkout; they are not needed to view this package.\n\n---\n\n'
            + readme.read_text())
    # Remove only files from the previous export's manifest, never arbitrary checkout files.
    manifest = destination / "course-files.json"
    if manifest.exists():
        for stale in set(json.loads(manifest.read_text())) - set(written):
            candidate = (destination / stale).resolve()
            if destination in candidate.parents and candidate.is_file():
                candidate.unlink()
    manifest.write_text(json.dumps(written, indent=2) + "\n")
    assert (destination / "index.html").is_file()
    for entry in json.loads((ROOT / "course.json").read_text()):
        for file in ["index.html", entry["notebook"], "figures.zip", "INSTRUCTOR_NOTES.md"]:
            assert (destination / entry["slug"] / file).is_file(), f"Missing public artifact: {entry['slug']}/{file}"
    print(f"Packaged {len(written)} static files in {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("destination")
    parser.add_argument('--include-local-media', action='store_true', help='Include local film previews in an offline review package.')
    args = parser.parse_args()
    export(args.destination, args.include_local_media)
