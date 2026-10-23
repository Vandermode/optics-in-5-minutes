"""Serve the course locally, including byte-range requests for film seeking."""
import argparse
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


class CourseHandler(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Accept-Ranges', 'bytes')
        super().end_headers()

    def send_head(self):
        self.range_remaining = None
        requested = self.headers.get('Range', '')
        match = re.fullmatch(r'bytes=(\d*)-(\d*)', requested)
        path = Path(self.translate_path(self.path))
        # Directory redirects, conditional requests and ordinary assets retain
        # the standard server behavior. Unsupported ranges may receive the full file.
        if self.command != 'GET' or not match or not any(match.groups()) or self.headers.get('If-Range') or not path.is_file():
            return super().send_head()
        try:
            source = path.open('rb')
        except OSError:
            return super().send_head()
        info = path.stat()
        size = info.st_size
        first, last = match.groups()
        if first:
            start = int(first)
            end = min(int(last), size - 1) if last else size - 1
        else:
            start, end = max(0, size - int(last)), size - 1
        if start >= size or end < start:
            source.close()
            self.send_response(416)
            self.send_header('Content-Range', f'bytes */{size}')
            self.send_header('Content-Length', '0')
            self.end_headers()
            return None
        self.range_remaining = end - start + 1
        self.send_response(206)
        self.send_header('Content-Type', self.guess_type(str(path)))
        self.send_header('Content-Range', f'bytes {start}-{end}/{size}')
        self.send_header('Content-Length', str(self.range_remaining))
        self.send_header('Last-Modified', self.date_time_string(info.st_mtime))
        self.end_headers()
        source.seek(start)
        return source

    def copyfile(self, source, outputfile):
        if self.range_remaining is None:
            return super().copyfile(source, outputfile)
        while self.range_remaining:
            chunk = source.read(min(65536, self.range_remaining))
            if not chunk:
                break
            outputfile.write(chunk)
            self.range_remaining -= len(chunk)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8787)
    parser.add_argument('--bind', default='127.0.0.1')
    parser.add_argument('--directory', type=Path, default=ROOT)
    args = parser.parse_args()
    with ThreadingHTTPServer((args.bind, args.port), partial(CourseHandler, directory=str(args.directory))) as server:
        print(f'Course preview: http://{args.bind}:{args.port}/', flush=True)
        server.serve_forever()
