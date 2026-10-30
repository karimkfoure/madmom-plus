#!/usr/bin/env python3
"""Render the compact recorded native benchmark data as a small SVG chart."""

import argparse
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path,
                        default=ROOT / 'docs/benchmarks/native-summary.json')
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'docs/images/native-performance.svg')
    args = parser.parse_args()
    report = json.loads(args.data.read_text())['comparisons'][-1]
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" width="760" height="350" viewBox="0 0 760 350" role="img" aria-labelledby="title desc">',
             '<title id="title">Native streaming optimization: CPU and memory</title>',
             '<desc id="desc">Historical comparison of commits 38c4b63 and 20e6bc1 on AMD Ryzen 9 5900X. Streaming CPU decreases while peak process memory changes little.</desc>',
             '<rect width="760" height="350" fill="#fff"/>',
             '<g font-family="sans-serif" font-size="14" fill="#14253d">',
             '<text x="22" y="29" font-size="19" font-weight="bold">Native streaming optimization</text>',
             '<text x="22" y="52">AMD Ryzen 9 5900X · one core · five alternating trials</text>',
             '<rect x="23" y="73" width="15" height="15" fill="#9babc0"/><text x="46" y="86">Before: 38c4b63</text>',
             '<rect x="250" y="73" width="15" height="15" fill="#167c80"/><text x="273" y="86">After: 20e6bc1</text>']
    for x, metric, title, limit, scale in [(22, 'cpu_seconds_per_audio_second', 'CPU seconds / audio second', .06, 1), (410, 'peak_rss_bytes', 'Peak process memory (MiB)', 100, 2**20)]:
        parts.append('<text x="%d" y="119" font-weight="bold">%s</text>' % (x, escape(title)))
        for index, stage in enumerate(['stream', 'bulk']):
            y = 147 + index * 78
            parts.append('<text x="%d" y="%d">%s</text>' % (x, y, 'Frame at a time' if stage == 'stream' else 'Causal bulk'))
            for variant, color, dy in [('before','#9babc0',8),('after','#167c80',29)]:
                value=report['stages'][stage][variant][metric] / scale
                width=value / limit * 230
                label=('%.5f' if scale == 1 else '%.2f') % value
                parts.append('<rect x="%d" y="%d" width="%.2f" height="17" fill="%s"/><text x="%.2f" y="%d">%s</text>' % (x,y+dy,width,color,x+width+7,y+dy+14,label))
    parts.extend(['<text x="22" y="324" font-size="12">Repeated short public audio; loading excluded; memory includes Python and dependencies.</text>', '</g></svg>'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text('\n'.join(parts)+'\n')


if __name__ == '__main__':
    main()
