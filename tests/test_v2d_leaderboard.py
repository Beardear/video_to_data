"""Exercise leaderboard sorting and baseline notes in Chrome."""

import argparse
import html
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTE = "We use a commercial friendly recreation of CARI4D with retrained weights and some non-commercial modules replaced. Performance may differ slightly from the original implementation."

CHECKS = r'''
setTimeout(() => {
  const failures = [];
  let directionsChecked = 0;
  function check(ok, label) { if (!ok) failures.push(label); }
  function rows() {
    return [...document.querySelectorAll('tbody tr')].map(tr => {
      const cells = tr.querySelectorAll('td');
      return { name: cells[1].querySelector('div').firstChild.textContent,
               rank: cells[0].textContent, element: tr };
    });
  }
  payload.tracks.forEach((track, tabIndex) => {
    document.querySelectorAll('[role="tab"]')[tabIndex].click();
    track.metrics.forEach((metric, metricIndex) => {
      const header = () => document.querySelectorAll('th[role="columnheader"]')[metricIndex];
      const bestDirection = metric.higher_is_better ? 'descending' : 'ascending';
      if (header().getAttribute('aria-sort') !== bestDirection) header().click();
      if (header().getAttribute('aria-sort') !== bestDirection) header().click();
      ['best', 'reverse'].forEach(direction => {
        if (direction === 'reverse') header().click();
        const visible = rows();
        const participants = visible.filter(r => ['Alpha', 'Beta', 'Gamma'].includes(r.name));
        const expected = direction === 'best' ? ['Alpha:1', 'Beta:2', 'Gamma:3'] : ['Gamma:3', 'Beta:2', 'Alpha:1'];
        check(JSON.stringify(participants.map(r => r.name + ':' + r.rank)) === JSON.stringify(expected), track.key + '/' + metric.key + '/' + direction);
        check(visible.find(r => r.name === 'Missing').rank === '—', 'missing score stays unranked');
        check(visible[visible.length - 1].name === 'Missing', 'missing score stays last');
        check(visible.find(r => r.name.includes('Baseline') || r.name.includes('Basline')).rank === '—', 'baseline stays unranked');
        directionsChecked++;
      });
    });
    const note = document.getElementById('v2d-cari4d-baseline-note');
    const hasCari4d = track.key === 'track_1' || track.key === 'track_2_tier3';
    check(Boolean(note) === hasCari4d, 'footnote scope/' + track.key);
    const baseline = rows().find(r => r.name.includes('Baseline') || r.name.includes('Basline'));
    check(baseline.name === (track.key === 'track_1' ? 'CARI4D (commercial) Baseline' : 'CHORD Baseline (CARI4D Reconstruction)'), 'baseline display name/' + track.key);
    const marker = baseline.element.querySelector('sup a');
    check(Boolean(marker) === hasCari4d, 'asterisk scope/' + track.key);
    if (hasCari4d) {
      check(note && note.textContent === '* ' + noteText, 'exact footnote text');
      check(marker && marker.textContent === '*' && marker.getAttribute('href') === '#v2d-cari4d-baseline-note', 'asterisk links to footnote');
    }
  });
  document.getElementById('result').textContent = JSON.stringify({ failures, directionsChecked });
}, 100);
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--renderer', type=Path, default=ROOT / 'docs/v2d_challenge/leaderboard.js')
    args = parser.parse_args()
    browser = shutil.which('google-chrome') or shutil.which('chromium')
    if not browser:
        raise SystemExit('Chrome or Chromium is required.')
    source = args.renderer.read_text()
    fallback = re.search(r'var FALLBACK = (.*?);\s*/\* FALLBACK END', source, re.S)
    payload = json.loads(fallback.group(1))
    for track in payload['tracks']:
        def scores(position):
            return {m['key']: 4 - position if m['higher_is_better'] else position for m in track['metrics']}
        track['rows'] = [{'team': name, 'scores': scores(i)} for i, name in enumerate(['Alpha', 'Beta', 'Gamma'], 1)]
        track['rows'] += [{'team': 'Missing', 'scores': {}}, {
            'team': 'CARI4D Basline' if track['key'] == 'track_1' else 'CHORD Baseline (CARI4D Reconstruction)',
            'scores': scores(1.5), 'is_baseline': True,
        }]
    document = '<!doctype html><meta charset="utf-8"><div id="v2d-leaderboard"></div><pre id="result">pending</pre>'
    document += '<script>const payload=' + json.dumps(payload) + ';const noteText=' + json.dumps(NOTE) + ';window.fetch=()=>Promise.resolve({ok:true,json:()=>Promise.resolve(payload)});</script>'
    document += '<script>' + source + '</script><script>' + CHECKS + '</script>'
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / 'check.html'
        path.write_text(document)
        result = subprocess.run([browser, '--headless', '--no-sandbox', '--disable-gpu',
                                 '--user-data-dir=' + str(Path(temp) / 'profile'),
                                 '--virtual-time-budget=2000', '--dump-dom', path.as_uri()],
                                text=True, capture_output=True, timeout=30, check=True)
    match = re.search(r'<pre id="result">(.*?)</pre>', result.stdout, re.S)
    report = json.loads(html.unescape(match.group(1)))
    expected = 2 * sum(len(t['metrics']) for t in payload['tracks'])
    assert report['directionsChecked'] == expected, report
    assert not report['failures'], report
    print(f"Passed {expected} metric/direction checks; baseline and footnote checks passed.")


if __name__ == '__main__':
    main()
