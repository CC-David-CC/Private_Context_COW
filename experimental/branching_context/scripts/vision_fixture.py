"""Checked campaign inputs and an independent CPU coordinate reference.

These helpers prepare tests. Production input/position code never imports them.
"""
import copy
import hashlib
import json
from pathlib import Path
import shlex
import struct


def fingerprint(path):
    path = Path(path).resolve()
    with path.open('rb') as stream:
        digest = hashlib.file_digest(stream, 'sha256').hexdigest()
    return dict(path=str(path), bytes=path.stat().st_size, sha256=digest)


def fixture_parts(fixture_path, originals):
    fixture_path = Path(fixture_path)
    fixture = json.loads(fixture_path.read_text())
    lines = (fixture_path.parent/'input.manifest').read_text().splitlines()
    header = lines[0].split()
    if len(header) != 4 or header[0] != 'COW_VISION_INPUT_V1':
        raise ValueError('Unknown fixture transport')
    consumed, count, images = map(int, header[1:])
    tokens = list(map(int, lines[1].split()))
    if (consumed != 0 or not 1 <= images <= 8 or len(lines) != images+2
            or len(tokens) != count or tokens != fixture['input_tokens'] or len(originals) != images):
        raise ValueError('Fixture token, image or manifest extent mismatch')
    expected = fixture.get('images', [dict(embedding=fixture.get('embedding'))])
    if len(expected) != images:
        raise ValueError('Fixture image identities do not match the manifest')
    result, previous_end, widths = [], 0, set()
    for line, original, pinned in zip(lines[2:], originals, expected):
        fields = shlex.split(line)
        if len(fields) != 8:
            raise ValueError('Invalid fixture image placement')
        start, rows, nx, ny, width, pad = map(int, fields[:6])
        embedding = fingerprint(fields[7]); source = fingerprint(original)
        if embedding != pinned['embedding'] or embedding['sha256'] != fields[6]:
            raise ValueError('Fixture embedding changed')
        if 'original' in pinned and source != pinned['original']:
            raise ValueError('Fixture original image/order changed')
        with Path(embedding['path']).open('rb') as stream:
            actual = stream.read(20)
        if (min(rows,nx,ny,width) <= 0 or rows > 1024 or rows != nx*ny
                or embedding['bytes'] != 20+rows*width*4
                or actual != struct.pack('<5i',0x31455653,rows,nx,ny,width)
                or start < previous_end or start+rows >= count or tokens[start:start+rows] != [pad]*rows):
            raise ValueError('Invalid fixture image geometry or token placement')
        result.append(dict(start=start, rows=rows, grid_width=nx, grid_height=ny,
                           width=width, pad=pad, embedding=embedding, original=source))
        widths.add(width); previous_end = start+rows
    if len(widths) != 1:
        raise ValueError('Fixture images use different embedding widths')
    return result


def configure_prefill(profile, chunk):
    result = copy.deepcopy(profile)
    execution = result['execution']
    if type(chunk) is not int or not 0 < chunk <= execution['capacity']:
        raise ValueError('Invalid native prefill chunk')
    for key in ('sut_command', 'oracle_command'):
        command = execution[key]
        if command.count('--prefill') != 1 or command.index('--prefill')+1 >= len(command):
            raise ValueError('Native command must declare one actual prefill chunk')
        command[command.index('--prefill')+1] = str(chunk)
    execution['prefill_chunk'] = chunk
    return result


def reference_positions(consumed, images, capacity):
    """Absolute-cell formula, independent of the production rotary walk.

    An image shortens following rotary positions by rows minus its larger grid
    dimension. Grid rows use the image's adjusted start, height and width.
    """
    if not 0 < consumed <= capacity:
        raise ValueError('Invalid independent position extent')
    previous = 0
    for image in images:
        if (image['start'] < previous or image['rows'] != image['grid_width']*image['grid_height']
                or image['start']+image['rows'] >= consumed):
            raise ValueError('Invalid independent image placement')
        previous = image['start']+image['rows']
    result = []
    for cell in range(capacity):
        before = sum(image['rows']-max(image['grid_width'],image['grid_height'])
                     for image in images if image['start']+image['rows'] <= cell)
        active = next((image for image in images if image['start'] <= cell < image['start']+image['rows']), None)
        if active is None:
            position = cell-before; result.append((position,position,position))
        else:
            start = active['start']-before; row = cell-active['start']
            result.append((start,start+row//active['grid_width'],start+row%active['grid_width']))
    return result


def boundary_coverage(images, chunks, audit):
    consumed = audit['consumed']; position = 0
    for start, count in chunks:
        if start != position or count <= 0:
            raise ValueError('Noncontiguous observed native batches')
        position += count
    if position != consumed:
        raise ValueError('Observed batches do not cover the audited prefix')
    def inside(cell):
        return any(image['start'] < cell < image['start']+image['rows'] for image in images)
    physical = set()
    for region in audit['regions']:
        if region['kind'] != 'k' or not region['pool_offset']:
            continue
        offset, size = region['pool_offset'], region['bytes_per_cell']
        if size <= 0 or offset % size:
            raise ValueError('Unusable native physical K-page geometry')
        if inside(offset//size): physical.add(offset//size)
    qsa = audit['qsa_page_cells']
    if qsa <= 0:
        raise ValueError('Missing QSA page geometry')
    return dict(batch_boundaries_inside_images=[start for start,_ in chunks[1:] if inside(start)],
                physical_k_boundaries_inside_images=sorted(physical),
                logical_qsa_boundaries_inside_images=[cell for cell in range(qsa,consumed,qsa) if inside(cell)],
                partial_qsa_tail_cells=consumed%qsa, qsa_page_cells=qsa)
