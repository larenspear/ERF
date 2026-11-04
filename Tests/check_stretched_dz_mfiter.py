#!/usr/bin/env python3
"""Guard MFIter enclosure of selected spacing readers and strain launches.

Four finite source contracts: TileNoZ returns only an IntVect; tau and the fast
substep launch their explicit spacing reads inside automatic, default-sync
MFIters; tau and the dycore strain loop call ComputeStrain_S inside those loops.
Explicit iterator aliases, early Finalize, sync/stream overrides and unsupported
preprocessor forms in the checked regions fail closed. ComputeStrain_S must
also contain no explicit stream/sync overrides or MFIter of its own.

This is an enclosure contract, not a GPU execution or whole-program lifetime
proof. With the separately audited AMReX MFIter behavior, it supports completion
of these readers at loop exit, conditional on GPU launch mode, no enclosing
NoSyncRegion/graph capture, stable stream configuration, ordinary automatic
destruction and no concurrent host launches. Assume the previous spacing
provenance/preservation checks, intended C++/AMReX/type/callee semantics, valid
storage/indices and disjoint outputs. Other calls must preserve the iterator,
stream/sync state and spacing through implicit aliases/globals, and must not
defer host launches beyond the loop. External headers must not rewrite macros
or execute runtime statements. No undefined behavior or nonlocal transfer.

Only tau, fast-substep and the direct dycore strain loop are covered. Other
readers, regrid/restart entry, implicit/diffusion/advection callers, projection,
EB and setup-array lifetimes remain separate obligations. The AMReX sources
and stream-coverage argument are an external contract, not parsed by this test.
Only the Python standard library is required.
"""
import re
import sys
from pathlib import Path

from advection_source import Source
from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import require
from check_stretched_dz_bindings import (DYCORE, brace_block, direct, no_literals,
                                       reject_uses, scopes, signature)
from check_stretched_dz_consumers import (FAST, READS, STRAIN, TAU, host_scope,
    kernel_regions, pattern, preprocess, read_context, standalone)
from check_stretched_dz_wrf import take
from momentum_geometry import unique

TILE = 'Source/Utils/ERF_TileNoZ.H'
FILES = (TILE, TAU, FAST, STRAIN, DYCORE)
FUNCTIONS = {TAU: 'erf_make_tau_terms', FAST: 'erf_substep_NS', DYCORE: 'ERF::advance_dycore'}
LOOPS = {
    TAU: (('S_data[IntVars::cons]', 'TileNoZ()', 1),),
    FAST: (('S_stage_data[IntVars::cons]', 'TilingIfNotGPU()', 3),
           ('S_stage_data[IntVars::cons]', 'TileNoZ()', 1)),
    DYCORE: (('state_old[IntVars::cons]', 'TileNoZ()', 2),
             ('state_new[IntVars::cons]', 'TileNoZ()', 1)),
}
# Explicit control changes are forbidden even outside the selected loops.
# Unknown external callees preserving this state remain a stated precondition.
CONTROLS = (
    'MFItInfo', 'NoSyncRegion', 'SyncAtExitOnly', 'StreamIter', 'SingleStreamRegion',
    'setNoSyncRegion', 'setSingleStreamRegion', 'setLaunchRegion', 'LaunchSafeGuard',
    'setGraphRegion', 'GraphSafeGuard', 'startGraphRecording', 'stopGraphRecording',
    'executeGraph', 'in_nosync_region', 'in_single_stream_region', 'in_launch_region',
    'in_graph_region', 'DisableDeviceSync', 'DisableDeviceSyncPre', 'DisableDeviceSyncPost',
    'SetDeviceSync', 'SetDeviceSyncPre', 'SetDeviceSyncPost', 'SetNumStreams',
    'UseDefaultStream', 'setStream', 'resetStream', 'setStreamIndex', 'resetStreamIndex',
    'setGpuStream', 'resetGpuStream', 'setExternalStream', 'resetExternalStream',
    'setExternalGpuStream', 'resetExternalGpuStream', 'ExternalGpuStreamRegion',
    'Finalize', 'allowMultipleMFIters', 'gpu_stream_index', 'max_gpu_streams',
    'cudaLaunchKernel', 'hipLaunchKernel', 'cudaStreamBeginCapture', 'cudaGraphLaunch')


def header(container, tiling):
    return 'for (MFIter mfi('+container+','+tiling+'); mfi.isValid(); ++mfi) {'


def load(root):
    texts = {}
    for path in FILES:
        raw = (root/path).read_text()
        require(not re.search(r'\\\r?\n|\b(?:u8|u|U|L)?R"|<%|%>|%:|\?\?[/=]', raw),
                path+': unsupported lexical form')
        text = strip_comments(raw)
        for m in re.finditer(r'(?m)^[ \t]*#\s*(define|undef)\b([^\n]*)', no_literals(text)):
            require(path == TILE and m[1] == 'define' and m[2].strip() == 'ERF_TILE_NO_Z_H_',
                    path+': unsupported macro')
        reject_uses(text, CONTROLS, [], path+' stream/sync controls')
        texts[path] = text
    return texts


def tile_contract(text):
    source = Source(text)
    take(source, '''#ifndef ERF_TILE_NO_Z_H_
        #define ERF_TILE_NO_Z_H_
        #include <AMReX.H>
        AMREX_FORCE_INLINE amrex::IntVect TileNoZ () {
            if (amrex::TilingIfNotGPU()) {
                return amrex::IntVect{amrex::FabArrayBase::mfiter_tile_size[0],
                    amrex::FabArrayBase::mfiter_tile_size[1], 1024000};
            } else { return amrex::IntVect::TheZeroVector(); }
        }
        #endif''')
    source.done()
    print('  ok   TileNoZ returns an IntVect without changing synchronization')


def caller_body(text, path):
    _, body, definition = signature(text, FUNCTIONS[path])
    direct(text, definition.start(), 'reader definition')
    if path == DYCORE:
        # The direct strain loop ends before statement fragments/callbacks.
        # Those later launches are separate completion obligations.
        marker = unique(r'#include\s+"ERF_TI_utils.H"', body, 'first dycore statement include')
        direct(body, marker.start(), 'dycore statement include')
        reject_uses(body[marker.start():], ['ComputeStrain_S'], [], 'dycore suffix')
        body = body[:marker.start()]
    require(not re.search(r'\b(?:goto|try|catch|co_await|co_yield|co_return)\b', no_literals(body)),
            'unsupported reader control transfer')
    return preprocess(body)


def loops(body, path):
    found = []
    for container, tiling, count in LOOPS[path]:
        matches = list(re.finditer(pattern(header(container, tiling)), body))
        require(len(matches) == count, 'MFIter constructor/iteration inventory changed')
        for m in matches:
            standalone(body, m.start())
            host_scope(body, m.start())
            end = m.end()+len(brace_block(body, m.end()-1))
            found.append((m, end))
    reject_uses(body, ['MFIter'], [m.span() for m, _ in found], 'iterator declarations')
    # A bare mfi can only feed enumerated non-retaining AMReX/ERF readers.
    # Mask only that identifier, so mutations in other arguments remain visible.
    allowed = [m.span() for m, _ in found]
    for m in re.finditer(r'\bmfi\s*\.\s*(?:validbox|tilebox|growntilebox|nodaltilebox|grownnodaltilebox)\s*\(', body):
        allowed.append((m.start(), m.start()+3))
    for m in re.finditer(r'\b(?:array|const_array)\s*\(\s*(mfi)\s*\)', body):
        allowed.append(m.span(1))
    for m in re.finditer(r'\b(?:ComputeStrain_EB|CrseAdd|FineAdd)\s*\(\s*(mfi)\s*,', body):
        allowed.append(m.span(1))
    for m in re.finditer(r'\bgetMultiEBCellFlagFab\s*\(\s*\)\s*\[\s*(mfi)\s*\]', body):
        allowed.append(m.span(1))
    reject_uses(body, ['mfi'], allowed, 'iterator uses')
    return found


def enclosing(body, launch, found):
    host_scope(body, launch)
    selected = [(m, end) for m, end in found if m.end() <= launch < end]
    require(len(selected) == 1, 'spacing launch must be inside one supported automatic MFIter')
    m, _ = selected[0]
    require(('{', m.end()-1) in scopes(body, launch), 'MFIter scope does not enclose launch')


def check(texts):
    tile_contract(texts[TILE])
    _, strain, _ = signature(texts[STRAIN], 'ComputeStrain_S')
    reject_uses(strain, ['MFIter', 'mfi'], [], 'strain launcher iterator inventory')
    require('#' not in no_literals(strain), 'unsupported strain directive')
    for path in FUNCTIONS:
        body = caller_body(texts[path], path)
        found = loops(body, path)
        reads = 0
        if path in (TAU, FAST):
            alias = unique(pattern('auto dz_ptr = stretched_dz_d.data();'), body, 'spacing pointer')
            regions = kernel_regions(body)
            spans = [alias.span()]
            for statement, count in READS[path]:
                matches = list(re.finditer(pattern(statement), body))
                require(len(matches) == count, 'spacing read inventory changed')
                for m in matches:
                    read_context(body, m.start(), alias, regions)
                    region = next(r for r in regions if r[0] < m.start() < r[1])
                    enclosing(body, region[3], found)
                    spans.append(m.span())
                    reads += 1
            reject_uses(body, ['dz_ptr'], spans, 'spacing pointer inventory')
        if path in (TAU, DYCORE):
            call = unique(pattern('ComputeStrain_S('), body, 'strain call')
            standalone(body, call.start())
            enclosing(body, call.start(), found)
            reject_uses(body, ['ComputeStrain_S'], [call.span()], 'strain call inventory')
        print(f'  ok   {FUNCTIONS[path]}: {reads} explicit read statements and selected strain calls enclosed')


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent
    try:
        check(load(root))
    except (OSError, ValueError) as error:
        print('FAIL source contract: '+str(error))
        return 1
    print('4 MFIter enclosure source contracts passed; completion remains conditional on AMReX and caller state.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
