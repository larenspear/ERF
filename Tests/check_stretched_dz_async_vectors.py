#!/usr/bin/env python3
"""Guard the no-substep callback's temporary Array4 descriptor tables.

Three finite source contracts fix four IntVars slots, two complete table setup
sequences, and table uses inside fifteen value-captured kernels. Six automatic
AsyncVectors own four descriptors each. Blocking copies precede pointer binding
and all eight launches; owners stay in their iteration scope and cannot escape,
resize or be rebound through an explicit use in this callback.

The lifetime argument is conditional on inspected AMReX commit
1de18774af5d8a368fa2a398f048e205a2c9f84f: AsyncVector deallocation reaches SArena,
which queues device-accessible memory for freeing after current-stream sync.
Allocation, copies, launches and destruction must share the iteration's stream.
Assume initialized/live AMReX, a device-accessible underlying arena, ordinary
C++/AMReX resolution, normal execution, no graph capture or concurrent host
launches, and no hidden stream/owner changes by other code or callees. CPU
ParallelFor is synchronous. WFromOmega must remain a nonretaining device callee.
External headers and unrelated index macros must preserve these assumptions.

Prior callback/iterator contracts remain prerequisites. Input vectors must have
four valid MultiFabs with layouts compatible with mfi. Backing MultiFab storage,
cell/component indices, arithmetic, projection/BC/EB internals, other temporaries,
initial GPU state and complete regrid/restart/destruction lifetimes are not proved.
This is a finite source guard plus an inspected dependency argument, not a GPU
execution or whole-program theorem. Standard-library Python only.
"""
import re
import sys
from pathlib import Path

from check_mapfactor_storage import strip_comments
from check_momentum_dispatch import require
from check_stretched_dz_bindings import (brace_block, direct, no_literals,
                                         reject_uses, scopes)
from check_stretched_dz_consumers import (declaration_scope, no_outer_closures,
                                          pattern, standalone)
from check_stretched_dz_no_substep_completion import (CALLBACK, CONTROLS, FILES as CALLBACK_FILES,
    LAUNCHES, callback_body, launches, load as load_callback, loops)
from momentum_geometry import call_arguments, unique

INDEX = 'Source/ERF_IndexDefines.H'
FILES = (*CALLBACK_FILES, INDEX)
ENUM = 'namespace IntVars { enum { cons = 0, xmom, ymom, zmom, NumTypes }; }'
COUNT = 'int n_data = IntVars::NumTypes;'
NAMES = ('sold', 'ssum', 'fslow')
OWNERS = tuple(name+suffix for suffix in ('_h', '_d') for name in NAMES)
SETUP = '''Vector<Array4<Real> > ssum_h(n_data);
    Vector<Array4<Real> > sold_h(n_data);
    Vector<Array4<Real> > fslow_h(n_data);
    for (int i = 0; i < n_data; ++i) {
        ssum_h[i] = S_sum[i].array(mfi);
        sold_h[i] = S_old[i].array(mfi);
        fslow_h[i] = F_slow[i].array(mfi);
    }
    Gpu::AsyncVector<Array4<Real> > sold_d(n_data);
    Gpu::AsyncVector<Array4<Real> > ssum_d(n_data);
    Gpu::AsyncVector<Array4<Real> > fslow_d(n_data);
    Gpu::copy(Gpu::hostToDevice, sold_h.begin(), sold_h.end(), sold_d.begin());
    Gpu::copy(Gpu::hostToDevice, ssum_h.begin(), ssum_h.end(), ssum_d.begin());
    Gpu::copy(Gpu::hostToDevice, fslow_h.begin(), fslow_h.end(), fslow_d.begin());
    Array4<Real>* sold = sold_d.dataPtr();
    Array4<Real>* ssum = ssum_d.dataPtr();
    Array4<Real>* fslow = fslow_d.dataPtr();'''
FORWARD = '''WFromOmega(i,j,k,rho_on_face*z_t_arr(i,j,k),
    ssum[IntVars::xmom], ssum[IntVars::ymom], mf_u, mf_v, z_nd_new, dxInv)'''
CALLEE = r'(?<![\w:.>])(?:amrex::)?ParallelFor'


def load(root):
    texts = load_callback(root)
    raw = (root/INDEX).read_text()
    require(not re.search(r'\\\r?\n|\b(?:u8|u|U|L)?R"|<%|%>|%:|\?\?[/=]', raw),
            'index header: unsupported lexical form')
    texts[INDEX] = strip_comments(raw)
    return texts


def table_extent(text, body, found):
    masked = no_literals(text)
    enum = unique(pattern(ENUM), masked, 'four-slot IntVars enum')
    declaration_scope(text, enum.start(), 'ERF_INDEX_DEFINES_H_')
    reject_uses(text, ['IntVars'], [enum.span()], 'IntVars definition inventory')
    protected = (*set(re.findall(r'\b[A-Za-z_]\w*\b', ENUM+COUNT+SETUP+FORWARD)),
                 'ParallelFor', 'AMREX_GPU_DEVICE', 'AMREX_USE_GPU',
                 'AMREX_USE_CUDA', 'AMREX_USE_HIP', 'AMREX_USE_SYCL', '_OPENMP',
                 'MFIter', 'TilingIfNotGPU', *CONTROLS)
    for macro in re.finditer(r'(?m)^[ \t]*#\s*(define|undef)\b([^\n]*)', masked):
        if macro[1] == 'define' and macro[2].strip() == 'ERF_INDEX_DEFINES_H_':
            continue
        reject_uses(macro[2], [*protected, 'ERF_INDEX_DEFINES_H_'], [], 'index macro')
    count = unique(pattern(COUNT), no_literals(body), 'table size')
    direct(body, count.start(), 'table size')
    standalone(body, count.start())
    require(count.end() < found[0][0].start(), 'table size must precede both loops')
    print('  ok   four enum slots and callback-local table size')
    return count.span()


def setups(body, found):
    spans = []
    for loop, end in found:
        offset = loop.end()
        iteration = body[offset:end]
        setup = unique(pattern(SETUP), no_literals(iteration), 'complete descriptor setup')
        direct(iteration, setup.start(), 'descriptor setup')
        standalone(iteration, setup.start())
        spans.append((offset+setup.start(), offset+setup.end()))
    reject_uses(body, [*OWNERS, 'AsyncVector'], spans, 'descriptor owner uses')
    print('  ok   six iteration-local AsyncVectors populated by six blocking copies')
    return spans


def kernel_spans(body):
    """Use only after launches has checked every complete launch argument."""
    masked = no_literals(body)
    matches = re.finditer(CALLEE+r'\s*\(', masked)
    arguments = call_arguments(masked, CALLEE)
    for launch, args, (prefix, _) in zip(matches, arguments, LAUNCHES):
        for lo, hi in args[len(prefix):]:
            opening = masked.index('{', lo, hi)
            yield launch.start(), opening+1, opening+1+len(brace_block(body, opening))


def uses(body, setup_spans, count_span):
    allowed = list(setup_spans)
    forwarding = unique(pattern(FORWARD), no_literals(body), 'nonretaining descriptor forwarding')
    reject_uses(body, ['WFromOmega'], [forwarding.span()], 'descriptor callee inventory')
    read = r'\b(?:sold|ssum|fslow)\b'
    descriptor = (r'(?<![\w:.>])(?:sold|ssum|fslow)\s*\[\s*IntVars\s*::\s*'
                  r'(?:cons|xmom|ymom|zmom)\s*\]')
    passed = False
    for index, (launch, lo, hi) in enumerate(kernel_spans(body)):
        # The main loop owns eleven kernels, the redistribution loop four.
        require(setup_spans[0 if index < 11 else 1][1] < launch,
                'kernel launch precedes descriptor setup')
        kernel = no_literals(body[lo:hi])
        for use in re.finditer(read, kernel):
            no_outer_closures(kernel, scopes(kernel, use.start()))
            match = re.compile(descriptor).match(kernel, use.start())
            require(match, 'table use must select one of the four IntVars slots')
            start, end = lo+match.start(), lo+match.end()
            if not re.match(r'\s*\(', kernel[match.end():]):
                # Only the two bare descriptor arguments in the exact moving-
                # terrain call may bypass Array4 scalar element access.
                require(index == 3 and forwarding.start() <= start < end <= forwarding.end(),
                        'descriptor escapes element access or approved WFromOmega call')
                passed = True
            allowed.append((start, end))
    require(passed, 'missing moving-terrain descriptor forwarding')
    reject_uses(body, NAMES, allowed, 'descriptor pointer uses')
    reject_uses(body, ['n_data'], [count_span, *setup_spans], 'table size uses')
    print('  ok   table pointers used only in fifteen direct kernels, with bounded descriptor slots')


def main():
    root = Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent.parent
    try:
        texts = load(root)
        body = callback_body(texts[CALLBACK])
        found, _ = loops(body)
        launches(body, found)
        count = table_extent(texts[INDEX], body, found)
        setup_spans = setups(body, found)
        uses(body, setup_spans, count)
    except (OSError, ValueError) as error:
        print('FAIL source contract: '+str(error))
        return 1
    print('3 AsyncVector source contracts passed; deferred freeing is conditional on inspected AMReX semantics.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
