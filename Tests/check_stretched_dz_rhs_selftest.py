#!/usr/bin/env python3
"""Mutation fixtures for slow-RHS spacing forwarding and PBL value capture."""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from advection_source import mutation_outcome
from check_mapfactor_storage import strip_comments
from check_stretched_dz_bindings import brace_block, no_literals, signature
from check_stretched_dz_rhs import CALLEES, FILES, FORWARD, GUARDS, MOM, PBL, POST, PRE, TKE
from momentum_geometry import call_arguments

ROOT = Path(__file__).resolve().parent.parent
VERIFY = ROOT/'Tests/check_stretched_dz_rhs.py'
SUCCESS = '5 spacing RHS/PBL source contracts passed;'


def cases(originals):
    yield 'baseline', {}, False, SUCCESS

    def edit(name, path, before, after, diagnostic='FAIL source contract:', fault=True):
        assert originals[path].count(before) == 1, (name, originals[path].count(before))
        return name, {path: originals[path].replace(before, after)}, fault, diagnostic if fault else SUCCESS

    for path, name in ((PRE, 'erf_slow_rhs_pre'), (POST, 'erf_slow_rhs_post'), (MOM, 'AdvectionSrcForMom')):
        anchor = 'BL_PROFILE_VAR("AdvectionSrcForMom", AdvectionSrcForMom);' if path == MOM else f'BL_PROFILE_REGION("{name}()");'
        for label, statement in (
            ('clear', 'stretched_dz_d.clear();'),
            ('resize', 'stretched_dz_d.resize(0);'),
            ('assignment', 'stretched_dz_d = other;'),
            ('swap', 'stretched_dz_d.swap(other);'),
            ('reference alias', 'auto& alias = stretched_dz_d;'),
            ('pointer alias', 'auto alias = stretched_dz_d.data();'),
            ('unknown call', 'retain(stretched_dz_d);'),
            ('element write', 'stretched_dz_d[0] = 1;'),
            ('element reference', 'auto& alias = stretched_dz_d[0];'),
            ('explicit capture', 'retain([&stretched_dz_d] {});'),
            ('quoted brace hiding', 'BL_PROFILE("}"); stretched_dz_d.clear();'),
            ('quoted comment hiding', 'BL_PROFILE("/*"); stretched_dz_d.clear();'),
        ):
            yield edit(name+' '+label, path, anchor, anchor+' '+statement, 'unmodeled use of stretched_dz_d')
        yield edit(name+' body include', path, anchor, anchor+'\n#include "unmodeled.H"', 'unsupported consumer preprocessor')
        yield edit(name+' body macro', path, anchor, anchor+'\n#define stretched_dz_d other\n', 'unsupported macro')
        yield edit(name+' lexical splice', path, anchor, anchor+'\n// hidden \\\nstretched_dz_d.clear();', 'unsupported C++ lexical form')
        # Wrap the complete body in a valid lambda; default captures contain no
        # explicit extra use of spacing, so scope checks must catch the escape.
        text = strip_comments(originals[path])
        _, body, definition = signature(no_literals(text), name)
        begin, end = definition.end(), definition.end()+len(body)
        for label, prefix, suffix in (
                ('deferred body', 'retain([&] {', '});'),
                ('named closure', 'auto deferred = [&] {', '};'),
                ('generic closure', 'auto deferred = [&]<typename T>() {', '};')):
            changed = text[:begin]+prefix+text[begin:end]+suffix+text[end:]
            yield name+' '+label, {path: changed}, True, 'FAIL source contract:'
        for callee in FORWARD[path]:
            args = list(call_arguments(no_literals(text), r'(?<![\w:.>])'+callee))
            assert len(args) == 1
            arg = [span for span in args[0] if text[span[0]:span[1]].strip() == 'stretched_dz_d']
            assert len(arg) == 1
            lo, hi = arg[0]
            for label, replacement in (('wrong vector', 'other'), ('nested mutation', 'mutate(stretched_dz_d)'),
                                        ('raw pointer', 'stretched_dz_d.data()')):
                yield f'{name} to {callee} '+label, {path: text[:lo]+replacement+text[hi:]}, True, 'argument binding stretched_dz_d'
            yield edit(name+' shadows '+callee, path, anchor, anchor+' auto '+callee+' = other;', 'callee uses')
        yield edit(name+' quoted literal control', path, anchor, anchor+' BL_PROFILE("} // /* stretched_dz_d.clear();");', fault=False)
        yield edit(name+' unrelated local control', path, anchor, anchor+' const int unrelated = 1;', fault=False)

    # Every external callee declaration participates in the position/type check.
    for name, (path, header, _) in CALLEES.items():
        for target, ending in ((path, '{'), (header, ';')):
            text = strip_comments(originals[target])
            _, _, match = signature(no_literals(text), name, ending)
            fragment = text[match.start():match.end()]
            changed, count = re.subn(r'&\s*stretched_dz_d\b', ' stretched_dz_d', fragment)
            assert count == 1
            yield name+' copied parameter '+Path(target).name, {target: text[:match.start()]+changed+text[match.end():]}, True, 'type mismatch'
    for name, path in (('erf_slow_rhs_pre', PRE), ('erf_slow_rhs_post', POST)):
        text = strip_comments(originals[path])
        _, _, definition = signature(no_literals(text), name)
        at = text.index('#ifdef ERF_USE_EAMXX_SHOC\n', definition.end())+len('#ifdef ERF_USE_EAMXX_SHOC\n')
        yield name+' conditional hidden write', {path: text[:at]+'stretched_dz_d.clear();\n'+text[at:]}, True, 'SHOC region'

    for label, before, after in (
        ('mutable pointer', 'const amrex::Real* dz;', 'amrex::Real* dz;'),
        ('reference pointer field', 'const amrex::Real* dz;', 'const amrex::Real*& dz;'),
        ('extra pointer field', 'const amrex::Real* dz;', 'const amrex::Real* dz; const amrex::Real* other;'),
        ('custom destructor', 'const amrex::Real* dz;', 'const amrex::Real* dz; ~PBLDerivativeDzInv_S() { retain(dz); }'),
        ('custom copy', 'const amrex::Real* dz;', 'const amrex::Real* dz; PBLDerivativeDzInv_S(const PBLDerivativeDzInv_S& x) { retain(x.dz); }'),
        ('operator pointer escape', 'const amrex::Real dz_lo = dz[(k > klo) ? k-1 : k];', 'retain(dz); const amrex::Real dz_lo = dz[(k > klo) ? k-1 : k];'),
        ('operator write', 'const amrex::Real dz_lo = dz[(k > klo) ? k-1 : k];', 'const_cast<amrex::Real*>(dz)[k] = 1; const amrex::Real dz_lo = dz[(k > klo) ? k-1 : k];'),
        ('operator element reference', 'const amrex::Real dz_lo = dz[(k > klo) ? k-1 : k];', 'const amrex::Real& dz_lo = dz[(k > klo) ? k-1 : k];'),
        ('operator member rebind', 'const amrex::Real dz_lo = dz[(k > klo) ? k-1 : k];', 'dz = nullptr; const amrex::Real dz_lo = dz[(k > klo) ? k-1 : k];'),
        ('result pointer', 'amrex::Real interior;', 'const amrex::Real* interior;'),
        ('result reference', 'amrex::Real interior;', 'const amrex::Real& interior;'),
        ('result extra pointer', 'amrex::Real interior;', 'amrex::Real interior; const amrex::Real* escape;'),
        ('member token fusion', 'amrex::Real interior;', 'amrex::Realinterior;'),
    ):
        yield edit('PBL '+label, PBL, before, after, 'unsupported source')
    for label, before, after in (
        ('wrong vector', 'pbl_dz_inv{stretched_dz_d.data(),', 'pbl_dz_inv{other.data(),'),
        ('offset pointer', 'pbl_dz_inv{stretched_dz_d.data(),', 'pbl_dz_inv{stretched_dz_d.data()+1,'),
        ('nonconst functor', 'const PBLDerivativeDzInv_S pbl_dz_inv', 'PBLDerivativeDzInv_S pbl_dz_inv'),
        ('wrong low endpoint', 'domain.smallEnd(2), domain.bigEnd(2)};', 'domain.smallEnd(0), domain.bigEnd(2)};'),
        ('wrong high endpoint', 'domain.smallEnd(2), domain.bigEnd(2)};', 'domain.smallEnd(2), domain.bigEnd(0)};'),
        ('alias escape', 'domain.smallEnd(2), domain.bigEnd(2)};', 'domain.smallEnd(2), domain.bigEnd(2)}; retain(pbl_dz_inv);'),
        ('pointer write', 'domain.smallEnd(2), domain.bigEnd(2)};', 'domain.smallEnd(2), domain.bigEnd(2)}; const_cast<Real*>(pbl_dz_inv.dz)[0] = 1;'),
        ('callee shadow', 'const PBLDerivativeDzInv_S pbl_dz_inv', 'auto AddTurbKESources = other; const PBLDerivativeDzInv_S pbl_dz_inv'),
    ):
        yield edit('post '+label, POST, before, after)
    text = originals[POST]
    needle = 'SurfLayer, pbl_dz_inv);'
    positions = [m.start() for m in re.finditer(re.escape(needle), text)]
    assert len(positions) == 3
    for index, start in enumerate(positions):
        yield f'post wrong TKE argument {index}', {POST: text[:start]+needle.replace('pbl_dz_inv', 'other')+text[start+len(needle):]}, True, 'PBL to TKE: argument binding'
    for label, before, after in (
        ('reference capture', '[=,moist_indices=solverChoice.moisture_indices]', '[&,moist_indices=solverChoice.moisture_indices]'),
        ('explicit reference capture', '[=,moist_indices=solverChoice.moisture_indices]', '[=,&pbl_derivative_dz_inv,moist_indices=solverChoice.moisture_indices]'),
        ('mutable functor parameter', 'const PBLDerivativeDzInvFunctor& pbl_derivative_dz_inv)', 'PBLDerivativeDzInvFunctor& pbl_derivative_dz_inv)'),
        ('functor alias', 'const auto pbl_dz_inv = pbl_derivative_dz_inv(i,j,k);', 'auto& alias = pbl_derivative_dz_inv; const auto pbl_dz_inv = pbl_derivative_dz_inv(i,j,k);'),
        ('functor escape', 'const auto pbl_dz_inv = pbl_derivative_dz_inv(i,j,k);', 'retain(pbl_derivative_dz_inv); const auto pbl_dz_inv = pbl_derivative_dz_inv(i,j,k);'),
        ('wrong k', 'const auto pbl_dz_inv = pbl_derivative_dz_inv(i,j,k);', 'const auto pbl_dz_inv = pbl_derivative_dz_inv(i,j,k+1);'),
        ('reference result', 'const auto pbl_dz_inv = pbl_derivative_dz_inv(i,j,k);', 'const auto& pbl_dz_inv = pbl_derivative_dz_inv(i,j,k);'),
        ('nested evaluation', 'const auto pbl_dz_inv = pbl_derivative_dz_inv(i,j,k);', 'defer([=] { const auto pbl_dz_inv = pbl_derivative_dz_inv(i,j,k); });'),
        ('body include', 'BL_PROFILE("AddTurbKESources()");', 'BL_PROFILE("AddTurbKESources()");\n#include "unmodeled.H"'),
        ('quoted brace hiding', 'BL_PROFILE("AddTurbKESources()");', 'BL_PROFILE("}"); retain(pbl_derivative_dz_inv);'),
    ):
        yield edit('TKE '+label, TKE, before, after)
    for path in GUARDS:
        yield Path(path).name+' macro', {path: originals[path]+'\n#define stretched_dz_d other\n'}, True, 'unsupported macro'
    for path, name in ((PBL, 'struct PBLDerivativeDzInv_S'), (TKE, 'void\nAddTurbKESources')):
        text = originals[path]
        start = text.index(name)
        brace = text.index('{', start)
        end = brace+len(brace_block(text, brace))+2
        yield name.replace('\n', ' ')+' disabled', {path: text[:start]+'#if 0\n'+text[start:end]+'\n#endif\n'+text[end:]}, True, 'unsupported conditional scope'

    yield 'comments and whitespace', {p: strip_comments(t).replace(',', ', /* gap */ ').replace(';', ';\n') for p, t in originals.items()}, False, SUCCESS
    yield edit('PBL comments in operator', PBL, 'const amrex::Real* dz;', 'const /* pointer value */ amrex::Real * dz;', fault=False)
    yield edit('TKE quoted literal', TKE, 'BL_PROFILE("AddTurbKESources()");', 'BL_PROFILE("} // /* pbl_derivative_dz_inv");', fault=False)
    yield edit('TKE unrelated arithmetic', TKE, 'int qty_index = RhoKE_comp;\n        auto pbl_mynn_B1_l', 'const int unused = 1; int qty_index = RhoKE_comp;\n        auto pbl_mynn_B1_l', fault=False)
    # Permuting the pointer and bounds is not allowed: aggregate member order
    # fixes their binding. A scalar-only unrelated declaration is harmless.
    yield edit('unrelated PBL struct', PBL, 'struct PBLDerivativeDzInv_S', 'struct Unrelated { int value; };\nstruct PBLDerivativeDzInv_S', fault=False)


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix='erf-spacing-rhs-mutants-'))
    originals = {path: (ROOT/path).read_text() for path in FILES}
    fixtures = list(cases(originals))
    faults = controls = unexpected = 0
    print('Fixture directory: '+str(work), flush=True)
    for index, (name, edits, fault, required) in enumerate(fixtures):
        root = work/f'{index:03d}'
        for path, text in originals.items():
            target = root/path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(edits.get(path, text))
        result = subprocess.run([sys.executable, str(VERIFY), str(root)], capture_output=True, text=True)
        (root/'stdout.txt').write_text(result.stdout)
        (root/'stderr.txt').write_text(result.stderr)
        caught, expected, _ = mutation_outcome(result, fault)
        expected = expected and required in result.stdout
        faults += fault
        controls += not fault and name != 'baseline'
        unexpected += not expected
        print(f"[{'CAUGHT' if caught and expected else 'passes' if expected else 'UNEXPECTED'}] {name}", flush=True)
        if not expected:
            print(result.stdout[-1600:]+result.stderr[-800:])
            print('required: '+required)
    print(f'\n{faults} faults; {controls} controls; {unexpected} unexpected outcomes')
    return int(bool(unexpected))


if __name__ == '__main__':
    sys.exit(main())
