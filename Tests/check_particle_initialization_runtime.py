#!/usr/bin/env python3
"""Serial CPU-double particle initialization ownership, count, mass and tag checks.

The 8 m cube has 8^3 base cells. Number density 128 gives integer physical
counts at every tested refinement, so these controls isolate ownership from
multiplicity rounding. No advection, coalescence or phase change can alter mass.
"""
import argparse
from collections import Counter
import hashlib
import itertools
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from particle_checkpoint_compare import read

PREFIX = 'super_droplets_moisture.'
FINE = ['amr.max_level=1', 'amr.ref_ratio_vect=2 2 2', 'amr.max_grid_size_z=16',
        'amr.n_error_buf=0', 'erf.regrid_int=-1', 'erf.coupling_type=TwoWay',
        'erf.cf_width=0', 'erf.refinement_indicators=patch', 'erf.patch.max_level=1',
        'erf.patch.in_box_lo_indices_crse=0 0 0', 'erf.patch.in_box_hi_indices_crse=3 7 7']
MASS = {'aerosol_mass_NaCl': 1e-19, 'species_mass_H2O': 4.1887902e-9,
        'species_mass_ice': 1e-9}
MASS['particle_mass'] = math.fsum(MASS.values())
INJECT = [PREFIX+'num_initializations=0', PREFIX+'num_injections=1',
          PREFIX+'injection.rate=128000', PREFIX+'injection.particles_per_cell=1',
          PREFIX+'injection.multiplicity_type=constant',
          PREFIX+'injection.species_mean_mass_H2O=4.1887902e-9',
          PREFIX+'injection.species_mean_mass_ice=1e-9',
          PREFIX+'injection.aerosol_mean_mass_NaCl=1e-19']
SPLIT = [PREFIX+'split_merge_amr=true']
# Each tuple is overrides, expected physical count, expected SD count (when
# prescribed analytically), whether native creation tags must equal level+1.
CASES = {
    'base': ([], 65536, 512, True),
    'half': (FINE, 65536, 2304, True),
    'full': (FINE+['erf.patch.in_box_hi_indices_crse=7 7 7'], 65536, 4096, True),
    'half-split': (FINE+SPLIT, 65536, 2304, True),
    'anisotropic': (FINE+['amr.ref_ratio_vect=2 2 1'], 65536, 1280, True),
    'two-levels': (FINE+['amr.max_level=2', 'amr.max_grid_size_z=32', 'amr.ref_ratio_vect=2 2 2 2 2 2',
        'erf.patch.in_box_hi_indices_crse=7 7 7', 'erf.refinement_indicators=patch deep', 'erf.deep.max_level=2',
        'erf.deep.in_box_lo_indices_crse=0 0 0', 'erf.deep.in_box_hi_indices_crse=7 15 15'], 65536, 18432, True),
    'sampled': (FINE+[PREFIX+'multiplicity_type=sampled', 'amrex.fpe_trap_zero=1'], 65536, 2304, True),
    'empty': (FINE+[PREFIX+'initial_number_density=0'], 0, 0, True),
    'multiple': (FINE+[PREFIX+'num_initializations=2'], 131072, 4608, True),
    'rounded': (FINE+[PREFIX+'initial_number_density=100'], 52224, 2304, True),
    'partial-box': (FINE+[PREFIX+'particle_box_hi=2 8 8'], 16384, 1024, True),
    'inject-base': (INJECT, 65536, 512, True),
    'inject-half': (FINE+INJECT, 65536, 512, False),
    'newlevel': (FINE+['erf.regrid_int=1', 'erf.patch.start_time=0.0015', 'max_step=3'], 65536, None, False),
    'derefinement': (FINE+SPLIT+['erf.regrid_int=1', 'erf.patch.end_time=0.0015', 'max_step=3'], 65536, 512, True),
    'remake': (FINE+['erf.regrid_int=1', 'erf.patch.end_time=0.0015', 'max_step=3',
        'erf.refinement_indicators=patch moved', 'erf.moved.max_level=1', 'erf.moved.start_time=0.0015',
        'erf.moved.in_box_lo_indices_crse=4 0 0', 'erf.moved.in_box_hi_indices_crse=7 7 7'], 65536, None, False),
}
# Independently count finest cell centers inside a radius-1.5 sphere, wholly
# within x<4. No coarse-cell selection enters this expected value.
BUBBLE_CELLS = sum(sum(((i+0.5)/2-c)**2 for i,c in zip(ijk,(2,4,4))) <= 1.5**2
                   for ijk in itertools.product(range(16), repeat=3))
CASES['bubble'] = (FINE+[PREFIX+'initial_distribution_type=bubble', PREFIX+'particle_bubble_center=2 4 4',
                       PREFIX+'particle_bubble_radius=1.5 1.5 1.5'], 16*BUBBLE_CELLS, BUBBLE_CELLS, True)


def inspect(checkpoint):
    data=read(checkpoint/'super_droplets_moisture')
    names=data['real_names']; mi=names.index('multiplicity')
    records=list(data['particles'].values())
    return {'superdroplets':len(records), 'physical':math.fsum(v[3][mi] for v in records),
            'by_level':dict(Counter(str(v[0]) for v in records)),
            'tags':dict(Counter(str(v[2][0]) for v in records)),
            'native_tags':all(v[2][0]==v[0]+1 for v in records),
            'left_physical':math.fsum(v[3][mi] for v in records if v[3][0]<4),
            'mass':{name:math.fsum(v[3][mi]*v[3][names.index(name)] for v in records) for name in MASS}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--erf', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--inputs',type=Path,default=Path(__file__).parent/'test_files/SDM_Restart/SDM_Restart.i')
    parser.add_argument('--case',choices=CASES,action='append')
    args=parser.parse_args()
    assert sys.byteorder=='little'
    args.work_dir.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='particle-init-',dir=args.work_dir.resolve()))
    exe=root/'erf_exec';shutil.copy2(args.erf,exe)
    results={'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'cases':[], 'passed':False}
    def save(): (root/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    for name in args.case or CASES:
        options,physical,sd,native=CASES[name]
        case=root/name;case.mkdir();shutil.copy2(args.inputs,case/'inputs')
        command=[str(exe),'inputs','max_step=1',PREFIX+'initial_number_density=128',
                 PREFIX+'advect_with_gravity=false',*options]
        (case/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        with (case/'run.log').open('w') as log:
            proc=subprocess.run(command,cwd=case,stdout=log,stderr=subprocess.STDOUT,timeout=60)
        entry={'name':name,'returncode':proc.returncode,'checks':[]};results['cases'].append(entry)
        def check(label,ok,actual,expected):entry['checks'].append({'name':label,'passed':bool(ok),'actual':actual,'expected':expected})
        check('execution',proc.returncode==0,proc.returncode,0)
        if proc.returncode==0:
            checkpoints=sorted(case.glob('chk[0-9]*'))
            check('checkpoint-written',bool(checkpoints),len(checkpoints),'>0')
            for checkpoint in checkpoints:
                value=inspect(checkpoint)
                prefix=checkpoint.name+'-'
                expected_physical=0 if name.startswith('inject-') and checkpoint.name=='chk00000' else physical
                check(prefix+'physical-number',value['physical']==expected_physical,value['physical'],expected_physical)
                if name not in ('bubble','partial-box'):
                    left_expected = (26624 if name=='rounded' else expected_physical/2)
                    check(prefix+'left-region-number',value['left_physical']==left_expected,value['left_physical'],left_expected)
                for mass,m in MASS.items():
                    expected=expected_physical*m
                    # Fixed particle masses; fsum reduces summation error. Regrid
                    # merges can round weighted means. 64 epsilon covers those
                    # bounded arithmetic operations at each mass's own scale.
                    tol=64*sys.float_info.epsilon*abs(expected)
                    check(prefix+mass,abs(value['mass'][mass]-expected)<=tol,value['mass'][mass],expected)
                if checkpoint==checkpoints[-1]:
                    if sd is not None:check(prefix+'superdroplets',value['superdroplets']==sd,value['superdroplets'],sd)
                    if native:check(prefix+'native-tags',value['native_tags'],value['tags'],'tag=level+1')
                entry.setdefault('observations',[]).append({'checkpoint':checkpoint.name,**value})
            log=(case/'run.log').read_text()
            expected_calls=1 if name=='newlevel' or not any(x=='amr.max_level=1' for x in options) else 2
            if name=='two-levels':expected_calls=3
            check('startup-only-seeding',log.count('InitializeParticles at level')==expected_calls,log.count('InitializeParticles at level'),expected_calls)
            if name in ('newlevel','remake','derefinement'):
                expected_levels=1 if name=='derefinement' else 2
                actual=len(list(checkpoints[-1].glob('Level_*')))
                check('final-AMR-levels',actual==expected_levels,actual,expected_levels)
        entry['passed']=all(c['passed'] for c in entry['checks']);save()
        print(name,'PASS' if entry['passed'] else 'FAIL',flush=True)
    results['passed']=all(case['passed'] for case in results['cases']);save()
    print(root)
    return 0 if results['passed'] else 1
if __name__=='__main__':sys.exit(main())
