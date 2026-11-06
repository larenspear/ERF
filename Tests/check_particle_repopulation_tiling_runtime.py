#!/usr/bin/env python3
"""Check physical budgets and per-cell SD representation across CPU particle tiles."""
import argparse
from collections import defaultdict
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

P='super_droplets_moisture.'
FINE=['amr.max_level=1','amr.ref_ratio_vect=2 2 2','amr.max_grid_size_z=16',
      'amr.n_error_buf=0','erf.regrid_int=-1','erf.coupling_type=TwoWay','erf.cf_width=0',
      'erf.refinement_indicators=patch','erf.patch.max_level=1',
      'erf.patch.in_box_lo_indices_crse=0 0 0','erf.patch.in_box_hi_indices_crse=3 7 7']
INJECT=[P+'num_initializations=0',P+'num_injections=1',P+'injection.rate=128000',
        P+'injection.particles_per_cell=1',P+'injection.multiplicity_type=constant',
        P+'injection.species_mean_mass_H2O=4.1887902e-9',
        P+'injection.species_mean_mass_ice=1e-9',P+'injection.aerosol_mean_mass_NaCl=1e-19']
BUBBLE=[P+'initial_distribution_type=bubble',P+'particle_bubble_center=2 4 4',P+'particle_bubble_radius=1.5 1.5 1.5']
BUBBLE_COUNT=sum(sum(((i+0.5)/2-c)**2 for i,c in zip(ijk,(2,4,4))) <= 1.5**2
                 for ijk in itertools.product(range(16),repeat=3))
MASS={'aerosol_mass_NaCl':1e-19,'species_mass_H2O':4.1887902e-9,'species_mass_ice':1e-9}
MASS['particle_mass']=math.fsum(MASS.values())
# Overrides, refinement vector, expected physical count (per step if injection).
CASES={
 'dense':([], (2,2,2),65536),
 'bubble':(BUBBLE,(2,2,2),16*BUBBLE_COUNT),
 'multiple-bubble':(BUBBLE+[P+'num_initializations=2'],(2,2,2),32*BUBBLE_COUNT),
 'injection':(INJECT,(2,2,2),65536),
 'injection-full':(INJECT+['erf.patch.in_box_hi_indices_crse=7 7 7'],(2,2,2),65536),
 'injection-anisotropic':(INJECT+['amr.ref_ratio_vect=2 2 1'],(2,2,1),65536),
 'injection-ratio4':(INJECT+['amr.ref_ratio_vect=4 4 4','amr.max_grid_size_z=32',P+'injection.rate=4096000'],(4,4,4),2097152),
 'injection-limited':(INJECT+[P+'injection.rate=4000'],(2,2,2),2048),
}
TILES={'untiled':['particles.do_tiling=false'],
       'regular':['particles.do_tiling=true','particles.tile_size=4 4 4'],
       'uneven':['particles.do_tiling=true','particles.tile_size=3 5 2'],
       'unit':['particles.do_tiling=true','particles.tile_size=1 1 1']}


def observe(checkpoint,ratio):
    data=read(checkpoint/'super_droplets_moisture');names=data['real_names'];mi=names.index('multiplicity');groups=defaultdict(list)
    for lev,grid,ints,values in data['particles'].values():
        assert ints[0]>0 and values[mi]>0
        key=','.join(map(str,(lev,*(math.floor(values[d]*ratio[d]**lev) for d in range(3)))))
        groups[key].append(values)
    cells={}
    for key,records in groups.items():
        cells[key]={'count':len(records),'number':math.fsum(v[mi] for v in records),
                    'multiplicities':sorted(v[mi] for v in records),
                    'mass':{name:math.fsum(v[mi]*v[names.index(name)] for v in records) for name in MASS}}
    values=list(cells.values())
    return {'count':sum(c['count'] for c in values),'number':math.fsum(c['number'] for c in values),
            'mass':{name:math.fsum(c['mass'][name] for c in values) for name in MASS},'cells':cells}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--erf',type=Path,required=True);parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,default=Path(__file__).parent/'test_files/SDM_Restart/SDM_Restart.i')
    parser.add_argument('--case',choices=CASES,action='append')
    args=parser.parse_args();assert sys.byteorder=='little';args.work_dir.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='repopulation-tiles-',dir=args.work_dir.resolve()));exe=root/'erf_exec';shutil.copy2(args.erf,exe)
    report={'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'cases':[],'passed':False}
    def save():(root/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    for name in args.case or CASES:
        overrides,ratio,physical=CASES[name];reference={}
        for tile,options in TILES.items():
            out=root/(name+'-'+tile);out.mkdir();shutil.copy2(args.inputs,out/'inputs')
            cmd=[str(exe),'inputs','max_step=2',P+'initial_number_density=128',P+'advect_with_gravity=false',P+'split_merge_amr=true',*FINE,*overrides,*options]
            (out/'command.json').write_text(json.dumps(cmd,indent=2)+'\n')
            with (out/'run.log').open('w') as log:proc=subprocess.run(cmd,cwd=out,stdout=log,stderr=subprocess.STDOUT,timeout=60)
            entry={'name':out.name,'returncode':proc.returncode,'checks':[]};report['cases'].append(entry)
            def check(label,ok,actual,expected):entry['checks'].append({'name':label,'passed':bool(ok),'actual':actual,'expected':expected})
            check('execution',proc.returncode==0,proc.returncode,0)
            if proc.returncode==0:
                checkpoints=sorted(out.glob('chk[0-9]*'));check('checkpoints',len(checkpoints)==3,len(checkpoints),3)
                for checkpoint in checkpoints:
                    step=int(checkpoint.name[3:]);expected=physical*(step if name.startswith('injection') else 1)
                    value=observe(checkpoint,ratio);(out/(checkpoint.name+'-particles.json')).write_text(json.dumps(value,indent=2)+'\n')
                    check(checkpoint.name+'-physical',value['number']==expected,value['number'],expected)
                    if name=='dense':check(checkpoint.name+'-dense-count',value['count']==2304,value['count'],2304)
                    if name=='injection' and step==1:
                        # 256 coarse + 256*8 split fine SDs + 256*3 recipients.
                        check('positive-repopulation',value['count']==3072,value['count'],3072)
                    for field,m in MASS.items():
                        mass=expected*m;tol=64*sys.float_info.epsilon*abs(mass)
                        check(checkpoint.name+'-'+field,abs(value['mass'][field]-mass)<=tol,value['mass'][field],mass)
                    if tile=='untiled':reference[checkpoint.name]=value
                    else:
                        ref=reference[checkpoint.name]
                        check(checkpoint.name+'-SD-count',value['count']==ref['count'],value['count'],ref['count'])
                        a,b=value['cells'],ref['cells'];same_keys=a.keys()==b.keys()
                        check(checkpoint.name+'-occupied-cells',same_keys,len(a),len(b))
                        differing=[key for key in a.keys()&b.keys() if a[key]!=b[key]]
                        # Fixed physical attributes and fixed particle positions
                        # make these CPU per-cell representations exactly
                        # comparable; IDs and random freezing thresholds are
                        # intentionally absent from the observation.
                        check(checkpoint.name+'-cell-representation',same_keys and not differing,differing[:8],[])
                    entry.setdefault('observations',[]).append({'checkpoint':checkpoint.name,'count':value['count'],'number':value['number']})
            entry['passed']=all(c['passed'] for c in entry['checks']);save();print(entry['name'],entry['passed'],flush=True)
    report['passed']=all(c['passed'] for c in report['cases']);save();print(root)
    return 0 if report['passed'] else 1
if __name__=='__main__':sys.exit(main())
