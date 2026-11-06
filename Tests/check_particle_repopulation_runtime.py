#!/usr/bin/env python3
"""Bounded serial-double conservation through particle split/repopulation/merge."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import shutil
import struct
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
MASS={'aerosol_mass_NaCl':1e-19,'species_mass_H2O':4.1887902e-9,'species_mass_ice':1e-9}
MASS['particle_mass']=math.fsum(MASS.values())
CASES={
 'injection':(FINE+INJECT,128,True),
 'injection-low':(FINE+INJECT+[P+'injection.rate=1000'],1,True),
 'injection-limited':(FINE+INJECT+[P+'injection.rate=4000'],4,True),
 'injection-high':(FINE+INJECT+[P+'injection.rate=1024000'],1024,True),
 'injection-full':(FINE+INJECT+['erf.patch.in_box_hi_indices_crse=7 7 7'],128,True),
 'injection-anisotropic':(FINE+INJECT+['amr.ref_ratio_vect=2 2 1'],128,True),
 'injection-ratio4':(FINE+INJECT+['amr.ref_ratio_vect=4 4 4','amr.max_grid_size_z=32',P+'injection.rate=4096000'],4096,True),
 'newlevel':(FINE+['erf.regrid_int=1','erf.patch.start_time=0.0015','max_step=3'],128,False),
 'remake':(FINE+['erf.regrid_int=1','erf.patch.end_time=0.0015','max_step=3',
    'erf.refinement_indicators=patch moved','erf.moved.max_level=1','erf.moved.start_time=0.0015',
    'erf.moved.in_box_lo_indices_crse=4 0 0','erf.moved.in_box_hi_indices_crse=7 7 7'],128,False),
 'derefinement':(FINE+['erf.regrid_int=1','erf.patch.end_time=0.0015','max_step=3'],128,False),
}


def observe(checkpoint):
    data=read(checkpoint/'super_droplets_moisture'); names=data['real_names']; mi=names.index('multiplicity')
    groups=defaultdict(list)
    for lev,grid,ints,v in data['particles'].values():
        assert ints[0]>0, 'fixture must contain only active particles'
        assert v[mi]>0, 'nonpositive multiplicity'
        key=','.join(str(math.floor(x)) for x in v[:3]);groups[key].append(v)
    cells={k:math.fsum(v[mi] for v in vals) for k,vals in groups.items()}
    records=[v for vals in groups.values() for v in vals]
    return {'superdroplets':len(records),'number':math.fsum(cells.values()),'cells':cells,
            'mass':{name:math.fsum(v[mi]*v[names.index(name)] for v in records) for name in MASS}}


def edit_population(checkpoint,multiplicity):
    """Make a controlled native-tag donor fixture in the documented native format."""
    root=checkpoint/'super_droplets_moisture';lines=iter((root/'Header').read_text().splitlines())
    assert next(lines)=='Version_Two_Dot_One_double';dim=int(next(lines));nr=int(next(lines))
    names=[next(lines) for _ in range(nr)];ni=int(next(lines));[next(lines) for _ in range(ni)]
    assert int(next(lines))==1;next(lines);next(lines);finest=int(next(lines));grids=[int(next(lines)) for _ in range(finest+1)]
    assert ni==1 and dim==3
    payloads={}
    for lev,ng in enumerate(grids):
        for grid in range(ng):
            f,n,offset=map(int,next(lines).split())
            if not n:continue
            path=root/f'Level_{lev}'/f'DATA_{f:05d}'
            if path not in payloads:payloads[path]=bytearray(path.read_bytes())
            data=payloads[path]
            for i in range(n):
                struct.pack_into('<i',data,offset+4*(i*(2+ni)+2),lev+1)
                where=offset+4*n*(2+ni)+8*(i*(dim+nr)+dim+names.index('multiplicity'))
                struct.pack_into('<d',data,where,multiplicity)
    for path,data in payloads.items():path.write_bytes(data)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--erf',type=Path,required=True);parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,default=Path(__file__).parent/'test_files/SDM_Restart/SDM_Restart.i')
    parser.add_argument('--case',choices=[*CASES,'controlled-donors'],action='append')
    args=parser.parse_args();assert sys.byteorder=='little';args.work_dir.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='repopulation-',dir=args.work_dir.resolve()));exe=root/'erf_exec';shutil.copy2(args.erf,exe)
    report={'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'cases':[],'passed':False}
    def save():(root/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    def run(name,options,checkpoint=None):
        out=root/name;out.mkdir();shutil.copy2(args.inputs,out/'inputs')
        command=[str(exe),'inputs','max_step=2',P+'initial_number_density=128',P+'advect_with_gravity=false',P+'split_merge_amr=true',*options]
        if checkpoint:shutil.copytree(checkpoint,out/'input-checkpoint');command+=['erf.restart=input-checkpoint']
        (out/'command.json').write_text(json.dumps(command,indent=2)+'\n')
        with (out/'run.log').open('w') as log:proc=subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,timeout=60)
        entry={'name':name,'returncode':proc.returncode,'checks':[]};report['cases'].append(entry)
        entry['checks'].append({'name':'execution','passed':proc.returncode==0,'actual':proc.returncode,'expected':0});save()
        return out,entry
    def check(entry,name,actual,expected):
        # Number/cell budgets are exact for these binary-representable counts.
        # Allow only bounded weighted-merge round-off for masses.
        tol=64*sys.float_info.epsilon*abs(expected) if name.startswith('mass-') else 0
        entry['checks'].append({'name':name,'passed':abs(actual-expected)<=tol,'actual':actual,'expected':expected,'tolerance':tol})
    def budget(entry,checkpoint,number_per_cell):
        value=observe(checkpoint);check(entry,'number-'+checkpoint.name,value['number'],512*number_per_cell)
        check(entry,'cell-budget-'+checkpoint.name,max([abs(v-number_per_cell) for v in value['cells'].values()]+[0]),0)
        check(entry,'occupied-cells-'+checkpoint.name,len(value['cells']),512 if number_per_cell else 0)
        for name,mass in MASS.items():check(entry,'mass-'+name+'-'+checkpoint.name,value['mass'][name],512*number_per_cell*mass)
        entry.setdefault('observations',[]).append({'checkpoint':checkpoint.name,**{k:v for k,v in value.items() if k!='cells'}})
    selected=args.case or [*CASES,'controlled-donors']
    for name in selected:
        if name=='controlled-donors':continue
        options,density,injection=CASES[name];out,entry=run(name,options)
        if entry['returncode']==0:
            checkpoints=sorted(out.glob('chk[0-9]*'))
            check(entry,'checkpoint-count',len(checkpoints),4 if name in ('newlevel','remake','derefinement') else 3)
            for chk in checkpoints:budget(entry,chk,density*int(chk.name[3:]) if injection else density)
        entry['passed']=all(c['passed'] for c in entry['checks']);save();print(name,entry['passed'],flush=True)
    if 'controlled-donors' in selected:
        seed,entry=run('donor-seed',FINE+INJECT+[P+'split_merge_amr=false','max_step=1'])
        entry['passed']=entry['returncode']==0
        if entry['passed']:
            for multiplicity in (0.5,1,1.5,3,4,10.5,128.5):
                ck=root/f'donor-{multiplicity}';shutil.copytree(seed/'chk00001',ck);edit_population(ck,multiplicity)
                out,entry=run(f'donor-{multiplicity}-resumed',FINE+['max_step=1','erf.check_int=-1','erf.check_per=1'],ck)
                if entry['returncode']==0:
                    budget(entry,out/'chk00001',multiplicity)
                    value=observe(out/'chk00001')
                    # Each fine donor has exactly three empty face neighbors
                    # within its coarse parent. All need unit multiplicity.
                    check(entry,'repopulation-count',value['superdroplets'],1280 if multiplicity>=4 else 512)
                entry['passed']=all(c['passed'] for c in entry['checks']);save();print(entry['name'],entry['passed'],flush=True)
    report['passed']=all(c['passed'] for c in report['cases']);save();print(root)
    return 0 if report['passed'] else 1
if __name__=='__main__':sys.exit(main())
