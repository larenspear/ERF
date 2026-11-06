#!/usr/bin/env python3
"""Serial CPU-double tile-local particle creation and allocation regression."""
import argparse
import hashlib
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
TRACER=['erf.moisture_model=None','erf.use_tracer_particles=1',
        'tracer_particles.place_randomly_in_cells=false','tracer_particles.advect_with_flow=false',
        'tracer_particles.advect_with_gravity=false']
INJECT=[P+'num_initializations=0',P+'num_injections=1',P+'injection.rate=128000',
        P+'injection.particles_per_cell=1',P+'injection.multiplicity_type=constant',
        P+'injection.species_mean_mass_H2O=4.1887902e-9',
        P+'injection.species_mean_mass_ice=1e-9',P+'injection.aerosol_mean_mass_NaCl=1e-19','max_step=2']
CASES={
 'sd-base':([],512,65536),
 'sd-fine':(FINE,2304,65536),
 'sd-partial':([P+'particle_box_hi=2 8 8'],128,16384),
 'sd-empty':([P+'initial_number_density=0'],0,0),
 'sd-multiple':([P+'num_initializations=2'],1024,131072),
 'sd-injection':(INJECT,512,65536),
 'tracer-base':(TRACER,512,512),
 'tracer-partial':(TRACER+['tracer_particles.particle_box_hi=2 8 8'],128,128),
 'tracer-empty':(TRACER+['tracer_particles.initial_particles_per_cell=0'],0,0),
}
TILES={'untiled':['particles.do_tiling=false'],
       'regular':['particles.do_tiling=true','particles.tile_size=4 4 4'],
       'uneven':['particles.do_tiling=true','particles.tile_size=3 5 2'],
       'unit':['particles.do_tiling=true','particles.tile_size=1 1 1']}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--erf',type=Path,required=True);parser.add_argument('--work-dir',type=Path,required=True)
    parser.add_argument('--inputs',type=Path,default=Path(__file__).parent/'test_files/SDM_Restart/SDM_Restart.i')
    parser.add_argument('--case',choices=CASES,action='append');parser.add_argument('--tile',choices=TILES,action='append')
    args=parser.parse_args();assert sys.byteorder=='little';args.work_dir.mkdir(parents=True,exist_ok=True)
    root=Path(tempfile.mkdtemp(prefix='particle-tiles-',dir=args.work_dir.resolve()));exe=root/'erf_exec';shutil.copy2(args.erf,exe)
    report={'binary_sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'cases':[],'passed':False}
    def save():(root/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    for name in args.case or CASES:
        overrides,sd,physical=CASES[name];tracer=name.startswith('tracer-');injection=name=='sd-injection'
        for tile in args.tile or TILES:
            out=root/(name+'-'+tile);out.mkdir();shutil.copy2(args.inputs,out/'inputs')
            command=[str(exe),'inputs','max_step=1',P+'initial_number_density=128',P+'advect_with_gravity=false',*overrides,*TILES[tile]]
            (out/'command.json').write_text(json.dumps(command,indent=2)+'\n')
            with (out/'run.log').open('w') as log:proc=subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,timeout=60)
            entry={'name':out.name,'returncode':proc.returncode,'checks':[]};report['cases'].append(entry)
            def check(label,actual,expected,tol=0):entry['checks'].append({'name':label,'passed':abs(actual-expected)<=tol,'actual':actual,'expected':expected,'tolerance':tol})
            check('execution',proc.returncode,0)
            if proc.returncode==0:
                checkpoints=sorted(out.glob('chk[0-9]*'));check('checkpoint-count',len(checkpoints),3 if injection else 2)
                for checkpoint in checkpoints:
                    step=int(checkpoint.name[3:]);factor=step if injection else 1
                    try:
                        data=read(checkpoint/('tracer_particles' if tracer else 'super_droplets_moisture'))
                    except (AssertionError, ValueError, IndexError) as error:
                        check(checkpoint.name+'-valid-payload',1,0)
                        entry.setdefault('read_errors',[]).append({'checkpoint':checkpoint.name,'error':repr(error)})
                        continue
                    values=[v[3] for v in data['particles'].values()];names=data['real_names']
                    mi=None if tracer else names.index('multiplicity')
                    number=len(values) if tracer else math.fsum(v[mi] for v in values)
                    check(checkpoint.name+'-count',len(values),sd*factor)
                    check(checkpoint.name+'-physical',number,physical*factor)
                    # Every allocation in these source-only fixtures creates a
                    # valid particle. Prefix-sum over-allocation leaves ID gaps
                    # even when later filtering hides uninitialized records.
                    check(checkpoint.name+'-allocation',data['next_id'],sd*factor+1)
                    masses={'mass':1e-6} if tracer else {'particle_mass':math.fsum([1e-19,4.1887902e-9,1e-9]),
                        'aerosol_mass_NaCl':1e-19,'species_mass_H2O':4.1887902e-9,'species_mass_ice':1e-9}
                    for field,mass in masses.items():
                        total=math.fsum((1 if tracer else v[mi])*v[names.index(field)] for v in values)
                        expected=mass*physical*factor
                        check(checkpoint.name+'-'+field,total,expected,64*sys.float_info.epsilon*abs(expected))
                    check(checkpoint.name+'-in-domain',sum(not all(0<=x<8 for x in v[:3]) for v in values),0)
                    entry.setdefault('observations',[]).append({'checkpoint':checkpoint.name,'count':len(values),'number':number,'next_id':data['next_id']})
            entry['passed']=all(c['passed'] for c in entry['checks']);save();print(entry['name'],entry['passed'],flush=True)
    report['passed']=all(c['passed'] for c in report['cases']);save();print(root)
    return 0 if report['passed'] else 1
if __name__=='__main__':sys.exit(main())
