"""Paired full-ERF consumed/pending history checks with deterministic dry controls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
WORK=HERE.parent
from particle_checkpoint_compare import compare as compare_particles

FINE=["amr.max_level=1", "amr.ref_ratio_vect=2 2 2", "amr.max_grid_size=8",
      "amr.max_grid_size_z=16", "amr.n_error_buf=0", "erf.regrid_int=-1",
      "erf.coupling_type=TwoWay", "erf.cf_width=0", "erf.refinement_indicators=patch",
      "erf.patch.max_level=1", "erf.patch.in_box_lo_indices_crse=0 0 0",
      "erf.patch.in_box_hi_indices_crse=7 7 7"]

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--erf",type=Path,required=True)
parser.add_argument("--tool",type=Path,required=True)
parser.add_argument("--work-dir",type=Path,required=True)
parser.add_argument("--inputs",type=Path,default=HERE / "test_files/SDM_RainHistory_Restart/SDM_RainHistory_Restart.i")
args=parser.parse_args()
assert sys.byteorder=="little", "native particle reader requires a little-endian host"
EXE=args.erf.resolve()
TOOL=args.tool.resolve()
args.work_dir.mkdir(parents=True,exist_ok=True)
ROOT=Path(tempfile.mkdtemp(prefix="particle-rain-history-",dir=args.work_dir.resolve()))
shutil.copy2(EXE,ROOT / "erf_exec")
results={"executable_sha256":hashlib.sha256(EXE.read_bytes()).hexdigest(),"runs":[],"checks":[],"comparisons":[]}

def save():
    (ROOT / "results.json").write_text(json.dumps(results,indent=2)+"\n")

def check(name,ok,**values):
    results["checks"].append({"name":name,"passed":bool(ok),**values})

def tool(action,*args):
    run=subprocess.run([str(TOOL),action,*map(str,args)],capture_output=True,text=True,timeout=30)
    assert run.returncode==0,run.stdout+run.stderr
    return [json.loads(line) for line in run.stdout.splitlines() if line.startswith("{")]

def run(name,options,checkpoint=None,binary=None):
    out=ROOT / name;out.mkdir()
    shutil.copy2(args.inputs,out / "inputs")
    command=[str(binary or ROOT / "erf_exec"),"inputs",*options]
    if checkpoint:
        shutil.copytree(checkpoint,out / "input-checkpoint")
        command += ["erf.restart=input-checkpoint"]
    (out / "command.json").write_text(json.dumps(command,indent=2)+"\n")
    with (out / "run.log").open("w") as log:
        process=subprocess.run(command,cwd=out,stdout=log,stderr=subprocess.STDOUT,timeout=90)
    results["runs"].append({"name":name,"returncode":process.returncode})
    save()
    assert process.returncode==0,str(out / "run.log")
    return out

def compare(name,left,right,levels,expected_offsets):
    record={"name":name,"fields":{},"particles":compare_particles(left / "super_droplets_moisture",right / "super_droplets_moisture")}
    results["comparisons"].append(record)
    for lev in levels:
        fields=[p.name[:-2] for p in (left / f"Level_{lev}").glob("*_H")]
        for field in fields:
            offset=expected_offsets.get((lev,field),0)
            rows=tool("compare",left / f"Level_{lev}" / field,right / f"Level_{lev}" / field,offset)
            record["fields"][f"{lev}/{field}"]=rows
            check(name+f"/{lev}/{field}",all(row["max_abs_diff"]==0 for row in rows),expected_offset=offset)
    p=record["particles"]
    check(name+"/particles",p["changed_level_or_grid"]==0 and p["changed_integer_records"]==0 and not p["changed_real_fields"] and p["next_id_equal"])

def main():
    try:
        for coupling in ("OneWay","TwoWay"):
            options=[*FINE,"erf.coupling_type="+coupling,"erf.patch.in_box_hi_indices_crse=7 7 7",
                     "erf.patch.end_time=0.0015","max_step=2","erf.regrid_int=-1",
                     "super_droplets_moisture.initial_species_mean_mass_H2O=0"]
            pre=run(coupling+"-seed",options) / "chk00002"
            for pending in (0,0.25):
                label=coupling+"-pending"+str(pending)
                reference=ROOT / (label+"-reference-checkpoint")
                shutil.copytree(pre,reference)
                if pending:
                    tool("add-bottom-history",reference / "Level_0/RainAccum",pending)
                history=ROOT / (label+"-history-checkpoint")
                shutil.copytree(reference,history)
                for field in ("RainAccum","SlmPrecip"):
                    tool("add-bottom-history",history / "Level_1" / field,1)
                for path in ("regrid","plot"):
                    tail=["max_step=4","erf.regrid_int=1"] if path=="regrid" else ["max_step=4","erf.regrid_int=-1","erf.plot_int_1=1","erf.plot_vars_1=density"]
                    low=run(label+"-"+path+"-reference",options+tail,reference)
                    high=run(label+"-"+path+"-history",options+tail,history)
                    for step in (3,4):
                        lname=label+"-"+path+"-step"+str(step)
                        lchk=low / f"chk{step:05d}";hchk=high / f"chk{step:05d}"
                        levels=(0,) if path=="regrid" else (0,1)
                        for lev in levels:
                            check(lname+f"/level{lev}",(lchk / f"Level_{lev}").is_dir() and (hchk / f"Level_{lev}").is_dir())
                        if path=="regrid":
                            check(lname+"/fine-removed",not (lchk / "Level_1").exists() and not (hchk / "Level_1").exists())
                        offsets={(lev,field):1 for lev in levels for field in ("RainAccum","SlmPrecip")
                                 if path=="regrid" or coupling=="TwoWay" or lev==1}
                        compare(lname,lchk,hchk,levels,offsets)
                        expected=pending/0.001 if step==3 else 0
                        for side,chk in (("reference",lchk),("history",hchk)):
                            row=tool("stats",chk / "Level_0/LsmData16")[0]
                            # LsmData16 keeps zero upper atmospheric cells; only
                            # the 8x8 physical bottom carries the surface rate.
                            check(lname+"/"+side+"/pending-rate",row["min"]==0 and row["max"]==expected and row["sum"]==64*expected,expected=expected,observed=row)
                        save()
    finally:
        results["summary"]={"runs":len(results["runs"]),"checks":len(results["checks"]),"failed":sum(not c["passed"] for c in results["checks"])}
        save()
        print(ROOT)
        print(json.dumps(results["summary"]))

if __name__ == "__main__":
    main()
    if any(not c["passed"] for c in results["checks"]):
        raise SystemExit(1)
