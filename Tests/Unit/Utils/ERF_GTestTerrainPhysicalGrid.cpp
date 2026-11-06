#include <gtest/gtest.h>

#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <ERF_TerrainMetrics.H>
#include <ERF_Utils.H>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace amrex;

namespace {

// Keep device-launch helpers at namespace scope for CUDA compilation.
template <class F>
void expect_result(const char* name, bool reject, F&& f)
{
    SCOPED_TRACE(name);
    bool rejected = false;
    try { f(); }
    catch (std::runtime_error const& e) {
        rejected = true;
        EXPECT_NE(std::string(e.what()).find("Physical vertical grid"), std::string::npos)
            << e.what();
    }
    EXPECT_EQ(rejected, reject);
}

class TerrainPhysicalGrid : public ::testing::Test
{
protected:
    bool previous_throw;
    bool had_levels;
    bool had_smoothing;
    Vector<Real> previous_levels;
    int previous_smoothing;

    void SetUp() override
    {
        previous_throw = amrex::system::throw_exception;
        amrex::system::throw_exception = true;
        ParmParse pp("erf");
        had_levels = pp.queryarr("terrain_z_levels", previous_levels);
        had_smoothing = pp.query("terrain_smoothing", previous_smoothing);
        pp.remove("terrain_z_levels");
        pp.remove("terrain_smoothing");
        pp.add("terrain_smoothing", 0);
    }

    void TearDown() override
    {
        ParmParse pp("erf");
        pp.remove("terrain_z_levels");
        pp.remove("terrain_smoothing");
        if (had_levels) pp.addarr("terrain_z_levels", previous_levels);
        if (had_smoothing) pp.add("terrain_smoothing", previous_smoothing);
        amrex::system::throw_exception = previous_throw;
    }
};

Geometry geometry(IntVect lo, int nz)
{
    const Box domain(lo,lo+IntVect(3,3,nz-1));
    const RealBox rb(0.,0.,0.,4.,4.,Real(nz));
    return Geometry(domain,rb,CoordSys::cartesian,{1,1,0});
}

void mapped(const char* label, Real surface, Vector<Real> levels, bool reject)
{
    const int nz=levels.size()-1;
    const Box domain(IntVect(0),IntVect(3,3,nz-1));
    const RealBox rb(0.,0.,0.,4.,4.,levels.back());
    const Geometry geom(domain,rb,CoordSys::cartesian,{1,1,0});
    ParmParse pp("erf"); pp.remove("terrain_z_levels"); pp.addarr("terrain_z_levels",levels);
    Vector<Vector<Real>> nodes(1),heights(1);
    Vector<Gpu::DeviceVector<Real>> device(1);
    init_zlevels(nodes,heights,device,Vector<Geometry>{geom},{},0.,0.,0.);
    const BoxArray ba(domain), ndba(amrex::convert(ba,IntVect::TheNodeVector()));
    const DistributionMapping dm(ba);
    MultiFab z(ndba,dm,1,IntVect(3)),cc(ba,dm,1,2),jac(ba,dm,1,1);
    init_default_zphys(0,geom,z,cc,0.);
    for(MFIter mfi(z);mfi.isValid();++mfi) {
        const auto a=z.array(mfi);
        ParallelFor(makeSlab(mfi.fabbox(),2,0),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=surface;});
    }
    GpuArray<ERF_BC,6> bc{ERF_BC::periodic,ERF_BC::periodic,ERF_BC::slip_wall,
                         ERF_BC::periodic,ERF_BC::periodic,ERF_BC::slip_wall};
    expect_result(label,reject,[&]{
        make_terrain_fitted_coords(0,geom,z,nodes[0],bc,FineTerrain::None,nullptr);
        make_J(geom,z,jac);
        const Real minimum=get_dzmin_terrain(z);
        EXPECT_GT(minimum, Real(0));
        EXPECT_TRUE(std::isfinite(minimum));
    });
}

enum class Fault { None, Collapse, Nan, Inf, Overflow, HighGhost, LowGhost };

void direct(const char* label, IntVect lo, int nz, Fault fault)
{
    const Geometry geom=geometry(lo,nz);
    const BoxArray ba(geom.Domain()),ndba(amrex::convert(ba,IntVect::TheNodeVector()));
    const DistributionMapping dm(ba);
    MultiFab z(ndba,dm,1,IntVect(3)),jac(ba,dm,1,1);
    for(MFIter mfi(z);mfi.isValid();++mfi) {
        const auto a=z.array(mfi); const int klo=lo[2];
        ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k){a(i,j,k)=Real(1000)+Real(k-klo);});
        IntVect point=lo+IntVect(1,1,1);
        if(fault==Fault::HighGhost) { point=geom.Domain().bigEnd()+IntVect(2); }
        if(fault==Fault::LowGhost) { point=lo+IntVect(-1,0,1); }
        const int i=point[0],j=point[1],k=point[2];
        if(fault!=Fault::None) {
            const Box pointbox(point,point);
            ParallelFor(pointbox,[=] AMREX_GPU_DEVICE(int,int,int){
                if(fault==Fault::Nan) a(i,j,k)=std::numeric_limits<Real>::quiet_NaN();
                else if(fault==Fault::Inf) a(i,j,k)=std::numeric_limits<Real>::infinity();
                else if(fault==Fault::Overflow) {
                    a(i,j,k-1)=-std::numeric_limits<Real>::max();
                    a(i,j,k)=std::numeric_limits<Real>::max();
                } else a(i,j,k)=a(i,j,k-1);
            });
        }
    }
    const bool reject=fault!=Fault::None;
    expect_result((std::string(label)+" make_J").c_str(),reject,[&]{make_J(geom,z,jac);});
    // The original spacing reduction reads high ghosts but no low ghosts.
    expect_result((std::string(label)+" dzmin").c_str(),reject&&fault!=Fault::LowGhost,[&]{get_dzmin_terrain(z);});
}

void eb_offset()
{
    const Geometry geom=geometry(IntVect(0),1);
    const BoxArray ba(geom.Domain()),ndba(amrex::convert(ba,IntVect::TheNodeVector()));
    const DistributionMapping dm(ba);
    MultiFab z(ndba,dm,1,IntVect(3)),cc(ba,dm,1,2);
    init_default_zphys(0,geom,z,cc,Real(-1.e20));
    expect_result("EB plane offset collapse at dzmin",true,[&]{get_dzmin_terrain(z);});
}
}

TEST_F(TerrainPhysicalGrid, OrdinaryTranslatedControl)
{
    mapped("ordinary translated control",1000.,{0.,1.,10.,100.,250.,500.,1000.,1500.,2048.},false);
}

TEST_F(TerrainPhysicalGrid, TinyZeroSurfaceControl)
{
    mapped("tiny zero-surface control",0.,{0.,1.e-14,10.,100.,250.,500.,1000.,1500.,2048.},false);
}

TEST_F(TerrainPhysicalGrid, RejectsTranslatedFirstLayerCollapse)
{
    mapped("translated first-layer collapse",1000.,{0.,1.e-14,10.,100.,250.,500.,1000.,1500.,2048.},true);
}

TEST_F(TerrainPhysicalGrid, RejectsTranslatedInteriorLayerCollapse)
{
    const Real next = Real(1) + Real(8)*std::numeric_limits<Real>::epsilon();
    mapped("translated interior-layer collapse",1000.,{0.,1.,next,100.,250.,500.,1000.,1500.,2048.},true);
}

TEST_F(TerrainPhysicalGrid, OneLayerTranslatedControl)
{
    mapped("one-layer translated control",1000.,{0.,2048.},false);
}

TEST_F(TerrainPhysicalGrid, ConstantFiniteControl)
{
    direct("constant finite control",IntVect(0),8,Fault::None);
}

TEST_F(TerrainPhysicalGrid, OneCellNonzeroOriginControl)
{
    direct("one-cell nonzero-origin control",IntVect(4,5,7),1,Fault::None);
}

TEST_F(TerrainPhysicalGrid, RejectsMixedCornerCollapse)
{
    direct("mixed corner collapse",IntVect(0),8,Fault::Collapse);
}

TEST_F(TerrainPhysicalGrid, RejectsNonfiniteNodes)
{
    direct("NaN node",IntVect(0),8,Fault::Nan);
    direct("infinite node",IntVect(0),8,Fault::Inf);
}

TEST_F(TerrainPhysicalGrid, RejectsOverflowingLayer)
{
    direct("overflowing layer",IntVect(0),8,Fault::Overflow);
}

TEST_F(TerrainPhysicalGrid, RejectsConsumedHighGhostCollapse)
{
    direct("high ghost corner collapse",IntVect(0),8,Fault::HighGhost);
}

TEST_F(TerrainPhysicalGrid, RejectsOnlyConsumedLowGhostCollapse)
{
    direct("low ghost collapse",IntVect(0),8,Fault::LowGhost);
}

TEST_F(TerrainPhysicalGrid, RejectsEBPlaneOffsetCollapse)
{
    eb_offset();
}
