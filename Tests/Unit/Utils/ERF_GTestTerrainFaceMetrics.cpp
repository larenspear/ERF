#include <gtest/gtest.h>
#include <AMReX.H>
#include <AMReX_ParmParse.H>
#include <ERF_TerrainMetrics.H>
#include <ERF_Utils.H>
#include <cmath>
#include <cfenv>
#include <cstdint>
#include <limits>
#include <string>

// These source-linked checks inspect CPU arrays and call device-only metric
// helpers on their CPU backend. GPU execution needs a separate device driver.
#ifndef AMREX_USE_GPU
using namespace amrex;
namespace {
class TerrainFaceMetrics : public ::testing::Test
{
protected:
    bool previous_throw, had_levels, had_smoothing;
    Vector<Real> previous_levels;
    int previous_smoothing;
    void SetUp() override
    {
        previous_throw=amrex::system::throw_exception;
        amrex::system::throw_exception=true;
        ParmParse pp("erf");
        had_levels=pp.queryarr("terrain_z_levels",previous_levels);
        had_smoothing=pp.query("terrain_smoothing",previous_smoothing);
        pp.remove("terrain_z_levels"); pp.remove("terrain_smoothing");
        pp.add("terrain_smoothing",0);
    }
    void TearDown() override
    {
        ParmParse pp("erf");
        pp.remove("terrain_z_levels"); pp.remove("terrain_smoothing");
        if(had_levels) pp.addarr("terrain_z_levels",previous_levels);
        if(had_smoothing) pp.add("terrain_smoothing",previous_smoothing);
        amrex::system::throw_exception=previous_throw;
    }
};

void check_metric(const std::string& name, const char* metric, Real actual, Real inv,
                  Real scale, const Array4<const Real>& z,
                  std::initializer_list<IntVect> columns, int upper, int lower)
{
    SCOPED_TRACE(name+" "+metric);
    // Every stored node in this dyadic fixture lies on the ulp(1000) lattice.
    // Accumulate signed integer coordinates, avoiding floating averages of
    // large absolute elevations in the reference calculation.
    const Real unit=std::nextafter(Real(1000),Real(2000))-Real(1000);
    std::int64_t ticks=0;
    for(const auto& p:columns) {
        const Real hi=(z(p[0],p[1],upper)-Real(1000))/unit;
        const Real lo=(z(p[0],p[1],lower)-Real(1000))/unit;
        const auto hi_ticks=static_cast<std::int64_t>(hi);
        const auto lo_ticks=static_cast<std::int64_t>(lo);
        ASSERT_EQ(Real(hi_ticks),hi);
        ASSERT_EQ(Real(lo_ticks),lo);
        ASSERT_GT(hi_ticks,lo_ticks);
        ticks+=hi_ticks-lo_ticks;
    }
    const Real expected=Real(ticks)*unit*inv*scale;
    ASSERT_TRUE(std::isfinite(actual));
    EXPECT_EQ(actual,expected);
}

void run_case(const std::string& name, Real surface, Real slope, Real first)
{
    const Box domain(IntVect(0),IntVect(3,3,7));
    const RealBox rb(0.,0.,0.,4.,4.,2048.);
    const Geometry geom(domain,rb,CoordSys::cartesian,{1,1,0});
    ParmParse pp("erf"); pp.remove("terrain_z_levels");
    pp.addarr("terrain_z_levels",Vector<Real>{0.,first,Real(3)*first,Real(6)*first,250.,500.,1000.,1500.,2048.});
    Vector<Vector<Real>> nodes(1), heights(1);
    Vector<Gpu::DeviceVector<Real>> device(1);
    init_zlevels(nodes,heights,device,Vector<Geometry>{geom},{},0.,0.,0.);
    const BoxArray ba(domain), ndba(convert(ba,IntVect::TheNodeVector()));
    const DistributionMapping dm(ba);
    MultiFab z(ndba,dm,1,IntVect(3)),cc(ba,dm,1,2);
    MultiFab ax(convert(ba,IntVect(1,0,0)),dm,1,0);
    MultiFab ay(convert(ba,IntVect(0,1,0)),dm,1,0);
    MultiFab az(convert(ba,IntVect(0,0,1)),dm,1,0);
    init_default_zphys(0,geom,z,cc,0.);
    for (MFIter mfi(z);mfi.isValid();++mfi) {
        const auto a=z.array(mfi);
        ParallelFor(makeSlab(mfi.fabbox(),2,0),[=] AMREX_GPU_DEVICE(int i,int j,int k){
            a(i,j,k)=surface+slope*Real((i&1)+2*(j&1));
        });
    }
    GpuArray<ERF_BC,6> bc{ERF_BC::periodic,ERF_BC::periodic,ERF_BC::slip_wall,
                         ERF_BC::periodic,ERF_BC::periodic,ERF_BC::slip_wall};
    make_terrain_fitted_coords(0,geom,z,nodes[0],bc,FineTerrain::None,nullptr);
    make_areas(geom,z,ax,ay,az);
    const auto inv=geom.InvCellSizeArray();
    for (MFIter mfi(z);mfi.isValid();++mfi) {
        const auto a=z.const_array(mfi);
        for (int k=0;k<2;++k) {
            const auto label=name+"-k"+std::to_string(k);
            check_metric(label,"ax",ax.const_array(mfi)(0,0,k),inv[2],Real(.5),a,{{0,0,0},{0,1,0}},k+1,k);
            check_metric(label,"ay",ay.const_array(mfi)(0,0,k),inv[2],Real(.5),a,{{0,0,0},{1,0,0}},k+1,k);
            check_metric(label,"cell",Compute_h_zeta_AtCellCenter(0,0,k,inv,a),inv[2],Real(.25),a,
                 {{0,0,0},{1,0,0},{0,1,0},{1,1,0}},k+1,k);
            check_metric(label,"iface",Compute_h_zeta_AtIface(0,0,k,inv,a),inv[2],Real(.5),a,{{0,0,0},{0,1,0}},k+1,k);
            check_metric(label,"jface",Compute_h_zeta_AtJface(0,0,k,inv,a),inv[2],Real(.5),a,{{0,0,0},{1,0,0}},k+1,k);
            check_metric(label,"kface",Compute_h_zeta_AtKface(0,0,k,inv,a),inv[2],Real(.125),a,
                 {{0,0,0},{1,0,0},{0,1,0},{1,1,0}},k+1,k-1);
            check_metric(label,"edge_i",Compute_h_zeta_AtEdgeCenterI(0,0,k,inv,a),inv[2],Real(.25),a,{{0,0,0},{1,0,0}},k+1,k-1);
            check_metric(label,"edge_j",Compute_h_zeta_AtEdgeCenterJ(0,0,k,inv,a),inv[2],Real(.25),a,{{0,0,0},{0,1,0}},k+1,k-1);
        }
    }
}

void extreme_metrics(bool tiny)
{
    const Box domain(IntVect(0),IntVect(1,1,0));
    const Geometry geom(domain,RealBox(0.,0.,0.,2.,2.,1.),CoordSys::cartesian,{0,0,0});
    const BoxArray ba(domain), ndba(convert(ba,IntVect::TheNodeVector()));
    const DistributionMapping dm(ba);
    MultiFab z(ndba,dm,1,1);
    MultiFab ax(convert(ba,IntVect(1,0,0)),dm,1,0);
    MultiFab ay(convert(ba,IntVect(0,1,0)),dm,1,0);
    MultiFab az(convert(ba,IntVect(0,0,1)),dm,1,0);
    const Real unit=tiny ? std::numeric_limits<Real>::denorm_min()
                        : std::numeric_limits<Real>::max()/Real(8);
    for(MFIter mfi(z);mfi.isValid();++mfi) {
        const auto a=z.array(mfi);
        ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            a(i,j,k)=tiny ? Real(k)*unit
                : (k<0 ? -Real(4)*unit : (k==0 ? -Real(3)*unit : Real(3)*unit));
        });
    }
    std::feclearexcept(FE_ALL_EXCEPT);
    make_areas(geom,z,ax,ay,az);
    const auto inv=geom.InvCellSizeArray();
    for(MFIter mfi(z);mfi.isValid();++mfi) {
        const auto a=z.const_array(mfi);
        // All columns are identical, so their mean must equal one stored edge.
        const Real adjacent=a(0,0,1)-a(0,0,0);
        const Real centered=Real(.5)*(a(0,0,1)-a(0,0,-1));
        ASSERT_GT(adjacent,Real(0)); ASSERT_TRUE(std::isfinite(adjacent));
        EXPECT_EQ(ax.const_array(mfi)(0,0,0),adjacent);
        EXPECT_EQ(ay.const_array(mfi)(0,0,0),adjacent);
        EXPECT_EQ(Compute_h_zeta_AtCellCenter(0,0,0,inv,a),adjacent);
        EXPECT_EQ(Compute_h_zeta_AtIface(0,0,0,inv,a),adjacent);
        EXPECT_EQ(Compute_h_zeta_AtJface(0,0,0,inv,a),adjacent);
        EXPECT_EQ(Compute_h_zeta_AtKface(0,0,0,inv,a),centered);
        EXPECT_EQ(Compute_h_zeta_AtEdgeCenterI(0,0,0,inv,a),centered);
        EXPECT_EQ(Compute_h_zeta_AtEdgeCenterJ(0,0,0,inv,a),centered);
        EXPECT_EQ(std::fetestexcept(FE_OVERFLOW),0);
    }
}
}

TEST_F(TerrainFaceMetrics, OrdinaryFlatControl)
{
    run_case("ordinary-flat",1000.,0.,1.);
}

TEST_F(TerrainFaceMetrics, OrdinarySlopingControl)
{
    run_case("ordinary-slope",1000.,100.,1.);
}

TEST_F(TerrainFaceMetrics, MappedThinLayersMatchStoredCoordinates)
{
    const Real unit=std::nextafter(Real(1000),Real(2000))-Real(1000);
    for(int n=3;n<=40;++n) {
        run_case("thin-slope-"+std::to_string(n),1000.,100.,Real(n)*unit);
    }
}

TEST_F(TerrainFaceMetrics, FiniteLargeStoredEdgesHaveFiniteAverages)
{
    extreme_metrics(false);
}

TEST_F(TerrainFaceMetrics, SubnormalStoredEdgesSurviveAveraging)
{
    extreme_metrics(true);
}
#endif
