#include <gtest/gtest.h>
#include <AMReX_ParmParse.H>
#include <ERF_TerrainMetrics.H>
#include <ERF_TerrainPoisson_3D_K.H>
#include <ERF_Utils.H>
#include <cmath>
#include <limits>

using namespace amrex;

namespace {

// These normal-valued grids go through the same coordinate mapping and metric
// validation as ERF. Only interior cells are sampled, so no physical BC stencil
// or periodic solution ghost enters the analytical expectations below.
struct MappedPoissonGrid {
    Box domain{IntVect(0),IntVect(3,3,7)};
    Geometry geom{domain,RealBox(0,0,0,4,4,2048),0,{1,1,0}};
    BoxArray ba{domain};
    DistributionMapping dm{ba};
    MultiFab z{convert(ba,IntVect::TheNodeVector()),dm,1,3};
    MultiFab cc{ba,dm,1,2},jac{ba,dm,1,0},sol{ba,dm,1,2},out{ba,dm,1,0};
    MultiFab ax{convert(ba,IntVect(1,0,0)),dm,1,0};
    MultiFab ay{convert(ba,IntVect(0,1,0)),dm,1,0};
    MultiFab az{convert(ba,IntVect(0,0,1)),dm,1,0};
    Real ulp=std::nextafter(Real(1000),Real(2000))-Real(1000);

    MappedPoissonGrid(bool thin,int slope_ulps,int axis,bool stored_saddle=false) {
        const Real first=thin?5*ulp:Real(1);
        const Real slope=Real(slope_ulps)*ulp;
        ParmParse pp("erf");
        pp.remove("terrain_z_levels");
        pp.addarr("terrain_z_levels",Vector<Real>{0,first,3*first,6*first,250,500,1000,1500,2048});
        Vector<Vector<Real>> nodes(1),heights(1);
        Vector<Gpu::DeviceVector<Real>> device(1);
        init_zlevels(nodes,heights,device,Vector<Geometry>{geom},{},0,0,0);
        init_default_zphys(0,geom,z,cc,0);
        for(MFIter mfi(z);mfi.isValid();++mfi) {
            const auto a=z.array(mfi);
            ParallelFor(makeSlab(mfi.fabbox(),2,0),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                a(i,j,k)=Real(1000)+slope*Real((i&3)+2*(j&3));
            });
        }
        GpuArray<ERF_BC,6> bc{ERF_BC::periodic,ERF_BC::periodic,ERF_BC::slip_wall,
                             ERF_BC::periodic,ERF_BC::periodic,ERF_BC::slip_wall};
        if(stored_saddle) {
            const Real maximum=std::numeric_limits<Real>::max();
            for(MFIter mfi(z);mfi.isValid();++mfi) {
                const auto a=z.array(mfi);
                ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                    a(i,j,k)=maximum*Real(.75)*Real(((i+j)&1)?1:-1)+Real(k)*(maximum/64);
                });
            }
        } else {
            make_terrain_fitted_coords(0,geom,z,nodes[0],bc,FineTerrain::None,nullptr);
        }
        make_J(geom,z,jac);
        make_areas(geom,z,ax,ay,az);
        for(MFIter mfi(sol);mfi.isValid();++mfi) {
            const auto x=sol.array(mfi);
            ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
                x(i,j,k)=axis==1?Real(j):(axis==4?Real(1):Real(k));
            });
        }
        Gpu::streamSynchronize();
    }

    ~MappedPoissonGrid() {ParmParse pp("erf");pp.remove("terrain_z_levels");}

    Real flux() const {
        const auto inv=geom.InvCellSizeArray();
        return terrpoisson_flux_z(1,1,1,sol.const_array(0),z.const_array(0),inv[0],inv[1]);
    }
    Real apply() {
        const auto inv=geom.InvCellSizeArray();
        terrpoisson_adotx(1,1,1,out.array(0),sol.const_array(0),ax.const_array(0),
                         ay.const_array(0),az.const_array(0),jac.const_array(0),z.const_array(0),
                         inv[0],inv[1],inv[2]);
        return out.const_array(0)(1,1,1);
    }
};


TEST(TerrainPoissonVerticalMetrics, MappedThinLayersUseStoredDistances) {
    MappedPoissonGrid grid(true,3,2);
    const Real u=grid.ulp;
    const Real expected=-Real(1)/60-Real(.75)*u*u;
    const Real tolerance=64*std::numeric_limits<Real>::epsilon()*std::abs(expected);
    EXPECT_NEAR(grid.apply()*u*u,expected,tolerance);
}

TEST(TerrainPoissonVerticalMetrics, LargeHeightOffsetKeepsFiniteReciprocal) {
    MappedPoissonGrid grid(false,0,2);
    const int exponent=std::numeric_limits<Real>::max_exponent;
    const Real base=std::scalbn(Real(.75),exponent);
    const Real spacing=std::scalbn(Real(1),exponent-6);
    for(MFIter mfi(grid.z);mfi.isValid();++mfi) {
        const auto z=grid.z.array(mfi);
        ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            z(i,j,k)=base+Real(k)*spacing;
        });
    }
    make_J(grid.geom,grid.z,grid.jac);
    Gpu::streamSynchronize();
    EXPECT_EQ(terrpoisson_inv_vertical_spacing(1,1,1,grid.z.const_array(0)),
              std::scalbn(Real(1),6-exponent));
}

TEST(TerrainPoissonVerticalMetrics, TwoLayerSpanMayExceedMaximum) {
    MappedPoissonGrid grid(false,0,2);
    const int exponent=std::numeric_limits<Real>::max_exponent;
    const Real spacing=std::scalbn(Real(.75),exponent);
    const Real outer=std::scalbn(Real(1),exponent-6);
    for(MFIter mfi(grid.z);mfi.isValid();++mfi) {
        const auto z=grid.z.array(mfi);
        ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            z(i,j,k)=k<=0 ? -spacing+Real(k)*outer :
                (k==1 ? Real(0) : spacing+Real(k-2)*outer);
        });
    }
    make_J(grid.geom,grid.z,grid.jac);
    Gpu::streamSynchronize();
    const Real actual=terrpoisson_inv_vertical_spacing(1,1,1,grid.z.const_array(0));
    EXPECT_GT(actual,Real(0));
    EXPECT_EQ(actual,Real(1)/spacing);
}

TEST(TerrainPoissonVerticalMetrics, SubnormalSpacingKeepsFiniteFlux) {
    MappedPoissonGrid grid(false,0,2);
    const Real spacing=std::numeric_limits<Real>::min()/2;
    const Real tiny=std::numeric_limits<Real>::denorm_min();
    for(MFIter mfi(grid.z);mfi.isValid();++mfi) {
        const auto z=grid.z.array(mfi);
        ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            z(i,j,k)=Real(k)*spacing+Real((i+j+k)&3)*tiny;
        });
    }
    for(MFIter mfi(grid.sol);mfi.isValid();++mfi) {
        const auto x=grid.sol.array(mfi);
        ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) {
            x(i,j,k)=Real(i*k);
        });
    }
    make_J(grid.geom,grid.z,grid.jac);
    Gpu::streamSynchronize();
    const Real flux=terrpoisson_flux_z(2,1,1,grid.sol.const_array(0),grid.z.const_array(0),Real(1),Real(1));
    EXPECT_TRUE(std::isfinite(flux));
    EXPECT_EQ(flux,-std::nextafter(std::numeric_limits<Real>::max(),Real(0)));
}

TEST(TerrainPoissonVerticalMetrics, LargeStoredSaddlePreservesConstant) {
    MappedPoissonGrid grid(false,0,4,true);
    EXPECT_EQ(grid.flux(),Real(0));
    EXPECT_EQ(grid.apply(),Real(0));
}


// BTF's real bottom extrapolation can round the ghost node back to the surface
// at a negative binade boundary. The valid first layer remains positive.
TEST(TerrainPoissonVerticalMetrics, CollapsedBottomGhostKeepsFiniteReciprocal) {
    MappedPoissonGrid grid(false,0,4);
    const Real surface=Real(-1);
    const Real first=std::nextafter(surface,Real(0))-surface;
    const Vector<Real> levels{0,first,3*first,6*first,250,500,1000,1500,2048};
    grid.z.setVal(surface);
    GpuArray<ERF_BC,6> bc{ERF_BC::periodic,ERF_BC::periodic,ERF_BC::slip_wall,
                         ERF_BC::periodic,ERF_BC::periodic,ERF_BC::slip_wall};
    make_terrain_fitted_coords(0,grid.geom,grid.z,levels,bc,FineTerrain::None,nullptr);
    MultiFab jac(grid.ba,grid.dm,1,1);
    make_J(grid.geom,grid.z,jac);
    Gpu::streamSynchronize();
    const auto z=grid.z.const_array(0);
    ASSERT_EQ(z(1,1,0),surface);
    ASSERT_EQ(z(1,1,1)-z(1,1,0),first);
    ASSERT_EQ(z(1,1,-1),z(1,1,0));
    EXPECT_EQ(terrpoisson_inv_vertical_spacing(1,1,0,z),Real(2)/first);
}

} // namespace
