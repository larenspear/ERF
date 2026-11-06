#include <gtest/gtest.h>
#include "ERF_GroundMassRemap.H"
#include "ERF_GroundMassCheckpoint.H"

namespace {
using namespace amrex;
Geometry geometry(const Box& box) {
    return Geometry(box,RealBox(0,0,0,8,6,4),0,{0,0,0});
}
MultiFab field(const BoxArray& ba,Real fill) {
    MultiFab mf(ba,DistributionMapping(ba),1,1);mf.setVal(fill);return mf;
}
void surface(MultiFab& mf,int bottom,Real value,bool varying=false) {
    for(MFIter mfi(mf);mfi.isValid();++mfi) {
        const auto a=mf.array(mfi);
        ParallelFor(mfi.validbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            if(k==bottom)a(i,j,k)=value+(varying?Real(6*i+12*j):Real(0));
        });
    }
    Gpu::streamSynchronize();
}
template<class Expected> void expect_mass_values(const MultiFab& mf,Expected expected) {
    MultiFab error(mf.boxArray(),mf.DistributionMap(),1,1);
    for(MFIter mfi(mf);mfi.isValid();++mfi) {
        const auto a=mf.const_array(mfi);
        const auto e=error.array(mfi);const auto valid=mfi.validbox();
        ParallelFor(mfi.fabbox(),[=] AMREX_GPU_DEVICE(int i,int j,int k) noexcept {
            e(i,j,k)=a(i,j,k)-expected(i,j,k,valid.contains(IntVect(i,j,k)));
        });
    }
    EXPECT_FALSE(error.contains_nan(0,1,1));EXPECT_EQ(error.norm0(0,1),Real(0));
}
} // namespace

TEST(GroundMassLifecycle, ApportionsParentsAndRetainsOnlyOldValidFineCells)
{
    for(int rz:{1,2,3}) {
        const IntVect rr(2,3,rz);const Box cd(IntVect(-4,-3,3),IntVect(3,2,6));
        const auto cg=geometry(cd),fg=geometry(refine(cd,rr));
        auto coarse=field(BoxArray(cd),Real(-999));surface(coarse,3,Real(120),true);
        BoxArray oldba(refine(Box(IntVect(-4,-3,3),IntVect(-1,2,6)),rr));
        auto old=field(oldba,Real(-555));surface(old,3*rz,Real(99));
        auto saved=erf_ground_mass::capture({{"GroundSpecies_water",&old}},fg);
        BoxArray newba(refine(Box(IntVect(-2,-3,3),IntVect(1,2,6)),rr));
        newba.maxSize(IntVect(4,9,2*rz));
        auto current=field(newba,Real(-17));
        erf_ground_mass::restore({{"GroundSpecies_water",&current}},saved,fg,
            {{"GroundSpecies_water",&coarse}},&cg,rr);
        expect_mass_values(current,[=] AMREX_GPU_HOST_DEVICE(int i,int j,int k,bool valid) noexcept {
            if(!valid || k!=3*rz)return Real(-17);
            return i<0?Real(99):Real(20+amrex::coarsen(i,2)+2*amrex::coarsen(j,3));
        });
        // Creation has no retained history, even if destination storage was reused.
        erf_ground_mass::restore({{"GroundSpecies_water",&current}},{},fg,
            {{"GroundSpecies_water",&coarse}},&cg,rr);
        expect_mass_values(current,[=] AMREX_GPU_HOST_DEVICE(int i,int j,int k,bool valid) noexcept {
            if(!valid || k!=3*rz)return Real(-17);
            return Real(20+amrex::coarsen(i,2)+2*amrex::coarsen(j,3));
        });
    }
}

TEST(GroundMassLifecycle, ApportionmentAndRestrictionComposeAcrossThreeLevels)
{
    const IntVect r1(2,3,2),r2(3,2,3);
    const Box d0(IntVect(0,0,3),IntVect(1,1,4)),d1=refine(d0,r1),d2=refine(d1,r2);
    const auto g0=geometry(d0),g1=geometry(d1),g2=geometry(d2);
    auto m0=field(BoxArray(d0),Real(-1)),m1=field(BoxArray(d1),Real(-2)),m2=field(BoxArray(d2),Real(-3));
    surface(m0,3,Real(72));
    erf_ground_mass::restore({{"GroundAerosol_0_NaCl",&m1}},{},g1,{{"GroundAerosol_0_NaCl",&m0}},&g0,r1);
    erf_ground_mass::restore({{"GroundAerosol_0_NaCl",&m2}},{},g2,{{"GroundAerosol_0_NaCl",&m1}},&g1,r2);
    auto fine=erf_precipitation::capture_surface_accumulation(m2,g2);
    EXPECT_EQ(fine->sum(0),Real(288));
    erf_precipitation::sum_down_surface_mass(m2,m1,g2,g1,r2);
    erf_precipitation::sum_down_surface_mass(m1,m0,g1,g0,r1);
    expect_mass_values(m0,[] AMREX_GPU_HOST_DEVICE(int,int,int k,bool valid) noexcept {
        return valid && k==3?Real(72):Real(-1);
    });
}

TEST(GroundMassLifecycle, ElevatedPatchAndEmptyFieldSetLeaveValuesAlone)
{
    const IntVect rr(2,3,2);const Box cd(IntVect(0),IntVect(3,3,3));
    const auto cg=geometry(cd),fg=geometry(refine(cd,rr));
    auto coarse=field(BoxArray(cd),Real(24));
    auto high=field(BoxArray(refine(Box(IntVect(0,0,2),IntVect(3,3,3)),rr)),Real(9));
    erf_ground_mass::restore({{"GroundSpecies_water",&high}},{},fg,{{"GroundSpecies_water",&coarse}},&cg,rr);
    EXPECT_EQ(high.min(0,1),Real(9));EXPECT_EQ(high.max(0,1),Real(9));
    erf_ground_mass::restore({},{},fg);
    EXPECT_EQ(erf_ground_mass::manifest({}),"ERF ground mass v1\n0\n");
}

TEST(GroundMassLifecycle, ManifestRecordsNamesCategoriesDuplicatesAndOrder)
{
    auto mf=field(BoxArray(Box(IntVect(0),IntVect(0))),Real(0));
    GroundMassAccumulationFields fields={{"GroundSpecies_NaCl",&mf},
        {"GroundAerosol_0_NaCl",&mf},{"GroundAerosol_1_NaCl",&mf}};
    const auto original=erf_ground_mass::manifest(fields);
    EXPECT_EQ(original,"ERF ground mass v1\n3\nGroundSpecies_NaCl\nGroundAerosol_0_NaCl\nGroundAerosol_1_NaCl\n");
    std::swap(fields[1],fields[2]);EXPECT_NE(erf_ground_mass::manifest(fields),original);
    fields.pop_back();EXPECT_NE(erf_ground_mass::manifest(fields),original);
}

TEST(GroundMassLifecycle, LegacyEmptyRequiresExactNamesAndCompleteParticleHeader)
{
    const Vector<std::string> names={"xvel","yvel","zvel","species_mass_H2O","species_mass_ice"};
    const std::string header="Version_Two_Dot_One_double\n3\n5\nxvel\nyvel\nzvel\n"
        "species_mass_H2O\nspecies_mass_ice\n1\nphase\n1\n0\n1\n0\n1\n0 0 0\n";
    EXPECT_TRUE(erf_ground_mass::matches_empty_particle_header(header,names));
    const auto multi_level=header.substr(0,header.find("\nphase\n"))+"\nphase\n1\n7\n9\n1\n2\n1\n0 2 0\n1 3 8\n0 2 0\n";
    EXPECT_TRUE(erf_ground_mass::matches_empty_particle_header(multi_level,names));
    auto reordered=names;std::swap(reordered[0],reordered[1]);
    EXPECT_FALSE(erf_ground_mass::matches_empty_particle_header(header,reordered));
    auto substituted=names;substituted.back()="aerosol_mass_NaCl";
    EXPECT_FALSE(erf_ground_mass::matches_empty_particle_header(header,substituted));
    EXPECT_FALSE(erf_ground_mass::matches_empty_particle_header(header,{}));
    EXPECT_FALSE(erf_ground_mass::matches_empty_particle_header(header.substr(0,header.size()-5),names));
    EXPECT_FALSE(erf_ground_mass::matches_empty_particle_header(header.substr(0,header.find("species_mass_H2O")),names));
    EXPECT_FALSE(erf_ground_mass::matches_empty_particle_header(header+std::string(1,'\0')+"extra",names));
    auto wrong_count=header;wrong_count.replace(wrong_count.find("\n5\n"),3,"\n-1\n");
    EXPECT_FALSE(erf_ground_mass::matches_empty_particle_header(wrong_count,names));
    auto wrong_table=header;wrong_table.replace(wrong_table.find("0 0 0"),5,"0 1 0");
    EXPECT_FALSE(erf_ground_mass::matches_empty_particle_header(wrong_table,names));
}
