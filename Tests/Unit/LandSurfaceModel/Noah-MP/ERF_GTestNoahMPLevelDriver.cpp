#include <gtest/gtest.h>

#include "LandSurfaceModel/Noah-MP/ERF_NOAHMP_LevelDriver.H"

// Which AMR levels run the Noah-MP driver on a land setup file of their own, and which
// take their land state from level 0 (noahmp_level_runs_driver).

TEST(NoahMPLevelDriver, LevelZeroAlwaysRunsTheDriver)
{
    for (const bool file : {false, true}) {
        for (const bool idealized : {false, true}) {
            for (const bool named : {false, true}) {
                EXPECT_TRUE(noahmp_level_runs_driver(0, file, idealized, named));
            }
        }
    }
}

TEST(NoahMPLevelDriver, InitFileRunsTheDriverOnAFinerLevel)
{
    // erf.nc_init_file_<lev>, the WRF/metgrid way, whatever the namelist says.
    for (const int lev : {1, 2}) {
        for (const bool idealized : {false, true}) {
            for (const bool named : {false, true}) {
                EXPECT_TRUE(noahmp_level_runs_driver(lev, true, idealized, named));
            }
        }
    }
}

TEST(NoahMPLevelDriver, IdealizedRunFollowsTheNamelist)
{
    // An idealized run has no init file: ERF_SETUP_FILE_0<lev+1> decides.
    for (const int lev : {1, 2}) {
        EXPECT_TRUE(noahmp_level_runs_driver(lev, false, true, true));
        EXPECT_FALSE(noahmp_level_runs_driver(lev, false, true, false));
    }
}

TEST(NoahMPLevelDriver, WrfRunIgnoresTheNamelistAlone)
{
    // A run from WRF or metgrid files without erf.nc_init_file_<lev> keeps taking the
    // level's land state from level 0, even if namelist.erf names a setup file: such runs
    // behave as they did before idealized runs could use one.
    for (const int lev : {1, 2}) {
        EXPECT_FALSE(noahmp_level_runs_driver(lev, false, false, true));
        EXPECT_FALSE(noahmp_level_runs_driver(lev, false, false, false));
    }
}

// The policy is independent of the external library; keep these tests enabled
// in CPU builds without NoahMP so unsupported AMR transitions stay covered.
TEST(NoahMPLevelDriver, OwnDriverCreationRequiresTimeZero)
{
    EXPECT_TRUE(noahmp_new_level_is_supported(true, 0.0));
    for (const double time : {0.01, 1.0, 7200.0}) {
        EXPECT_FALSE(noahmp_new_level_is_supported(true, time));
    }
}

TEST(NoahMPLevelDriver, FilelessLevelsCanAppearAtNonzeroTime)
{
    for (const double time : {0.0, 0.01, 1.0, 7200.0}) {
        EXPECT_TRUE(noahmp_new_level_is_supported(false, time));
    }
}

TEST(NoahMPLevelDriver, OnlyFilelessLevelsCanBeRemadeOrRemoved)
{
    EXPECT_FALSE(noahmp_level_rebuild_or_removal_is_supported(true));
    EXPECT_TRUE(noahmp_level_rebuild_or_removal_is_supported(false));
}
