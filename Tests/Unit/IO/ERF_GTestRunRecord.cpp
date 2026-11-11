#include "ERF_RunRecord.H"
#include "ERF_Provenance.H"
#include "../ERF_GTestTempDir.H"

#include <gtest/gtest.h>

#include <csignal>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>

namespace
{

std::string read_file (const std::filesystem::path& path)
{
    std::ifstream in(path);
    std::stringstream text;
    text << in.rdbuf();
    return text.str();
}

bool contains (const std::string& text, const std::string& piece)
{
    return text.find(piece) != std::string::npos;
}

erf_provenance::ExecutionProvenance execution ()
{
    erf_provenance::ExecutionProvenance e;
    e.execution_uuid = "11111111-1111-4111-8111-111111111111";
    e.simulation_uuid = "22222222-2222-4222-8222-222222222222";
    return e;
}

struct Scratch
{
    std::filesystem::path dir = erf_gtest_temp_path("erf_run_record");
    Scratch () { std::filesystem::create_directories(dir); }
    ~Scratch () { std::error_code ignored; std::filesystem::remove_all(dir, ignored); }
    [[nodiscard]] std::filesystem::path record () const { return dir / "erf_run.json"; }
};

TEST(RunRecord, RunningThenCheckpointThenCompleted)
{
    Scratch scratch;
    erf_run_record::RunRecord record;
    record.set_path(scratch.record().string());
    char exe[] = "erf_exec", inputs[] = "inputs", override_[] = "max_step=4";
    char* argv[] = {exe, inputs, override_};
    record.set_command_line(3, argv);

    record.begin(execution());
    std::string text = read_file(scratch.record());
    EXPECT_TRUE(contains(text, "\"status\":\"running\"")) << text;
    EXPECT_TRUE(contains(text, "\"reason\":null")) << text;
    EXPECT_TRUE(contains(text, "\"inputs\":\"inputs\"")) << text;
    EXPECT_TRUE(contains(text, "\"arguments\":[\"max_step=4\"]")) << text;
    EXPECT_TRUE(contains(text, "\"execution_uuid\":\"11111111-1111-4111-8111-111111111111\"")) << text;
    EXPECT_TRUE(contains(text, "\"step\":null")) << text;
    EXPECT_TRUE(contains(text, "\"time\":null")) << text;
    EXPECT_TRUE(contains(text, "\"last_checkpoint\":null")) << text;
    EXPECT_TRUE(contains(text, "\"exit_code\":null")) << text;
    EXPECT_FALSE(record.terminal());

    record.initialized(execution(), 0, 0.0);
    text = read_file(scratch.record());
    EXPECT_TRUE(contains(text, "\"status\":\"running\"")) << text;
    EXPECT_TRUE(contains(text, "\"step\":0")) << text;
    EXPECT_TRUE(contains(text, "\"time\":0")) << text;

    record.checkpoint_written("chk00002", 2, 0.5);
    text = read_file(scratch.record());
    EXPECT_TRUE(contains(text, "\"status\":\"running\"")) << text;
    EXPECT_TRUE(contains(text, "\"last_checkpoint\":\"chk00002\"")) << text;
    EXPECT_TRUE(contains(text, "\"step\":2")) << text;
    EXPECT_TRUE(contains(text, "\"time\":0.5")) << text;

    record.finish("max_step", 4, 1.0);
    text = read_file(scratch.record());
    EXPECT_TRUE(contains(text, "\"status\":\"completed\"")) << text;
    EXPECT_TRUE(contains(text, "\"reason\":\"max_step\"")) << text;
    EXPECT_TRUE(contains(text, "\"step\":4")) << text;
    EXPECT_TRUE(contains(text, "\"exit_code\":0")) << text;
    EXPECT_TRUE(record.terminal());

    // A later exit() leaves a completed record alone.
    record.exited();
    EXPECT_EQ(read_file(scratch.record()), text);

    // The partial file never remains.
    for (const auto& entry : std::filesystem::directory_iterator(scratch.dir)) {
        EXPECT_EQ(entry.path().filename().string(), "erf_run.json");
    }
}

TEST(RunRecord, AbortBeforeInitializationIsRecorded)
{
    Scratch scratch;
    erf_run_record::RunRecord record;
    record.set_path(scratch.record().string());
    char exe[] = "erf_exec", inputs[] = "inputs";
    char* argv[] = {exe, inputs};
    record.set_command_line(2, argv);

    record.aborted("Assertion `cfl > zero' failed");
    const std::string text = read_file(scratch.record());
    EXPECT_TRUE(contains(text, "\"status\":\"aborted\"")) << text;
    EXPECT_TRUE(contains(text, "\"reason\":\"abort\"")) << text;
    EXPECT_TRUE(contains(text, "\"message\":\"Assertion `cfl > zero' failed\"")) << text;
    EXPECT_TRUE(contains(text, "\"execution_uuid\":null")) << text;
    EXPECT_TRUE(contains(text, "\"step\":null")) << text;
    EXPECT_TRUE(record.terminal());

    // Only the first failure is the cause.
    record.aborted("second failure");
    EXPECT_FALSE(contains(read_file(scratch.record()), "second failure"));
}

TEST(RunRecord, AbortAfterCompletionIsRecorded)
{
    Scratch scratch;
    erf_run_record::RunRecord record;
    record.set_path(scratch.record().string());
    record.begin(execution());
    record.initialized(execution(), 0, 0.0);
    record.finish("max_step", 4, 1.0);

    // amrex.abort_on_unused_inputs fires inside amrex::Finalize, after the time loop.
    record.aborted("ERROR: unused ParmParse variables");
    const std::string text = read_file(scratch.record());
    EXPECT_TRUE(contains(text, "\"status\":\"aborted\"")) << text;
    EXPECT_TRUE(contains(text, "\"reason\":\"abort\"")) << text;
    EXPECT_TRUE(contains(text, "\"message\":\"ERROR: unused ParmParse variables\"")) << text;
    EXPECT_TRUE(contains(text, "\"step\":4")) << text;
    EXPECT_TRUE(contains(text, "\"exit_code\":null")) << text;
}

TEST(RunRecord, SignalIsRecorded)
{
    Scratch scratch;
    erf_run_record::RunRecord record;
    record.set_path(scratch.record().string());
    record.begin(execution());
    record.initialized(execution(), 7, 1.75);

    record.signaled(SIGTERM);
    const std::string text = read_file(scratch.record());
    EXPECT_TRUE(contains(text, "\"status\":\"aborted\"")) << text;
    EXPECT_TRUE(contains(text, "\"reason\":\"signal\"")) << text;
    EXPECT_TRUE(contains(text, "\"message\":\"SIGTERM (")) << text;
    EXPECT_TRUE(contains(text, "\"step\":7")) << text;
    EXPECT_TRUE(contains(text, "\"exit_code\":null")) << text;
    EXPECT_TRUE(record.terminal());
}

TEST(RunRecord, SignalDoesNotHideAnotherRanksAbort)
{
    // Two ranks share the file; the one that aborted wrote the cause, and the rank the
    // launcher then kills must keep it.
    Scratch scratch;
    erf_run_record::RunRecord aborting_rank, killed_rank;
    for (auto* rank : {&aborting_rank, &killed_rank}) {
        rank->set_path(scratch.record().string());
        rank->begin(execution());
        rank->initialized(execution(), 7, 1.75);
    }
    aborting_rank.aborted("NaN detected in state");
    const std::string text = read_file(scratch.record());
    killed_rank.signaled(SIGTERM);
    EXPECT_EQ(read_file(scratch.record()), text);
    EXPECT_TRUE(contains(text, "\"message\":\"NaN detected in state\"")) << text;
}

TEST(RunRecord, ExitBeforeTheEndIsRecorded)
{
    Scratch scratch;
    erf_run_record::RunRecord record;
    record.set_path(scratch.record().string());
    record.begin(execution());
    record.initialized(execution(), 3, 0.75);
    record.exited();
    const std::string text = read_file(scratch.record());
    EXPECT_TRUE(contains(text, "\"status\":\"exited\"")) << text;
    EXPECT_TRUE(contains(text, "\"reason\":\"exit\"")) << text;
    EXPECT_TRUE(contains(text, "\"step\":3")) << text;
    EXPECT_TRUE(contains(text, "\"exit_code\":1")) << text;
}

TEST(RunRecord, NoneDisablesTheRecord)
{
    Scratch scratch;
    erf_run_record::RunRecord record;
    record.set_path("None");
    record.begin(execution());
    record.initialized(execution(), 0, 0.0);
    record.finish("max_step", 1, 0.1);
    record.signaled(SIGTERM);
    EXPECT_TRUE(std::filesystem::is_empty(scratch.dir));
}

} // namespace
