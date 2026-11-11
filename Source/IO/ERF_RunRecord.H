/**
 * \file ERF_RunRecord.H
 *
 * One JSON file per run, erf_run.json in the run directory by default (erf.run_record
 * names it; none disables it), that says how the run is going and how it ended:
 * status running, completed, aborted or exited; the reason; the last completed step,
 * time and checkpoint. A record left at "running" means the process was killed
 * outright (SIGKILL, an out-of-memory kill, a node failure) before it could say why.
 */
#ifndef ERF_RUN_RECORD_H_
#define ERF_RUN_RECORD_H_

#include <string>
#include <vector>

namespace erf_provenance { struct ExecutionProvenance; }

namespace erf_run_record {

class RunRecord
{
public:
    // Either call arms the record: main passes the command line, a test sets the path. An
    // ERF object built without main, as in the unit tests, writes nothing. The path "none"
    // (any case) or an empty path disables the record.
    void set_path (const std::string& path);
    void set_command_line (int argc, char* argv[]);
    // The run exists: status running, with no step or time until initialization is done.
    void begin (const erf_provenance::ExecutionProvenance& execution);
    // Initialization finished at this step and time; a restart may have changed the identity.
    void initialized (const erf_provenance::ExecutionProvenance& execution, int step, double time);
    void checkpoint_written (const std::string& directory, int step, double time);
    void finish (const char* reason, int step, double time);
    // amrex::Abort on the calling rank; the caller then aborts. Replaces a completed
    // record too: a run that aborts while finalizing did not end well.
    void aborted (const char* message);
    // A fatal signal on the calling rank, the scheduler's SIGTERM or a crash; the caller
    // then lets AMReX print the backtrace and abort. Leaves a record that another rank
    // already marked aborted alone, so the ranks killed afterwards do not hide the cause.
    void signaled (int signal);
    // An exit() before amrex::Finalize; does nothing once the record is terminal.
    void exited ();

    [[nodiscard]] bool terminal () const { return m_status != "running"; }

private:
    std::string m_path = "erf_run.json";
    std::string m_status = "running", m_reason, m_message;
    std::string m_inputs, m_started, m_execution_uuid, m_simulation_uuid, m_last_checkpoint;
    std::vector<std::string> m_arguments;
    int m_step = -1, m_exit_code = -1;
    double m_time = 0.0;
    bool m_begun = false, m_path_set = false;

    void resolve_path ();
    void write () const;
    [[nodiscard]] bool file_says_aborted () const;
};

// The process-wide record, used by main and the ERF time loop.
void set_command_line (int argc, char* argv[]);
void begin (const erf_provenance::ExecutionProvenance& execution);
void initialized (const erf_provenance::ExecutionProvenance& execution, int step, double time);
void checkpoint_written (const std::string& directory, int step, double time);
void finish (const char* reason, int step, double time);
void abort (const char* message);   // AMReX error handler
void install_signal_handlers ();    // after amrex::Initialize has installed AMReX's
void exited ();                     // exit trap

}

#endif
