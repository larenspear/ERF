#include <ERF_RunRecord.H>
#include <ERF_DescriptionWriter.H>
#include <ERF_Provenance.H>
#include <AMReX.H>
#include <AMReX_ParallelDescriptor.H>
#include <AMReX_ParmParse.H>
#include <AMReX_buildInfo.H>

#include <algorithm>
#include <cctype>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace erf_run_record {

namespace {

std::string signal_name (int signal)
{
    switch (signal) {
    case SIGTERM: return "SIGTERM";
    case SIGINT:  return "SIGINT";
    case SIGSEGV: return "SIGSEGV";
    case SIGFPE:  return "SIGFPE";
    case SIGILL:  return "SIGILL";
    case SIGABRT: return "SIGABRT";
#ifdef SIGBUS
    case SIGBUS:  return "SIGBUS";
#endif
    default:      return "signal";
    }
}

bool means_none (const std::string& path)
{
    std::string lower = path;
    std::transform(lower.begin(), lower.end(), lower.begin(),
                   [] (unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return lower == "none";
}

} // namespace

void RunRecord::set_path (const std::string& path)
{
    m_path = means_none(path) ? std::string() : path;
    m_path_set = true;
}

void RunRecord::set_command_line (int argc, char* argv[])
{
    if (argc > 1 && std::string(argv[1]).find('=') == std::string::npos) m_inputs = argv[1];
    for (int i = m_inputs.empty() ? 1 : 2; i < argc; ++i) m_arguments.emplace_back(argv[i]);
    resolve_path();
}

void RunRecord::resolve_path ()
{
    if (m_path_set) return;
    amrex::ParmParse pp("erf");
    std::string path = m_path;
    pp.query("run_record", path);
    set_path(path);
}

void RunRecord::begin (const erf_provenance::ExecutionProvenance& execution)
{
    if (!m_path_set) return;   // armed by main (or a test); an ERF built elsewhere writes nothing
    m_execution_uuid = execution.execution_uuid;
    m_simulation_uuid = execution.simulation_uuid;
    m_started = erf_provenance::current_utc();
    m_begun = true;
    if (amrex::ParallelDescriptor::IOProcessor()) write();
}

void RunRecord::initialized (const erf_provenance::ExecutionProvenance& execution, int step, double time)
{
    m_execution_uuid = execution.execution_uuid;
    m_simulation_uuid = execution.simulation_uuid;
    m_step = step;
    m_time = time;
    if (m_begun && amrex::ParallelDescriptor::IOProcessor()) write();
}

void RunRecord::checkpoint_written (const std::string& directory, int step, double time)
{
    m_last_checkpoint = directory;
    m_step = step;
    m_time = time;
    if (m_begun && amrex::ParallelDescriptor::IOProcessor()) write();
}

void RunRecord::finish (const char* reason, int step, double time)
{
    m_status = "completed";
    m_reason = reason;
    m_step = step;
    m_time = time;
    m_exit_code = 0;
    if (m_begun && amrex::ParallelDescriptor::IOProcessor()) write();
}

void RunRecord::aborted (const char* message)
{
    if (m_status == "aborted") return;   // the first failure is the cause
    resolve_path();
    m_status = "aborted";
    m_reason = "abort";
    m_message = message ? message : "";
    m_exit_code = -1;
    write();
}

void RunRecord::signaled (int signal)
{
    if (terminal()) return;
    resolve_path();
    if (file_says_aborted()) return;
    m_status = "aborted";
    m_reason = "signal";
    m_message = signal_name(signal) + " (" + std::to_string(signal) + ")";
    m_exit_code = -1;
    write();
}

void RunRecord::exited ()
{
    if (terminal() || !m_begun) return;
    m_status = "exited";
    m_reason = "exit";
    m_exit_code = EXIT_FAILURE;
    write();
}

bool RunRecord::file_says_aborted () const
{
    if (m_path.empty()) return false;
    std::ifstream in(m_path);
    const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    return text.find("\"status\":\"aborted\"") != std::string::npos;
}

void RunRecord::write () const
{
    if (m_path.empty()) return;
    std::ostringstream text;
    erf_description::Writer writer(text, erf_description::Format::JSON);
    writer.value("schema_version", 1);
    writer.value("application", "ERF");
    writer.value("status", m_status.c_str());
    writer.optional_string("reason", m_reason.c_str());
    writer.optional_string("message", m_message.c_str());
    writer.optional_string("execution_uuid", m_execution_uuid.c_str());
    writer.optional_string("simulation_uuid", m_simulation_uuid.c_str());
    writer.optional_string("inputs", m_inputs.c_str());
    writer.strings("arguments", m_arguments);
    writer.optional_string("started_utc", m_started.c_str());
    writer.value("updated_utc", erf_provenance::current_utc().c_str());
    writer.value("ranks", amrex::ParallelDescriptor::NProcs());
    writer.value("writer_rank", amrex::ParallelDescriptor::MyProc());
    if (m_step >= 0) writer.value("step", m_step); else writer.value("step", nullptr);
    if (m_step >= 0) writer.value("time", m_time); else writer.value("time", nullptr);
    writer.optional_string("last_checkpoint", m_last_checkpoint.c_str());
    writer.optional_string("source_describe", amrex::buildInfoGetGitHash(1));
    if (m_exit_code >= 0) writer.value("exit_code", m_exit_code); else writer.value("exit_code", nullptr);
    writer.finish();

    // Write beside the record and rename over it, so a reader never sees a partial file.
    const std::string partial = m_path + ".tmp." + std::to_string(amrex::ParallelDescriptor::MyProc());
    {
        std::ofstream out(partial, std::ios::trunc);
        if (!out) {
            static bool warned = false;
            if (!warned) {
                warned = true;
                amrex::ErrorStream() << "ERF: cannot write the run record " << m_path
                                     << "; does its directory exist?\n";
            }
            return;
        }
        out << text.str();
    }
    std::error_code error;
    std::filesystem::rename(partial, m_path, error);
    if (error) std::filesystem::remove(partial, error);
}

namespace {

RunRecord& record ()
{
    static RunRecord the_record;
    return the_record;
}

using Handler = void (*)(int);

constexpr int handled_signals[] = {SIGTERM, SIGINT, SIGSEGV, SIGFPE, SIGILL, SIGABRT
#ifdef SIGBUS
                                   , SIGBUS
#endif
};
constexpr int n_handled = sizeof(handled_signals) / sizeof(handled_signals[0]);
Handler previous_handlers[n_handled] = {};

void on_signal (int signal)
{
    // A second signal while recording the first must still end the process.
    static volatile std::sig_atomic_t recording = 0;
    if (!recording) {
        recording = 1;
        record().signaled(signal);
    }
    Handler previous = nullptr;
    for (int i = 0; i < n_handled; ++i) {
        if (handled_signals[i] == signal) previous = previous_handlers[i];
    }
    if (previous != nullptr && previous != SIG_DFL && previous != SIG_IGN) {
        previous(signal);   // AMReX's handler: the backtrace, then the abort
    } else {
        std::signal(signal, SIG_DFL);
        std::raise(signal);
    }
}

} // namespace

void set_command_line (int argc, char* argv[]) { record().set_command_line(argc, argv); }

void begin (const erf_provenance::ExecutionProvenance& execution) { record().begin(execution); }

void initialized (const erf_provenance::ExecutionProvenance& execution, int step, double time)
{
    record().initialized(execution, step, time);
}

void checkpoint_written (const std::string& directory, int step, double time)
{
    record().checkpoint_written(directory, step, time);
}

void finish (const char* reason, int step, double time) { record().finish(reason, step, time); }

void abort (const char* message)
{
    // A second failure while recording the first must still end the run.
    static bool recording = false;
    if (!recording) {
        recording = true;
        record().aborted(message);
    }
    std::fputs("amrex::Abort::", stderr);
    amrex::write_to_stderr_without_buffering(message);
    if (amrex::system::throw_exception) throw amrex::RuntimeError(message ? message : "");
    amrex::ParallelDescriptor::Abort();
}

void install_signal_handlers ()
{
    for (int i = 0; i < n_handled; ++i) {
        const Handler previous = std::signal(handled_signals[i], on_signal);
        previous_handlers[i] = (previous == SIG_ERR) ? nullptr : previous;
    }
}

void exited () { record().exited(); }

}
