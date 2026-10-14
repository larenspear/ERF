#!/usr/bin/env python3
"""Portability check for code that must build with MSVC on 64-bit Windows.

ERF is built and tested on Windows in CI, but only the configurations the Windows
job enables. Code behind NetCDF, Fortran bridges or wind farms is not compiled
there, and the compiler cannot catch what it does not compile. This check looks,
in Source/, Tests/ and Exec/, for the classes of mistake that have bitten on
Windows:

  long          32 bits on 64-bit Windows (LLP64). A 'long' that is meant to hold
                a 64-bit value truncates silently; a 'long' handed to an AMReX API
                taking amrex::Long does not compile. Use amrex::Long, std::int64_t
                or size_t. (Only 'long'; 'long long' and 'long double' are fine.)
  posix-header  <unistd.h>, <dirent.h>, <sys/time.h>, ... have no MSVC equivalent.
                Guard them with #ifndef _WIN32 or use the C++ standard library.
  posix-call    gettimeofday, gmtime_r, setenv, popen, ... likewise.
  printf-long   %ld / %lu assume a 64-bit long; with size_t or amrex::Long
                arguments they are wrong on Windows. Use %zu, PRId64 or streams.

Each class has a baseline count below. New occurrences fail the check; when you
fix some, lower the baseline in the same commit. A file that mentions _WIN32 or
_MSC_VER is taken to have guarded its POSIX uses and is skipped for those two
classes only. Run with --list to see every occurrence, not just the excess.
"""
import re, subprocess, sys

# Counts on development as of 2026-10-13, after the first round of fixes. Each is a
# ceiling: new occurrences fail, fixing some lets you lower it.
BASELINE = {"long": 68, "posix-header": 1, "posix-call": 0, "printf-long": 12}

CLASSES = {
    "long": re.compile(
        r"(?<![\w:])(unsigned\s+long|long\s+unsigned(\s+int)?|long(\s+int)?)\b"
        r"(?!\s+(long|double)\b)(?=\s*[&*]?\s*[A-Za-z_(>]|\s*>)"),
    "posix-header": re.compile(
        r"#include\s*<(unistd\.h|dirent\.h|sys/time\.h|sys/stat\.h|sys/resource\.h"
        r"|sys/wait\.h|pwd\.h|signal\.h|fcntl\.h)>"),
    "posix-call": re.compile(
        r"\b(gettimeofday|getpid|localtime_r|gmtime_r|ctime_r|strtok_r|popen|pclose"
        r"|usleep|fork|drand48|srand48|setenv|unsetenv|getcwd|realpath|alloca|mkstemp)\s*\("),
    "printf-long": re.compile(r"%\d*(ld|lu)\b"),
}
GUARDED_CLASSES = {"posix-header", "posix-call"}


def main():
    want_list = "--list" in sys.argv
    files = [f for f in subprocess.run(["git", "ls-files", "Source", "Tests", "Exec"],
                                       capture_output=True, text=True, check=True).stdout.split()
             if f.endswith((".H", ".h", ".cpp", ".hpp"))]
    hits = {c: [] for c in CLASSES}
    for f in files:
        text = open(f, encoding="utf-8", errors="replace").read()
        guarded = "_WIN32" in text or "_MSC_VER" in text
        text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
        for lineno, line in enumerate(text.split("\n"), 1):
            nocomment = line.split("//")[0]
            code = re.sub(r'"(\\.|[^"\\])*"', '""', nocomment)  # string literals removed
            for cls, rx in CLASSES.items():
                if cls in GUARDED_CLASSES and guarded:
                    continue
                # printf formats live inside the literals; everything else must not match them
                if rx.search(nocomment if cls == "printf-long" else code):
                    hits[cls].append(f"{f}:{lineno}: {line.strip()[:110]}")
    failed = False
    for cls in CLASSES:
        n, base = len(hits[cls]), BASELINE[cls]
        status = "ok" if n <= base else "FAIL"
        print(f"{status:4s} {cls:13s} {n:4d} (baseline {base})")
        if n > base or want_list:
            for h in hits[cls]:
                print("     " + h)
        if n > base:
            failed = True
    if failed:
        print("\nNew occurrences of a portability class listed above. Either use a 64-bit or"
              " standard type / a guard, or, if the use is deliberate, raise the baseline in"
              " .github/workflows/style/check_portability.py in the same commit and say why.")
        return 1
    print("\nTo lower a baseline after fixing occurrences, edit BASELINE in this file.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
