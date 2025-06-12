#pragma once

#include <string>

#define SLOG_TRACE (1)
#define SLOG_DEBUG (2)
#define SLOG_INFO  (3)
#define SLOG_WARN  (4)
#define SLOG_ERROR (5)

class Solver;

typedef void (*solve_fatal_cb_t   )(Solver*, void* arg, std::string message);
typedef void (*solve_log_cb_t     )(Solver*, void* arg, int level, std::string message);
typedef void (*solve_progress_cb_t)(Solver*, void* arg, int step, int n, int nmax, std::string message);

class __declspec(dllexport)  Solver
{
public:
	void setFatalCB(solve_fatal_cb_t cb, void* arg) { fatalCB = cb; fatalArg = arg; }
	void setLogCB(solve_log_cb_t cb, void* arg) { logCB = cb; logArg = arg; }
	void setProgressCB(solve_progress_cb_t cb, void* arg) { progressCB = cb; progressArg = arg; }
	void setLogLevel(int level) { logLevel = level; }

protected:

	void fatal(std::string message) { log(SLOG_ERROR, message); if (fatalCB) fatalCB(this, fatalArg, message); }
	void log(int level, std::string message) { if (logCB) logCB(this, logArg, level, message); }
	void progress(int step, int n, int nmax, std::string message) { if (progressCB) progressCB(this, progressArg, step, n, nmax, message); }
	void logf(int level, const std::string fmt_str, ...);
private:
	int numThreads;
	int logLevel;
	solve_fatal_cb_t fatalCB = nullptr;
	void* fatalArg;
	solve_log_cb_t logCB = nullptr;
	void* logArg;
	solve_progress_cb_t progressCB = nullptr;
	void* progressArg;

};

