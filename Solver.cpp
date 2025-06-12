#include "pch.h"
#include "Solver.h"
#include <varargs.h>
#include <stdio.h>

void Solver::logf(int level, const std::string fmt_str, ...)

{
	va_list ap;
	char* fp = NULL;
	std::string strBuf;
	strBuf.resize(256);
	va_start(ap, fmt_str);
	sprintf_s((char*)strBuf.c_str(), strBuf.size(), fmt_str.c_str(), ap);
	va_end(ap);
	log(level, strBuf);
}
