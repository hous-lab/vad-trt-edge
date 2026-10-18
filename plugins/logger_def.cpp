#include <cstdio>
#include <cstdlib>
#include "NvInfer.h"
#include "NvInferRuntimePlugin.h"

class LogErrorLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kERROR) {
            fprintf(stderr, "[TRT Plugin] %s\n", msg);
        }
    }
};

static LogErrorLogger s_logger;
namespace nvinfer1 { namespace plugin { 
ILogger* gLogger = &s_logger;
ILogger& gLogError = s_logger;

void reportAssertion(const char* msg, const char* file, int line) noexcept {
    fprintf(stderr, "[TRT Plugin Assertion] %s at %s:%d\n", msg, file, line);
    std::abort();
}

void throwPluginError(char const* file, char const* function, int line, int, char const* msg) noexcept {
    fprintf(stderr, "[TRT Plugin Error] %s:%d (%s): %s\n", file, line, function, msg);
    std::abort();
}
}}
