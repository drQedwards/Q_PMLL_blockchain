/*
 * pypm_init_main_demo_utils.cpp
 *
 * A single-file C++ translation of the my_lib Python demo.
 * It embeds the CPython interpreter, gathers the same runtime fingerprint
 * (Python version, platform, requests & torch versions) and prints it as JSON.
 *
 * Build (Linux):
 *   g++ -std=c++17 -I$(python3 -m sysconfig --includes) \
 *       pypm_init_main_demo_utils.cpp \
 *       -o pypm_fingerprint \
 *       $(python3-config --ldflags)
 *
 * Run:
 *   ./pypm_fingerprint
 *
 * Author: Dr. Josef Kurk Edwards & ChatGPT (2025)
 * License: MIT
 */
#include <Python.h>
#include <iostream>

int main() {
    // 1. Boot the embedded interpreter.
    Py_Initialize();

    // 2. Python script that reproduces my_lib.utils.runtime_fingerprint().
    const char *script = R"PY(
import platform, json
import importlib.metadata as im

fp = {
    "python": platform.python_version(),
    "platform": f"{platform.system()}-{platform.machine()}",
}

for pkg in ("requests", "torch"):
    try:
        fp[pkg] = im.version(pkg)
    except im.PackageNotFoundError:
        fp[pkg] = "not-installed"

print("🔍 pypm embedded runtime fingerprint")
print(json.dumps(fp, indent=2))
)PY";

    // 3. Execute the script.
    if (PyRun_SimpleString(script) != 0) {
        std::cerr << "Error: embedded Python script failed." << std::endl;
        Py_Finalize();
        return 1;
    }

    // 4. Clean shutdown.
    Py_Finalize();
    return 0;
}
