/* pypm.c  —  v0.0.2
 *
 * Build:   cc -Wall -Wextra -ldl -lcurl -o pypm pypm.c
 * Runtime: ./pypm <verb> [options]
 *
 * Notes:
 *   • CLI parsing = getopt_long (simple, in-libc, minimal deps)
 *   • Plugins     = POSIX dlopen() with enhanced error checking
 *   • Workspace   = upward search for pypm-workspace.toml + env override
 *   • Sandbox     = mkdtemp(3) or custom dir + shell spawn
 *   • Hermetic    = tar-based placeholder (extensible for real use)
 *
 * Goals: Robustness, modularity, and user-friendliness.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>
#include <curl/curl.h>

#define PYP_VERSION "0.0.2"
#define MAX_PATH 4096

/* -------------------- Helpers -------------------- */

/** Print error message and exit with failure. */
static void fatal(const char *msg) {
    perror(msg);
    exit(EXIT_FAILURE);
}

/** Check if a file exists at the given path. */
static int file_exists(const char *path) {
    struct stat st;
    return stat(path, &st) == 0;
}

/* -------------------- Workspace -------------------- */

/**
 * Find the workspace root by searching for 'pypm-workspace.toml'.
 * Honors PYP_WORKSPACE_ROOT env var as an override.
 * Returns a static buffer or NULL if not found.
 */
static char *find_workspace_root(void) {
    static char cwd[MAX_PATH];
    const char *env_root = getenv("PYP_WORKSPACE_ROOT");
    if (env_root) {
        snprintf(cwd, sizeof(cwd), "%s", env_root);
        char probe[MAX_PATH];
        snprintf(probe, sizeof(probe), "%s/pypm-workspace.toml", cwd);
        return file_exists(probe) ? cwd : NULL;
    }

    if (!getcwd(cwd, sizeof(cwd))) fatal("getcwd");
    char probe[MAX_PATH];
    char *s = cwd;
    while (s && *s) {
        snprintf(probe, sizeof(probe), "%s/pypm-workspace.toml", s);
        if (file_exists(probe)) return s;
        s = strrchr(s, '/');
        if (s) *s = '\0';
    }
    return NULL;
}

/* -------------------- Doctor -------------------- */

/** Diagnose the build environment and report issues. */
static int run_doctor(void) {
    puts("🔍  pypm doctor — beginning diagnostics");
    int issues = 0;

    if (!system("python3 -c 'import sysconfig, sys; "
                "exit(0 if sysconfig.get_config_var(\"INCLUDEPY\") else 1)'")) {
        puts("✅  Python dev headers found");
    } else {
        puts("❌  Missing python<version>-dev / -headers package");
        issues++;
    }

    if (!system("cc --version > /dev/null 2>&1")) {
        puts("✅  C compiler available (cc)");
    } else {
        puts("❌  No C compiler in PATH");
        issues++;
    }

    printf("🏁  Diagnostics complete (%d issues found)\n", issues);
    return issues ? 1 : 0;
}

/* -------------------- Sandbox -------------------- */

/**
 * Spawn a shell in an isolated sandbox directory.
 * Uses a custom dir if provided, else creates a temp one.
 */
static int run_sandbox(const char *custom_dir) {
    char template[] = "/tmp/pypm-sandbox-XXXXXX";
    char *dir = custom_dir ? (char *)custom_dir : mkdtemp(template);
    if (!dir) fatal("mkdtemp or invalid custom directory");

    printf("🐚  Spawning ephemeral shell in %s\n", dir);
    if (chdir(dir) != 0) fatal("chdir");

    char *shell = getenv("SHELL") ?: "/bin/bash";
    execvp(shell, (char *[]){shell, "-l", NULL});
    fatal("execvp"); /* Only reached if exec fails */
    return 0;
}

/* -------------------- Plugin Subsystem -------------------- */

typedef int (*plugin_main_f)(int, char **);

/** Load and execute a plugin by name, passing args. */
static int load_and_run_plugin(const char *name, int argc, char **argv) {
    char so_path[MAX_PATH];
    snprintf(so_path, sizeof(so_path), "%s/.pypm/plugins/%s.so",
             getenv("HOME") ? getenv("HOME") : ".", name);

    void *h = dlopen(so_path, RTLD_LAZY);
    if (!h) {
        fprintf(stderr, "Plugin load error: %s\n", dlerror());
        return 1;
    }

    plugin_main_f entry = dlsym(h, "pypm_plugin_main");
    if (!entry) {
        fprintf(stderr, "Bad plugin (no pypm_plugin_main): %s\n", name);
        dlclose(h);
        return 1;
    }

    int rc = entry(argc, argv);
    dlclose(h);
    return rc;
}

/** Install a plugin from a source URL or path. */
static int plugin_cmd_add(const char *name, const char *src) {
    printf("🔌  Installing plugin %s from %s …\n", name, src);
    char plugin_dir[MAX_PATH];
    snprintf(plugin_dir, sizeof(plugin_dir), "%s/.pypm/plugins",
             getenv("HOME") ? getenv("HOME") : ".");
    if (!file_exists(plugin_dir) && mkdir(plugin_dir, 0755) != 0)
        fatal("mkdir plugins dir");

    char dst[MAX_PATH];
    snprintf(dst, sizeof(dst), "%s/%s.so", plugin_dir, name);

    CURL *curl = curl_easy_init();
    if (!curl) fatal("curl init");
    FILE *out = fopen(dst, "wb");
    if (!out) {
        curl_easy_cleanup(curl);
        fatal("fopen plugin destination");
    }

    curl_easy_setopt(curl, CURLOPT_URL, src);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, out);
    CURLcode res = curl_easy_perform(curl);
    curl_easy_cleanup(curl);
    fclose(out);

    if (res != CURLE_OK) {
        unlink(dst);
        fprintf(stderr, "Download failed: %s\n", curl_easy_strerror(res));
        return 1;
    }

    puts("✅  Plugin installed");
    return 0;
}

/* -------------------- Hermetic (“pypylock”) -------------------- */

/** Create a hermetic bundle (placeholder implementation). */
static int run_pypylock(const char *output) {
    printf("📦  Creating hermetic bundle %s …\n", output);
    char cmd[MAX_PATH + 50];
    snprintf(cmd, sizeof(cmd), "tar czf %s .venv", output);
    if (!system(cmd)) {
        puts("✅  pypylock bundle created");
        return 0;
    }
    puts("❌  Bundle creation failed");
    return 1;
}

/* -------------------- CLI Dispatch -------------------- */

/** Display usage information. */
static void usage(void) {
    puts("pypm " PYP_VERSION
         "\nUSAGE: pypm <command> [options]\n"
         "\nCommands:\n"
         "  doctor                 Diagnose build environment\n"
         "  sandbox [-d dir]       Spawn an isolated shell (default: /tmp)\n"
         "  plugin add <name> <src>    Install plugin from URL/path\n"
         "  plugin run <name> [args]   Dispatch to plugin\n"
         "  pypylock [-o file]     Produce hermetic archive (default: dist/venv.tar.gz)\n"
         "  version                Print version\n"
         "  help                   This message\n"
         "\nEnvironment:\n"
         "  PYP_WORKSPACE_ROOT     Override workspace root detection\n");
}

/** Main entry point and command dispatcher. */
int main(int argc, char **argv) {
    if (argc < 2) {
        usage();
        return 1;
    }

    const char *cmd = argv[1];
    char *root = find_workspace_root();
    if (root) printf("🗄️  Workspace root: %s\n", root);

    if (!strcmp(cmd, "doctor")) {
        return run_doctor();
    } else if (!strcmp(cmd, "sandbox")) {
        const char *dir = NULL;
        int opt;
        while ((opt = getopt(argc - 1, argv + 1, "d:")) != -1) {
            if (opt == 'd') dir = optarg;
            else { usage(); return 1; }
        }
        return run_sandbox(dir);
    } else if (!strcmp(cmd, "plugin")) {
        if (argc < 3) { usage(); return 1; }
        const char *sub = argv[2];
        if (!strcmp(sub, "add") && argc == 5)
            return plugin_cmd_add(argv[3], argv[4]);
        if (!strcmp(sub, "run") && argc >= 4)
            return load_and_run_plugin(argv[3], argc - 3, argv + 3);
        usage();
        return 1;
    } else if (!strcmp(cmd, "pypylock")) {
        const char *out = "dist/venv.tar.gz";
        int opt;
        while ((opt = getopt(argc - 1, argv + 1, "o:")) != -1) {
            if (opt == 'o') out = optarg;
            else { usage(); return 1; }
        }
        return run_pypylock(out);
    } else if (!strcmp(cmd, "version")) {
        puts(PYP_VERSION);
        return 0;
    } else if (!strcmp(cmd, "help") || !strcmp(cmd, "--help") || !strcmp(cmd, "-h")) {
        usage();
        return 0;
    } else {
        fprintf(stderr, "Unknown command: %s\n", cmd);
        usage();
        return 1;
    }
}
