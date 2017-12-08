#include <Python.h>
#include "frameobject.h"
#include "internal/pystate.h"

#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <time.h>
#include <unistd.h>

static PyMethodDef module_methods[] = {
    {NULL, NULL},
};

PyDoc_STRVAR(module_doc,
"perfmap module.");

static FILE *handle;
static Py_ssize_t perf_code_extra;

struct PerfJitHeader {
    uint32_t magic_;
    uint32_t version_;
    uint32_t size_;
    uint32_t elf_mach_target_;
    uint32_t reserved_;
    uint32_t process_id_;
    uint64_t time_stamp_;
    uint64_t flags_;
};

struct PerfJitBase {
    uint32_t event_;
    uint32_t size_;
    uint64_t time_stamp_;
};

struct PerfJitCodeLoad {
    struct PerfJitBase base;
    uint32_t process_id_;
    uint32_t thread_id_;
    uint64_t vma_;
    uint64_t code_address_;
    uint64_t code_size_;
    uint64_t code_id_;
};

struct PerfJitCodeUnwindingInfo {
    struct PerfJitBase base;
    uint64_t unwinding_size_;
    uint64_t eh_frame_hdr_size_;
    uint64_t mapped_size_;
    // Followed by size_ - sizeof(PerfJitCodeUnwindingInfo) bytes of data.
};

static void *marker = NULL;

static void
perfmap_free(void *mod)
{
    if (marker && marker != MAP_FAILED)
        munmap(marker, 4096);
    fclose(handle);
}

static struct PyModuleDef module_def = {
    PyModuleDef_HEAD_INIT,
    "perfmap",
    module_doc,
    0, /* non-negative size to be able to unload the module */
    module_methods,
    NULL,
    NULL,
    NULL,
    perfmap_free,
};

typedef PyObject *(*Trampoline) (PyFrameObject *, int);

static Trampoline trampolines[512];
static unsigned char *code;

static uint64_t
nanoseconds() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec*1000000000 + ts.tv_nsec;
}

static uint64_t code_id = 0;

static void
write_code_load(PyCodeObject *co) {
    const char *name = PyUnicode_AsUTF8(co->co_name);
    if (!name) {
        PyErr_Clear();
        return;
    }
    size_t name_size = strlen(name);
    struct PerfJitCodeLoad event = {
        {
            0,
            sizeof(event) + name_size + 1 + 32,
            nanoseconds(),
        },
        getpid(),
        syscall(SYS_gettid),
        0,
        (uintptr_t)code,
        32,
        code_id++,
    };
    fwrite(&event, sizeof(event), 1, handle);
    fwrite(name, name_size, 1, handle);
    fwrite("\0", 1, 1, handle);
    fwrite(code, 32, 1, handle);
}

static void
write_unwind(PyCodeObject *co) {
    char eh_header[20];
    struct PerfJitCodeUnwindingInfo event = {
        {
            4,
            sizeof(event) + sizeof(eh_header) + 4,
            nanoseconds(),
        },
        sizeof(eh_header),
        sizeof(eh_header),
        0,
    };
    fwrite(&event, sizeof(event), 1, handle);
    memset(eh_header, 0, sizeof(eh_header));
    eh_header[0] = 1; // version
    eh_header[1] = 0x0b | 0x10; // SData4 | PcRel
    eh_header[2] = 0x03; // UData4
    eh_header[3] = 0x0b | 0x30; // SData4 | DataRel
    fwrite(eh_header, sizeof(eh_header), 1, handle);
    fwrite("\0\0\0\0", 4, 1, handle);
}

static PyObject *
perfmapeval(PyFrameObject *f, int throwflag)
{
    PyCodeObject *co = f->f_code;
    Trampoline trampoline;
    if (_PyCode_GetExtra((PyObject *)co, perf_code_extra, (void **)&trampoline) == -1) {
        return NULL;
    }
    if (trampoline == NULL) {
        trampoline = (Trampoline)code;
        if (_PyCode_SetExtra((PyObject *)co, perf_code_extra, trampoline) == -1)
            return NULL;
        if (!code_id) {
            printf("%p\n", perfmapeval);
            write_unwind(co);
            write_code_load(co);
        }
    }
    return trampoline(f, throwflag);
}

static void noop(void *x) {}

PyMODINIT_FUNC
PyInit_perfmap(void)
{
    code = mmap(NULL, 4096, PROT_EXEC|PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
    if (!code) {
        PyErr_NoMemory();
        return NULL;
    }
    unsigned char *p = code;
    // We need to keep the stack aligned to 16-bytes. We might as well have a
    // frame pointer.
    *p++ = 0x55; // push %rbp
    *p++ = 0x48; // mov %rsp,%rbp
    *p++ = 0x89;
    *p++ = 0xe5;
    *p++ = 0x48; // mov $_PyEval_EvalFrameDefault,%rax
    *p++ = 0xb8;
    Trampoline t = _PyEval_EvalFrameDefault;
    p = mempcpy(p, &t, sizeof(void *));
    *p++ = 0xff; // call *rax
    *p++ = 0xd0;
    *p++ = 0x5d; // pop %rbp
    *p++ = 0xc3; // retq

    perf_code_extra = _PyEval_RequestCodeExtraIndex(noop);
    if (perf_code_extra == -1) {
        return NULL;
    }
    char fnbuf[64];
    snprintf(fnbuf, sizeof(fnbuf), "./jit-%d.dump", getpid());
    int fd = open(fnbuf, O_CREAT | O_TRUNC | O_RDWR, 0666);
    if (fd == -1) {
        PyErr_SetFromErrno(PyExc_IOError);
        return NULL;
    }
    marker = mmap(NULL, 4096, PROT_READ | PROT_EXEC, MAP_PRIVATE, fd, 0);
    if (marker == MAP_FAILED) {
        close(fd);
        PyErr_SetFromErrno(PyExc_IOError);
        return NULL;
    }
    handle = fdopen(fd, "w+");
    if (!handle) {
        close(fd);
        PyErr_SetFromErrno(PyExc_IOError);
        return NULL;
    }
    struct PerfJitHeader header = {
        0x4A695444,
        1,
        sizeof(header),
        62,
        0xdeadbeef,
        getpid(),
        nanoseconds(),
        0,
    };
    fwrite(&header, sizeof(header), 1, handle);

    for (size_t i = 0; i < sizeof(trampolines)/sizeof(Trampoline); i++)
        trampolines[i] = _PyEval_EvalFrameDefault;
    
    PyObject *m = PyModule_Create(&module_def);
    if (!m) {
        fclose(handle);
        return NULL;
    }

    PyThreadState_GET()->interp->eval_frame = perfmapeval;
    return m;
}
