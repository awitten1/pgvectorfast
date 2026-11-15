/*
 * Linux/x86-64 entry interception. A nearby relay makes a five-byte direct
 * jump sufficient. Instructions are decoded, never matched to fixed bytes.
 * Install only in the single-threaded postmaster before children are forked.
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "postgres.h"
#include "calico_internal.h"
#include "utils/elog.h"

#include <capstone/capstone.h>
#include <dlfcn.h>
#include <elf.h>
#include <link.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <limits.h>

static char *relay_page;
static size_t relay_used;
static size_t page_size;
static struct {
    void *address;
    void *replacement;
    void *original;
    bool native_when_inactive;
} installed[256];
static int installed_count;
static const unsigned char *executable;
static size_t executable_size;
static void redirect_address(void *address, void *replacement, const char *name,
                             void **original, bool native_when_inactive);

/* Callback entries need not appear in .dynsym. Use the executable's real
 * function symbols as well, and reject an unbounded entry (including a stripped
 * private callback) rather than decode into an adjacent function. This is a
 * startup-only qualification; no ELF lookup remains on the buffer path. */
static const Elf64_Ehdr *
executable_image(void)
{
    const Elf64_Ehdr *header;
    if (!executable)
    {
        struct stat statbuf;
        int fd = open("/proc/self/exe", O_RDONLY | O_CLOEXEC);
        if (fd < 0 || fstat(fd, &statbuf) || statbuf.st_size < sizeof(Elf64_Ehdr))
            elog(FATAL, "cannot inspect PostgreSQL ELF image: %m");
        executable_size = statbuf.st_size;
        executable = mmap(NULL, executable_size, PROT_READ, MAP_PRIVATE, fd, 0);
        close(fd);
        if (executable == MAP_FAILED)
            elog(FATAL, "cannot map PostgreSQL ELF image: %m");
        /* Meson's pg_config flags can omit b_lto. Inspect executable metadata
         * before installing any route: LTO can inline buffer mapping and
         * writer operations past the selected entry points. */
        if (memmem(executable, executable_size, "-flto", 5) ||
            memmem(executable, executable_size, ".lto_priv.", 10) ||
            memmem(executable, executable_size, ".gnu.lto_", 9) ||
            memmem(executable, executable_size, ".llvm.", 6))
            elog(FATAL, "Calico does not qualify LTO PostgreSQL executables");
    }
    header = (const Elf64_Ehdr *) executable;
    if (memcmp(header->e_ident, ELFMAG, SELFMAG) ||
        header->e_ident[EI_CLASS] != ELFCLASS64 ||
        header->e_ident[EI_DATA] != ELFDATA2LSB || header->e_machine != EM_X86_64 ||
        (header->e_type != ET_EXEC && header->e_type != ET_DYN) ||
        header->e_shentsize != sizeof(Elf64_Shdr) ||
        header->e_shoff > executable_size ||
        header->e_shnum > (executable_size - header->e_shoff) / sizeof(Elf64_Shdr))
        elog(FATAL, "unsupported PostgreSQL ELF image for Calico");
    return header;
}

static size_t
function_size(void *address, const Dl_info *info, const char *name)
{
    const Elf64_Ehdr *header = executable_image();
    const Elf64_Shdr *sections;
    size_t result = 0;
    sections = (const Elf64_Shdr *) (executable + header->e_shoff);
    for (unsigned int i = 0; i < header->e_shnum; i++)
    {
        const Elf64_Shdr *section = &sections[i];
        const Elf64_Sym *symbols;
        const Elf64_Shdr *strings;
        uintptr_t bias = header->e_type == ET_DYN ? (uintptr_t) info->dli_fbase : 0;
        if (section->sh_type != SHT_SYMTAB && section->sh_type != SHT_DYNSYM)
            continue;
        if (section->sh_entsize != sizeof(Elf64_Sym) ||
            section->sh_offset > executable_size ||
            section->sh_size > executable_size - section->sh_offset ||
            section->sh_link >= header->e_shnum)
            elog(FATAL, "invalid PostgreSQL ELF symbol table");
        strings = &sections[section->sh_link];
        if (strings->sh_offset > executable_size ||
            strings->sh_size > executable_size - strings->sh_offset)
            elog(FATAL, "invalid PostgreSQL ELF string table");
        symbols = (const Elf64_Sym *) (executable + section->sh_offset);
        for (size_t j = 0; j < section->sh_size / sizeof(Elf64_Sym); j++)
        {
            const Elf64_Sym *symbol = &symbols[j];
            const char *symbol_name;
            if (symbol->st_name >= strings->sh_size)
                elog(FATAL, "invalid PostgreSQL ELF symbol name");
            symbol_name = (const char *) executable + strings->sh_offset + symbol->st_name;
            if (!memchr(symbol_name, 0, strings->sh_size - symbol->st_name))
                elog(FATAL, "unterminated PostgreSQL ELF symbol name");
            /* Cross-entry optimized clones have their own addresses and
             * cannot be covered by a detour of the exported entry. */
            if (ELF64_ST_TYPE(symbol->st_info) == STT_FUNC &&
                strncmp(symbol_name, name, strlen(name)) == 0 &&
                symbol_name[strlen(name)] == '.' &&
                bias + symbol->st_value != (uintptr_t) address &&
                (strstr(symbol_name + strlen(name), ".constprop.") ||
                 strstr(symbol_name + strlen(name), ".isra.") ||
                 strstr(symbol_name + strlen(name), ".llvm.")))
                elog(FATAL, "Calico cannot intercept optimized PostgreSQL clone %s", symbol_name);
            if (ELF64_ST_TYPE(symbol->st_info) == STT_FUNC &&
                symbol->st_shndx != SHN_UNDEF && symbol->st_size &&
                bias + symbol->st_value == (uintptr_t) address &&
                (!result || symbol->st_size < result))
                result = symbol->st_size;
        }
    }
    if (!result)
        ereport(FATAL, (errmsg("cannot bound PostgreSQL entry %s for Calico", name),
            errhint("Use a qualified executable retaining its function symbol table.")));
    return result;
}

/* Resolve private data once in the postmaster. Children inherit pointers to
 * their own native backend-local storage; buffer accesses need no lookup or
 * synchronization. Private layout is checked by the version adapter as well
 * as the ELF symbol size. A stripped or ambiguous image is unsupported. */
void *
calico_native_data(const char *name, size_t size, size_t alignment)
{
    const Elf64_Ehdr *header = executable_image();
    const Elf64_Shdr *sections = (const Elf64_Shdr *) (executable + header->e_shoff);
    Dl_info info;
    uintptr_t result = 0;
    uintptr_t bias;

    if (!dladdr(dlsym(RTLD_DEFAULT, "ReadBuffer"), &info))
        elog(FATAL, "cannot locate PostgreSQL executable for Calico");
    bias = header->e_type == ET_DYN ? (uintptr_t) info.dli_fbase : 0;
    for (unsigned int i = 0; i < header->e_shnum; i++)
    {
        const Elf64_Shdr *table = &sections[i];
        const Elf64_Shdr *strings;
        const Elf64_Sym *symbols;
        if (table->sh_type != SHT_SYMTAB && table->sh_type != SHT_DYNSYM)
            continue;
        if (table->sh_entsize != sizeof(Elf64_Sym) ||
            table->sh_offset > executable_size ||
            table->sh_size > executable_size - table->sh_offset ||
            table->sh_link >= header->e_shnum)
            elog(FATAL, "invalid PostgreSQL ELF symbol table");
        strings = &sections[table->sh_link];
        if (strings->sh_offset > executable_size ||
            strings->sh_size > executable_size - strings->sh_offset)
            elog(FATAL, "invalid PostgreSQL ELF string table");
        symbols = (const Elf64_Sym *) (executable + table->sh_offset);
        for (size_t j = 0; j < table->sh_size / sizeof(Elf64_Sym); j++)
        {
            const Elf64_Sym *symbol = &symbols[j];
            const Elf64_Shdr *storage;
            const char *symbol_name;
            uintptr_t address;
            if (ELF64_ST_TYPE(symbol->st_info) != STT_OBJECT ||
                symbol->st_shndx == SHN_UNDEF)
                continue;
            if (symbol->st_name >= strings->sh_size)
                elog(FATAL, "invalid PostgreSQL ELF symbol name");
            symbol_name = (const char *) executable + strings->sh_offset + symbol->st_name;
            if (!memchr(symbol_name, 0, strings->sh_size - symbol->st_name))
                elog(FATAL, "unterminated PostgreSQL ELF symbol name");
            if (strcmp(symbol_name, name) != 0)
                continue;
            if (symbol->st_shndx >= header->e_shnum)
                elog(FATAL, "invalid PostgreSQL storage for %s", name);
            storage = &sections[symbol->st_shndx];
            address = bias + symbol->st_value;
            if (symbol->st_size != size || address % alignment ||
                (storage->sh_flags & (SHF_ALLOC | SHF_WRITE)) != (SHF_ALLOC | SHF_WRITE) ||
                symbol->st_value < storage->sh_addr ||
                symbol->st_value - storage->sh_addr > storage->sh_size ||
                size > storage->sh_size - (symbol->st_value - storage->sh_addr) ||
                (result && result != address))
                elog(FATAL, "incompatible PostgreSQL private state %s for Calico", name);
            result = address;
        }
    }
    if (!result)
        ereport(FATAL, (errmsg("cannot locate PostgreSQL private state %s for Calico", name),
            errhint("Retain the executable's private data symbol table.")));
    /* Section metadata must also describe storage in a writable load segment. */
    if (header->e_phentsize == sizeof(Elf64_Phdr) &&
        header->e_phoff <= executable_size &&
        header->e_phnum <= (executable_size - header->e_phoff) / sizeof(Elf64_Phdr))
    {
        const Elf64_Phdr *segments = (const Elf64_Phdr *) (executable + header->e_phoff);
        uintptr_t value = result - bias;
        for (unsigned int i = 0; i < header->e_phnum; i++)
        {
            const Elf64_Phdr *segment = &segments[i];
            if (segment->p_type == PT_LOAD && (segment->p_flags & PF_W) &&
                value >= segment->p_vaddr && value - segment->p_vaddr <= segment->p_memsz &&
                size <= segment->p_memsz - (value - segment->p_vaddr))
                return (void *) result;
        }
    }
    elog(FATAL, "PostgreSQL private state %s is outside writable storage", name);
    return NULL;
}

void
calico_interception_finish(void)
{
    if (executable)
        munmap((void *) executable, executable_size);
    executable = NULL;
}

static void
verify_original_entry(void *address, const Dl_info *info, size_t length,
                      const char *name)
{
    const Elf64_Ehdr *header = (const Elf64_Ehdr *) executable;
    const Elf64_Phdr *segments;
    uintptr_t bias = header->e_type == ET_DYN ? (uintptr_t) info->dli_fbase : 0;
    uintptr_t value = (uintptr_t) address - bias;
    if (header->e_phentsize != sizeof(Elf64_Phdr) ||
        header->e_phoff > executable_size ||
        header->e_phnum > (executable_size - header->e_phoff) / sizeof(Elf64_Phdr))
        elog(FATAL, "invalid PostgreSQL ELF load segments");
    segments = (const Elf64_Phdr *) (executable + header->e_phoff);
    for (unsigned int i = 0; i < header->e_phnum; i++)
    {
        const Elf64_Phdr *segment = &segments[i];
        uint64 offset;
        if (segment->p_type != PT_LOAD || !(segment->p_flags & PF_X) ||
            value < segment->p_vaddr || value - segment->p_vaddr > segment->p_filesz ||
            length > segment->p_filesz - (value - segment->p_vaddr))
            continue;
        offset = value - segment->p_vaddr;
        if (segment->p_offset > executable_size || offset > executable_size - segment->p_offset ||
            length > executable_size - segment->p_offset - offset)
            elog(FATAL, "invalid PostgreSQL ELF code range");
        /* Covers movabs/jmp and prefixed detours as well as a leading JMP.
         * Qualified PIE code has no runtime text relocations here. */
        if (memcmp(address, executable + segment->p_offset + offset, length))
            elog(FATAL, "PostgreSQL entry %s differs from its ELF image; competing instrumentation or unqualified text relocation", name);
        return;
    }
    elog(FATAL, "cannot locate PostgreSQL code bytes for %s", name);
}

static void
relative_jump(unsigned char *site, const void *target)
{
    intptr_t delta = (const char *) target - (const char *) site - 5;
    int32 offset;
    if (delta < INT32_MIN || delta > INT32_MAX)
        elog(FATAL, "Calico relay outside direct-jump range");
    offset = (int32) delta;
    site[0] = 0xe9;
    memcpy(site + 1, &offset, 4);
}

static char *
relay_allocate(void *address, size_t size)
{
    uintptr_t base;
    if (!page_size)
        page_size = (size_t) sysconf(_SC_PAGESIZE);
    if (relay_page && relay_used + size < page_size &&
        llabs((long long) ((char *) address - relay_page)) < INT32_MAX - 65536)
    {
        char *result = relay_page + relay_used;
        relay_used += size;
        if (mprotect(relay_page, page_size, PROT_READ | PROT_WRITE))
            elog(FATAL, "cannot write Calico relay: %m");
        return result;
    }
    base = (uintptr_t) address & ~((uintptr_t) page_size - 1);
    for (uintptr_t distance = 65536; distance < 0x70000000; distance += 65536)
    {
        void *p = mmap((void *) (base + distance), page_size,
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS | MAP_FIXED_NOREPLACE, -1, 0);
        if (p != MAP_FAILED)
        {
            relay_page = p;
            relay_used = size;
            return p;
        }
    }
    elog(FATAL, "cannot reserve a nearby Calico relay");
    return NULL;
}

void
calico_redirect(const char *name, void *replacement, void **original)
{
    void *address = dlsym(RTLD_DEFAULT, name);
    if (!address)
        elog(FATAL, "Calico requires exported PostgreSQL entry %s", name);
    calico_redirect_address(address, replacement, name, original);
}

void
calico_redirect_address(void *address, void *replacement,
                        const char *name, void **original)
{
    redirect_address(address, replacement, name, original, false);
}

void
calico_redirect_bufmgr(const char *name, void *replacement, void **original)
{
    void *address = dlsym(RTLD_DEFAULT, name);
    if (!address)
        elog(FATAL, "Calico requires exported PostgreSQL entry %s", name);
    /* Selected entries share native private state. A separate optimized
     * clone of a required entry must be qualified independently. */
    redirect_address(address, replacement, name, original, true);
}

static void
redirect_address(void *address, void *replacement,
                 const char *name, void **original, bool native_when_inactive)
{
    csh decoder;
    cs_insn *insns = NULL;
    size_t count, used = 0, available;
    bool in_place;
    unsigned char *entry = address;
    unsigned char *relay;
    unsigned char saved[32];
    uintptr_t start, end;
    Dl_info info;
    if (!page_size)
        page_size = (size_t) sysconf(_SC_PAGESIZE);
    for (int i = 0; i < installed_count; i++)
        if (installed[i].address == address)
        {
            if (installed[i].replacement != replacement ||
                installed[i].native_when_inactive != native_when_inactive)
                elog(FATAL, "conflicting Calico callback routes at %s", name);
            if (original)
                *original = installed[i].original;
            return;
        }
    if (installed_count == lengthof(installed))
        elog(FATAL, "too many Calico interception sites");
    if (!dladdr(address, &info) || !info.dli_fbase)
        elog(FATAL, "cannot identify Calico interception site %s", name);
    available = function_size(address, &info, name);
    verify_original_entry(address, &info, Min(available, sizeof(saved)), name);
    /* An indirect caller must still land on ENDBR64 when IBT is enabled. */
    if (available >= 4 && memcmp(entry, "\xf3\x0f\x1e\xfa", 4) == 0)
    {
        entry += 4;
        available -= 4;
    }
    if (available < 5)
        elog(FATAL, "PostgreSQL entry %s is too small for Calico interception", name);
    /* Keep the hot entry to a single jump even when a callable original
     * is needed. Its relocated prologue lives separately in the relay. */
    in_place = !native_when_inactive && available >= 14;
    if (cs_open(CS_ARCH_X86, CS_MODE_64, &decoder) != CS_ERR_OK)
        elog(FATAL, "cannot initialize Calico instruction decoder");
    cs_option(decoder, CS_OPT_DETAIL, CS_OPT_ON);
    count = cs_disasm(decoder, entry, Min(available, sizeof(saved)),
                      (uintptr_t) entry, 0, &insns);
    for (size_t i = 0; i < count && used < (in_place ? 14 : 5); i++)
    {
        if (insns[i].id == X86_INS_INT3)
            break;
        /* A jump into another module is an existing interception, not a
         * compiler-generated intra-executable tail call. */
        if (i == 0 && insns[i].id == X86_INS_JMP)
        {
            cs_x86 *x = &insns[i].detail->x86;
            Dl_info target_info;
            if (x->op_count != 1 || x->operands[0].type != X86_OP_IMM ||
                !dladdr((void *) x->operands[0].imm, &target_info) ||
                target_info.dli_fbase != info.dli_fbase)
                elog(FATAL, "Calico found a competing detour at %s", name);
        }
        used += insns[i].size;
    }
    if (used < 5 || used > sizeof(saved))
        elog(FATAL, "cannot safely intercept PostgreSQL entry %s", name);
    if (used < 14)
        in_place = false;
    if (original && in_place)
    {
        /* A short entry patch can leave an early branch target in place. */
        for (size_t i = 0, consumed = 0; consumed < used; i++)
        {
            cs_insn *in = &insns[i];
            cs_x86 *x = &in->detail->x86;
            if ((cs_insn_group(decoder, in, CS_GRP_JUMP) ||
                 cs_insn_group(decoder, in, CS_GRP_CALL)) &&
                x->op_count == 1 && x->operands[0].type == X86_OP_IMM &&
                x->operands[0].imm >= (int64) (uintptr_t) entry &&
                x->operands[0].imm < (int64) (uintptr_t) (entry + used))
            {
                in_place = false;
                used = 0;
                for (size_t j = 0; used < 5; j++)
                    used += insns[j].size;
                break;
            }
            consumed += in->size;
        }
    }
    memcpy(saved, entry, used);
    relay = NULL;
    if (!in_place || original)
    {
        relay = (unsigned char *) relay_allocate(address, 256);
        if (native_when_inactive)
        {
            const bool *active = &calico_local_active;
            int32 active_offset = 192 - 20;

            if (!original)
                elog(FATAL, "Calico buffer entry %s requires its native route", name);
            /* Read the existing process-local flag without entering the DSO.
             * R11 is scratch in the SysV ABI; arguments, stack and return
             * address are untouched. Inactive calls fall straight into the
             * relocated native prefix below. Only installation writes code. */
            memcpy(relay, "\x49\xbb", 2);       /* movabs active, %r11 */
            memcpy(relay + 2, &active, 8);
            memcpy(relay + 10, "\x41\x80\x3b\x00", 4); /* cmpb $0,(%r11) */
            memcpy(relay + 14, "\x0f\x85", 2);  /* jne active route */
            memcpy(relay + 16, &active_offset, 4);
            memcpy(relay + 192, "\xff\x25\x00\x00\x00\x00", 6);
            memcpy(relay + 198, &replacement, 8);
        }
        else
        {
            /* Indirect targets retain their compiler-emitted ENDBR64. */
            memcpy(relay, "\xff\x25\x00\x00\x00\x00", 6);
            memcpy(relay + 6, &replacement, 8);
        }
    }
    if (original)
    {
        unsigned char *dest = relay + (native_when_inactive ? 20 : 32);
        size_t copied = 0, consumed = 0;
        memcpy(dest, "\xf3\x0f\x1e\xfa", 4);
        copied = 4;
        for (size_t i = 0; consumed < used; i++)
        {
            cs_insn *in = &insns[i];
            cs_x86 *x = &in->detail->x86;
            unsigned char *out = dest + copied;
            bool branch = cs_insn_group(decoder, in, CS_GRP_JUMP) ||
                          cs_insn_group(decoder, in, CS_GRP_CALL);
            memcpy(out, in->bytes, in->size);
            if (branch)
            {
                int64 target;
                int64 distance;
                int32 delta;
                size_t length = in->size, offset = x->encoding.imm_offset;
                if (x->op_count != 1 || x->operands[0].type != X86_OP_IMM)
                    elog(FATAL, "unsupported original branch at %s", name);
                target = x->operands[0].imm;
                if (target >= (int64) (uintptr_t) entry &&
                    target < (int64) (uintptr_t) (entry + used))
                    elog(FATAL, "internal prologue branch at %s", name);
                if (x->encoding.imm_size == 1)
                {
                    if (in->bytes[0] == 0xeb)
                    { out[0] = 0xe9; length = 5; offset = 1; }
                    else if ((in->bytes[0] & 0xf0) == 0x70)
                    { out[0] = 0x0f; out[1] = in->bytes[0] + 0x10; length = 6; offset = 2; }
                    else
                        elog(FATAL, "unsupported short branch at %s", name);
                }
                if (x->encoding.imm_size != 1 && x->encoding.imm_size != 4)
                    elog(FATAL, "unsupported original branch width at %s", name);
                distance = target - (int64) (uintptr_t) out - (int64) length;
                if (distance < INT32_MIN || distance > INT32_MAX)
                    elog(FATAL, "cannot relocate original branch at %s", name);
                delta = (int32) distance;
                memcpy(out + offset, &delta, 4);
                copied += length;
            }
            else
            {
                for (unsigned int j = 0; j < x->op_count; j++)
                    if (x->operands[j].type == X86_OP_MEM &&
                        x->operands[j].mem.base == X86_REG_RIP)
                    {
                        int64 target = in->address + in->size + x->disp;
                        int64 distance = target - (int64) (uintptr_t) out - in->size;
                        int32 delta = (int32) distance;
                        if (x->encoding.disp_size != 4 || distance != delta)
                            elog(FATAL, "cannot relocate original entry %s", name);
                        memcpy(out + x->encoding.disp_offset, &delta, 4);
                    }
                copied += in->size;
            }
            consumed += in->size;
            if (native_when_inactive && dest + copied + 5 > relay + 192)
                elog(FATAL, "Calico native prefix exceeds its relay at %s", name);
        }
        relative_jump(dest + copied, entry + used);
        *original = dest;
    }
    cs_free(insns, count);
    cs_close(&decoder);
    if (relay && mprotect(relay_page, page_size, PROT_READ | PROT_EXEC))
        elog(FATAL, "cannot protect Calico relay: %m");
    start = (uintptr_t) entry & ~((uintptr_t) page_size - 1);
    end = ((uintptr_t) entry + used + page_size - 1) & ~((uintptr_t) page_size - 1);
    if (mprotect((void *) start, end - start, PROT_READ | PROT_WRITE | PROT_EXEC))
        elog(FATAL, "cannot enable Calico interception: %m");
    if (memcmp(entry, saved, used))
        elog(FATAL, "PostgreSQL entry changed during Calico installation: %s", name);
    memset(entry, 0x90, used);
    if (in_place)
    {
        memcpy(entry, "\xff\x25\x00\x00\x00\x00", 6);
        memcpy(entry + 6, &replacement, 8);
    }
    else
        relative_jump(entry, relay);
    __builtin___clear_cache((char *) entry, (char *) entry + used);
    if (mprotect((void *) start, end - start, PROT_READ | PROT_EXEC))
        elog(FATAL, "cannot restore PostgreSQL executable protections: %m");
    installed[installed_count].address = address;
    installed[installed_count].replacement = replacement;
    installed[installed_count].original = original ? *original : NULL;
    installed[installed_count].native_when_inactive = native_when_inactive;
    installed_count++;
}
