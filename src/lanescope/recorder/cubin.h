#pragma once

#include <elf.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace lanescope {

// Return whether the name of a function in an ELF cubin holds `text`. A cubin that the
// search cannot read counts as one that holds it, which keeps its module patched.
inline bool NamesFunction(const void* cubin, size_t size, const std::string& text) {
    const auto* bytes = static_cast<const unsigned char*>(cubin);
    Elf64_Ehdr header;
    if (bytes == nullptr || size < sizeof(header)) {
        return true;
    }
    std::memcpy(&header, bytes, sizeof(header));
    if (std::memcmp(header.e_ident, ELFMAG, SELFMAG) != 0 || header.e_ident[EI_CLASS] != ELFCLASS64 ||
        header.e_shentsize != sizeof(Elf64_Shdr) || header.e_shoff > size ||
        header.e_shnum > (size - header.e_shoff) / sizeof(Elf64_Shdr)) {
        return true;
    }
    std::vector<Elf64_Shdr> sections(header.e_shnum);
    std::memcpy(sections.data(), bytes + header.e_shoff, sections.size() * sizeof(Elf64_Shdr));
    bool searched = false;
    for (const Elf64_Shdr& table : sections) {
        if (table.sh_type != SHT_SYMTAB) {
            continue;
        }
        if (table.sh_entsize != sizeof(Elf64_Sym) || table.sh_link >= sections.size()) {
            return true;
        }
        const Elf64_Shdr& names = sections[table.sh_link];
        if (table.sh_offset > size || table.sh_size > size - table.sh_offset || names.sh_offset > size ||
            names.sh_size > size - names.sh_offset) {
            return true;
        }
        const char* strings = reinterpret_cast<const char*>(bytes + names.sh_offset);
        for (uint64_t at = 0; at + sizeof(Elf64_Sym) <= table.sh_size; at += sizeof(Elf64_Sym)) {
            Elf64_Sym symbol;
            std::memcpy(&symbol, bytes + table.sh_offset + at, sizeof(symbol));
            if (ELF64_ST_TYPE(symbol.st_info) != STT_FUNC || symbol.st_name >= names.sh_size) {
                continue;
            }
            const char* name = strings + symbol.st_name;
            if (std::string_view(name, strnlen(name, names.sh_size - symbol.st_name)).find(text) !=
                std::string_view::npos) {
                return true;
            }
        }
        searched = true;
    }
    return !searched;
}

}  // namespace lanescope
