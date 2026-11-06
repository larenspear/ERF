#ifndef ERF_DESCRIPTION_WRITER_H_
#define ERF_DESCRIPTION_WRITER_H_

#include <cstddef>
#include <initializer_list>
#include <ostream>
#include <string>
#include <vector>

namespace erf_description {

enum class Format { JSON, YAML };

/** Emit ERF's fixed description structure in either format, without a runtime
 * parser dependency. YAML uses only mappings, flow sequences and JSON scalars;
 * all strings are quoted, including version strings that look like numbers.
 */
class Writer
{
public:
    Writer (std::ostream& output, Format format) : os(output), yaml(format == Format::YAML)
    {
        if (!yaml) os << '{';
    }

    void begin_object (const char* name)
    {
        key(name, true);
        os << (yaml ? '\n' : '{');
        first.push_back(true);
    }

    void end_object ()
    {
        first.pop_back();
        if (!yaml) os << '}';
    }

    void finish () { os << (yaml ? "" : "}\n"); }

    void value (const char* name, std::nullptr_t)
    {
        key(name); os << "null"; end_value();
    }

    void value (const char* name, bool v)
    {
        key(name); os << (v ? "true" : "false"); end_value();
    }

    void value (const char* name, int v)
    {
        key(name); os << std::to_string(v); end_value();
    }

    void value (const char* name, const char* v)
    {
        key(name);
        if (v == nullptr) os << "null";
        else quoted(v);
        end_value();
    }

    // Existing build-info APIs use empty strings for unavailable metadata.
    void optional_string (const char* name, const char* v)
    {
        value(name, v != nullptr && *v != '\0' ? v : nullptr);
    }

    void strings (const char* name, std::initializer_list<const char*> values)
    {
        key(name);
        os << '[';
        bool first_value = true;
        for (const char* v : values) {
            if (!first_value) os << ", ";
            quoted(v);
            first_value = false;
        }
        os << ']';
        end_value();
    }

private:
    std::ostream& os;
    bool yaml;
    std::vector<bool> first{true};

    void key (const char* name, bool object = false)
    {
        if (yaml) {
            os << std::string((first.size() - 1) * 2, ' ') << name << ':';
            if (!object) os << ' ';
        } else {
            if (!first.back()) os << ',';
            quoted(name);
            os << ':';
        }
        first.back() = false;
    }

    void end_value () { if (yaml) os << '\n'; }

    void quoted (const char* value)
    {
        constexpr char hex[] = "0123456789abcdef";
        os << '"';
        for (const auto* c = reinterpret_cast<const unsigned char*>(value); *c; ++c) {
            if (*c == '"' || *c == '\\') {
                os << '\\' << static_cast<char>(*c);
            } else if (*c < 0x20 || *c == 0x7f) {
                os << "\\u00" << hex[*c >> 4] << hex[*c & 15];
            } else if (*c == 0xc2 && c[1] >= 0x80 && c[1] <= 0x9f) {
                // C1 control characters, including YAML's NEL line break.
                ++c;
                os << "\\u00" << hex[*c >> 4] << hex[*c & 15];
            } else if (*c == 0xe2 && c[1] == 0x80 && (c[2] == 0xa8 || c[2] == 0xa9)) {
                // Avoid YAML 1.1 parser folding of Unicode line/paragraph separators.
                os << "\\u202" << (c[2] == 0xa8 ? '8' : '9');
                c += 2;
            } else {
                os << static_cast<char>(*c);
            }
        }
        os << '"';
    }
};

} // namespace erf_description
#endif
