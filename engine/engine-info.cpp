#include "ggml-backend.h"

#include <cstdio>
#include <cstring>

static void print_json_string(const char * value) {
    std::putchar('"');
    for (const unsigned char * p = reinterpret_cast<const unsigned char *>(value); *p; ++p) {
        switch (*p) {
            case '"': std::fputs("\\\"", stdout); break;
            case '\\': std::fputs("\\\\", stdout); break;
            case '\b': std::fputs("\\b", stdout); break;
            case '\f': std::fputs("\\f", stdout); break;
            case '\n': std::fputs("\\n", stdout); break;
            case '\r': std::fputs("\\r", stdout); break;
            case '\t': std::fputs("\\t", stdout); break;
            default:
                if (*p < 0x20) {
                    std::printf("\\u%04x", static_cast<unsigned int>(*p));
                } else {
                    std::putchar(*p);
                }
        }
    }
    std::putchar('"');
}

int main() {
    ggml_backend_load_all();
    std::putchar('[');
    bool first = true;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto * device = ggml_backend_dev_get(i);
        auto * provider = ggml_backend_dev_backend_reg(device);
        if (std::strcmp(ggml_backend_reg_name(provider), "RPC") == 0) {
            continue;
        }
        ggml_backend_dev_props props{};
        ggml_backend_dev_get_props(device, &props);
        const char * kind;
        switch (props.type) {
            case GGML_BACKEND_DEVICE_TYPE_GPU: kind = "gpu"; break;
            case GGML_BACKEND_DEVICE_TYPE_IGPU: kind = "igpu"; break;
            case GGML_BACKEND_DEVICE_TYPE_CPU: kind = "cpu"; break;
            case GGML_BACKEND_DEVICE_TYPE_ACCEL: kind = "accelerator"; break;
            case GGML_BACKEND_DEVICE_TYPE_META: continue;
        }
        if (!first) {
            std::putchar(',');
        }
        first = false;
        std::fputs("{\"name\":", stdout);
        print_json_string(props.name);
        std::fputs(",\"description\":", stdout);
        print_json_string(props.description);
        std::printf(",\"kind\":\"%s\",\"total_memory\":%zu,\"free_memory\":%zu}",
                    kind, props.memory_total, props.memory_free);
    }
    std::puts("]");
}
