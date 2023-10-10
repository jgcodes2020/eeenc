#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include "chunks.hpp"

static inline void canonical_encode() {
  int c;
  while ((c = getchar()) != EOF) {
    for (int i = 7; i >= 0; i--) {
      putchar(((((c >> i) & 1) << 5) | 0x45) ^ 0x20);
    }
  }
}

static inline void simd_encode() {
  while (!feof(stdin)) {
    size_t read = fread(eeenc::input_data, 1, sizeof(eeenc::input_data), stdin);
    size_t to_write = eeenc::process_chunk(read);
    fwrite(eeenc::output_data, 1, to_write, stdout);
  }
}

int main(int argc, char* argv[]) {
  if (argc == 2 && strcmp(argv[1], "-c") == 0)
    canonical_encode();
  else
    simd_encode();
}