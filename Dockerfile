FROM gcc:14-bookworm AS build
WORKDIR /build
COPY . .
RUN set -eu; \
    flags='-std=c++20 -O3 -DNDEBUG -march=x86-64-v3 -pthread -static-libstdc++ -static-libgcc'; \
    mkdir -p /tmp/training/events /tmp/training/clicks /tmp/training/links; \
    (cd /tmp/training && LC_ALL=C awk -f /build/pgo-train.awk); \
    g++ $flags -fprofile-generate=/tmp/profile src/main.cpp -o shopping-sessions; \
    DATA_DIR=/tmp/training OUTPUT_DIR=/tmp/training/output NCPU=1 ./shopping-sessions; \
    g++ $flags -fprofile-use=/tmp/profile -fprofile-partial-training -Werror=missing-profile src/main.cpp -o shopping-sessions; \
    rm -rf /tmp/training /tmp/profile

FROM debian:bookworm-slim
COPY --from=build /build/shopping-sessions /usr/local/bin/shopping-sessions
USER 1000:1000
ENTRYPOINT ["/usr/local/bin/shopping-sessions"]
