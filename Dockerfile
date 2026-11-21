FROM gcc:14-bookworm AS build
WORKDIR /build
COPY src/ src/
RUN g++ -std=c++20 -O3 -DNDEBUG -march=x86-64-v3 -pthread -static-libstdc++ -static-libgcc src/main.cpp -o shopping-sessions

FROM debian:bookworm-slim
COPY --from=build /build/shopping-sessions /usr/local/bin/shopping-sessions
USER 1000:1000
ENTRYPOINT ["/usr/local/bin/shopping-sessions"]
