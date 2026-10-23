# Tachyon daemon for Kapa. Built from a Tachyon checkout passed as the
# `tachyon_src` build context (compose: TACHYON_SRC, default ../tachyon).
# Tachyon needs only libc and the NVIDIA driver's libcuda.so.1, which the
# NVIDIA container runtime mounts in: no CUDA image.
FROM rust:1.88-slim-bookworm AS build
COPY --from=tachyon_src . /src
WORKDIR /src
RUN cargo build --release --locked --bin tachyon

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -u 1000 -m kapa
COPY --from=build /src/target/release/tachyon /usr/local/bin/tachyon
COPY entrypoint.sh /entrypoint.sh
USER 1000
ENV TACHYON_HOME=/models TACHYON_SOCKET=/run/tachyon/tachyon.sock
ENTRYPOINT ["/bin/sh", "/entrypoint.sh"]
