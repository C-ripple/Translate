FROM --platform=linux/amd64 ubuntu@sha256:534baea6a22c03a63003dbc8dbe78fe34bc0d7e595d9a9dc9834884ff530eb55

# Exact Jammy ABI-5 compatibility libraries, as required by the SDK simulator.
# Official package checksums:
# https://packages.ubuntu.com/jammy/amd64/libncurses5/download
# https://packages.ubuntu.com/jammy/amd64/libtinfo5/download
COPY libncurses5_6.3-2ubuntu0.3_amd64.deb libtinfo5_6.3-2ubuntu0.3_amd64.deb /tmp/
RUN cd /tmp && \
    echo '58da01991712c6856af0658dc5e9cb533c1a6854ab950055ade185e35bc9c276  libncurses5_6.3-2ubuntu0.3_amd64.deb' | sha256sum -c - && \
    echo '4df4288404108f1a156d014e8764a064e977e34e6d44931ab60451694c03c90d  libtinfo5_6.3-2ubuntu0.3_amd64.deb' | sha256sum -c - && \
    dpkg -i libtinfo5_6.3-2ubuntu0.3_amd64.deb libncurses5_6.3-2ubuntu0.3_amd64.deb && \
    rm libncurses5_6.3-2ubuntu0.3_amd64.deb libtinfo5_6.3-2ubuntu0.3_amd64.deb
COPY libatomic1_14.2.0-4ubuntu2~24.04.1_amd64.deb /tmp/
RUN cd /tmp && \
    echo 'fe49cbbc7be753528380c724a8eef5f1e31dffa9221f692c5069048d81c7449d  libatomic1_14.2.0-4ubuntu2~24.04.1_amd64.deb' | sha256sum -c - && \
    dpkg -i libatomic1_14.2.0-4ubuntu2~24.04.1_amd64.deb && \
    rm libatomic1_14.2.0-4ubuntu2~24.04.1_amd64.deb
WORKDIR /tmp
