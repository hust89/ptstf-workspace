# Stage 1: Build Environment
FROM rust:1.78-slim AS builder

WORKDIR /usr/src/app

# Install system compilation tools
RUN apt-get update && apt-get install -y protobuf-compiler libssl-dev pkg-config && rm -rf /var/lib/apt/lists/*

# Copy whole cargo workspace
COPY . .

# Compile optimized release target
RUN cargo build --release -p ptstf-grpc

# Stage 2: Minimal Executable Container
FROM debian:bookworm-slim

# Install runtime SSL CA store
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/local/bin

# Copy final compiled executable from builder stage
COPY --from=builder /usr/src/app/target/release/ptstf-grpc .

# Expose microservice communication port
EXPOSE 50051

# Launch high-throughput microservice
CMD ["./ptstf-grpc"]
