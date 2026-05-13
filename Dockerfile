# --- Stage 1: Builder ---
# This stage compiles the Rust extension module into a Python wheel.
FROM python:3.11-slim-bullseye AS builder

# Install Rust toolchain
RUN apt-get update && apt-get install -y curl build-essential && \
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# Install Maturin for building the Rust-Python bridge
RUN pip install maturin

# Copy only the Rust project to build it
WORKDIR /app
COPY ./atomic-lang-model ./atomic-lang-model

# Build the wheel
# The output will be in /app/atomic-lang-model/target/wheels
RUN cd atomic-lang-model && maturin build --release --out dist


# --- Stage 2: Final Application ---
# This stage builds the final, lightweight image for the application.
FROM python:3.11-slim-bullseye

# Set the working directory in the container
WORKDIR /app

# Install Python dependencies for the demo
COPY ./nasa_demo/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the compiled wheel from the builder stage and install it
COPY --from=builder /app/atomic-lang-model/dist/*.whl .
RUN pip install *.whl && rm *.whl

# Copy the application code and the data
COPY ./nasa_demo ./nasa_demo
COPY ./atomic-lang-model/python ./atomic-lang-model/python

# Expose the port the app runs on
EXPOSE 8080

# Define the command to run the application
CMD ["python", "nasa_demo/app.py"]
