FROM python:3.11-slim
WORKDIR /workspace
COPY . /workspace
RUN python -m pip install --no-cache-dir -e .
CMD ["vectorhijack-sr", "smoke", "--config", "configs/smoke/synthetic.yaml"]
