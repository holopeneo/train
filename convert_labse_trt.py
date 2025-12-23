#!/usr/bin/env python3
# Build Torch-TensorRT TorchScript for Triton (pytorch backend).
# Outputs sentence_embedding and token_embeddings in FP16.

import os
import sys
import platform
from dataclasses import dataclass

import torch
from transformers import AutoModel, AutoTokenizer


@dataclass
class Cfg:
    # HF model
    MODEL_ID: str = os.getenv("MODEL_ID", "sergeyzh/LaBSE-ru-turbo")

    # Output location (inside Triton model repo mount)
    MODEL_REPO_ROOT: str = os.getenv("MODEL_REPO_ROOT", "/models")
    TRT_MODEL_NAME: str = os.getenv("TRT_MODEL_NAME", "labse_trt_v2")
    TRT_MODEL_VERSION: str = os.getenv("TRT_MODEL_VERSION", "1")

    # Shapes
    SEQ_LEN: int = int(os.getenv("SEQ_LEN", "512"))
    MIN_BATCH: int = int(os.getenv("MIN_BATCH", "8"))
    OPT_BATCH: int = int(os.getenv("OPT_BATCH", "8"))
    MAX_BATCH: int = int(os.getenv("MAX_BATCH", "8"))

    # Embedding size
    EMBED_DIM: int = int(os.getenv("EMBED_DIM", "768"))

    # Input dtype
    INPUT_DTYPE: str = os.getenv("INPUT_DTYPE", "int64").lower()  # int64 or int32

    # Precision
    FP16: bool = os.getenv("FP16", "1") == "1"

    # Compilation strictness
    REQUIRE_FULL_COMPILATION: bool = os.getenv("REQUIRE_FULL_COMPILATION", "1") == "1"

    # Output filename inside version folder
    OUTPUT_FILENAME: str = os.getenv("OUTPUT_FILENAME", "model.pt")

    # Triton config + gRPC
    WRITE_CONFIG: bool = os.getenv("WRITE_CONFIG", "1") == "1"
    SENTENCE_OUTPUT_NAME: str = os.getenv("SENTENCE_OUTPUT_NAME", "sentence_embedding")
    TOKEN_OUTPUT_NAME: str = os.getenv("TOKEN_OUTPUT_NAME", "token_embeddings")
    TRITON_URL: str = os.getenv("TRITON_URL", "localhost:8001")
    MANAGE_TRITON: bool = os.getenv("MANAGE_TRITON", "1") == "1"

    # Triton scheduling
    ENABLE_DYNAMIC_BATCHING: bool = os.getenv("ENABLE_DYNAMIC_BATCHING", "1") == "1"
    PREFERRED_BATCH_SIZES: str = os.getenv("PREFERRED_BATCH_SIZES", "4,8,16,32")
    MAX_QUEUE_DELAY_US: int = int(os.getenv("MAX_QUEUE_DELAY_US", "100"))
    INSTANCE_COUNT: int = int(os.getenv("INSTANCE_COUNT", "1"))
    GPU_IDS: str = os.getenv("GPU_IDS", "0")


def torch_dtype_from_name(name: str) -> torch.dtype:
    if name == "int64":
        return torch.int64
    if name == "int32":
        return torch.int32
    raise ValueError("INPUT_DTYPE must be 'int64' or 'int32'")


def config_dtype_from_name(name: str) -> str:
    if name == "int64":
        return "TYPE_INT64"
    if name == "int32":
        return "TYPE_INT32"
    raise ValueError("INPUT_DTYPE must be 'int64' or 'int32'")


def mean_pooling(token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(token_embeddings.dtype)
    summed = (token_embeddings * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-9)
    return summed / denom


class LabseWrapper(torch.nn.Module):
    def __init__(self, hf_model: torch.nn.Module, fp16_outputs: bool) -> None:
        super().__init__()
        self.hf_model = hf_model
        self.fp16_outputs = fp16_outputs

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        if input_ids.dtype != torch.int64:
            input_ids = input_ids.to(torch.int64)
        if attention_mask.dtype != torch.int64:
            attention_mask = attention_mask.to(torch.int64)

        outputs = self.hf_model(input_ids=input_ids, attention_mask=attention_mask)
        token_embeddings = outputs.last_hidden_state
        sentence_embedding = mean_pooling(token_embeddings, attention_mask)

        if self.fp16_outputs:
            sentence_embedding = sentence_embedding.to(torch.float16)
            token_embeddings = token_embeddings.to(torch.float16)

        return sentence_embedding, token_embeddings


def validate_cfg(cfg: Cfg) -> None:
    if cfg.SEQ_LEN <= 0:
        raise ValueError("SEQ_LEN must be > 0")
    if cfg.MIN_BATCH <= 0 or cfg.OPT_BATCH <= 0 or cfg.MAX_BATCH <= 0:
        raise ValueError("MIN_BATCH/OPT_BATCH/MAX_BATCH must be > 0")
    if not (cfg.MIN_BATCH <= cfg.OPT_BATCH <= cfg.MAX_BATCH):
        raise ValueError("MIN_BATCH <= OPT_BATCH <= MAX_BATCH must hold")
    if cfg.INPUT_DTYPE not in {"int64", "int32"}:
        raise ValueError("INPUT_DTYPE must be 'int64' or 'int32'")


def print_env(cfg: Cfg) -> None:
    gpu_name = "N/A"
    if torch.cuda.is_available():
        try:
            gpu_name = torch.cuda.get_device_name(0)
        except Exception:
            gpu_name = "Unknown"

    try:
        import torch_tensorrt

        trt_ver = getattr(torch_tensorrt, "__version__", "unknown")
    except Exception:
        trt_ver = "NOT_INSTALLED"

    out_path = os.path.join(
        cfg.MODEL_REPO_ROOT, cfg.TRT_MODEL_NAME, cfg.TRT_MODEL_VERSION, cfg.OUTPUT_FILENAME
    )

    print("==== Environment ====")
    print("MODEL_ID:", cfg.MODEL_ID)
    print("MODEL_REPO_ROOT:", cfg.MODEL_REPO_ROOT)
    print("TRT_MODEL_NAME:", cfg.TRT_MODEL_NAME)
    print("TRT_MODEL_VERSION:", cfg.TRT_MODEL_VERSION)
    print("output_path:", out_path)
    print("SEQ_LEN:", cfg.SEQ_LEN)
    print("BATCH (min/opt/max):", [cfg.MIN_BATCH, cfg.OPT_BATCH, cfg.MAX_BATCH])
    print("INPUT_DTYPE:", cfg.INPUT_DTYPE)
    print("FP16:", cfg.FP16)
    print("require_full_compilation:", cfg.REQUIRE_FULL_COMPILATION)
    print("torch:", torch.__version__)
    print("torch_cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    print("gpu:", gpu_name)
    print("python:", sys.version.split()[0])
    print("platform:", platform.platform())
    print("torch_tensorrt:", trt_ver)
    print("=====================")


def build(cfg: Cfg) -> str:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Torch-TensorRT compilation.")

    try:
        import torch_tensorrt
    except Exception as exc:
        raise RuntimeError(
            "torch_tensorrt failed to import. Install a version that matches your "
            "torch version."
        ) from exc

    out_dir = os.path.join(cfg.MODEL_REPO_ROOT, cfg.TRT_MODEL_NAME, cfg.TRT_MODEL_VERSION)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, cfg.OUTPUT_FILENAME)

    print("Loading HF model/tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.MODEL_ID)
    hf_model = AutoModel.from_pretrained(cfg.MODEL_ID).eval().cuda()
    model = LabseWrapper(hf_model, cfg.FP16).eval().cuda()

    print("Preparing example inputs...")
    dummy_texts = ["test sentence"] * max(1, cfg.OPT_BATCH)
    enc = tokenizer(
        dummy_texts,
        padding="max_length",
        truncation=True,
        max_length=cfg.SEQ_LEN,
        return_tensors="pt",
    )

    input_dtype = torch_dtype_from_name(cfg.INPUT_DTYPE)
    input_ids = enc["input_ids"].to(device="cuda", dtype=input_dtype)
    attention_mask = enc["attention_mask"].to(device="cuda", dtype=input_dtype)

    trt_inputs = [
        torch_tensorrt.Input(
            min_shape=(cfg.MIN_BATCH, cfg.SEQ_LEN),
            opt_shape=(cfg.OPT_BATCH, cfg.SEQ_LEN),
            max_shape=(cfg.MAX_BATCH, cfg.SEQ_LEN),
            dtype=input_dtype,
        ),
        torch_tensorrt.Input(
            min_shape=(cfg.MIN_BATCH, cfg.SEQ_LEN),
            opt_shape=(cfg.OPT_BATCH, cfg.SEQ_LEN),
            max_shape=(cfg.MAX_BATCH, cfg.SEQ_LEN),
            dtype=input_dtype,
        ),
    ]

    enabled_precisions = {torch.float16} if cfg.FP16 else {torch.float32}

    print("Compiling with Torch-TensorRT...")
    trt_mod = torch_tensorrt.compile(
        model,
        inputs=trt_inputs,
        enabled_precisions=enabled_precisions,
        require_full_compilation=cfg.REQUIRE_FULL_COMPILATION,
    )

    print("Saving TorchScript module...")
    torch_tensorrt.save(
        trt_mod,
        out_path,
        inputs=[input_ids, attention_mask],
        output_format="torchscript",
    )

    print("DONE:", out_path)
    return out_path


def write_triton_config(cfg: Cfg, model_dir: str) -> None:
    input_dtype = config_dtype_from_name(cfg.INPUT_DTYPE)
    output_dtype = "TYPE_FP16" if cfg.FP16 else "TYPE_FP32"
    config_path = os.path.join(model_dir, "config.pbtxt")

    config_lines = [
        f'name: "{cfg.TRT_MODEL_NAME}"',
        'platform: "pytorch_libtorch"',
        f'default_model_filename: "{cfg.OUTPUT_FILENAME}"',
        "",
        f"max_batch_size: {cfg.MAX_BATCH}",
        "",
        "input [",
        "  {",
        '    name: "input_ids"',
        f"    data_type: {input_dtype}",
        f"    dims: [{cfg.SEQ_LEN}]",
        "  },",
        "  {",
        '    name: "attention_mask"',
        f"    data_type: {input_dtype}",
        f"    dims: [{cfg.SEQ_LEN}]",
        "  }",
        "]",
        "",
        "output [",
        "  {",
        f'    name: "{cfg.SENTENCE_OUTPUT_NAME}"',
        f"    data_type: {output_dtype}",
        f"    dims: [{cfg.EMBED_DIM}]",
        "  },",
        "  {",
        f'    name: "{cfg.TOKEN_OUTPUT_NAME}"',
        f"    data_type: {output_dtype}",
        f"    dims: [{cfg.SEQ_LEN}, {cfg.EMBED_DIM}]",
        "  }",
        "]",
    ]

    if cfg.INSTANCE_COUNT > 0:
        config_lines.extend(
            [
                "",
                "instance_group [",
                "  {",
                "    kind: KIND_GPU",
                f"    count: {cfg.INSTANCE_COUNT}",
                f"    gpus: [{cfg.GPU_IDS}]",
                "  }",
                "]",
            ]
        )

    if cfg.ENABLE_DYNAMIC_BATCHING:
        config_lines.extend(
            [
                "",
                "dynamic_batching {",
                f"  preferred_batch_size: [{cfg.PREFERRED_BATCH_SIZES}]",
                f"  max_queue_delay_microseconds: {cfg.MAX_QUEUE_DELAY_US}",
                "}",
            ]
        )

    config_lines.append("")

    with open(config_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(config_lines))
    print("Saved Triton config:", config_path)


def load_model_via_grpc(cfg: Cfg) -> None:
    try:
        import tritonclient.grpc as grpcclient
    except Exception as exc:
        raise RuntimeError("tritonclient[grpc] is required for MANAGE_TRITON=1.") from exc

    client = grpcclient.InferenceServerClient(url=cfg.TRITON_URL, verbose=False)
    if not client.is_server_live():
        raise RuntimeError(f"Triton server is not live at {cfg.TRITON_URL}")
    if not client.is_server_ready():
        raise RuntimeError(f"Triton server is not ready at {cfg.TRITON_URL}")

    try:
        client.load_model(cfg.TRT_MODEL_NAME)
        print(f"Loaded model on Triton via gRPC: {cfg.TRT_MODEL_NAME}")
    except Exception as exc:
        print(f"Warning: could not load model via gRPC: {exc}", file=sys.stderr)


def main() -> None:
    cfg = Cfg()
    validate_cfg(cfg)
    print_env(cfg)
    out_path = build(cfg)

    model_dir = os.path.dirname(out_path)
    if cfg.WRITE_CONFIG:
        write_triton_config(cfg, model_dir)
    if cfg.MANAGE_TRITON:
        load_model_via_grpc(cfg)


if __name__ == "__main__":
    main()
