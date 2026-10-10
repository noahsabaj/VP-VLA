# Copyright 2025 starVLA community. All rights reserved.
# Licensed under the MIT License, Version 1.0 (the "License"); 
# Implemented by [Jinhui YE / HKUST University] in [2025].

import logging
import socket
import argparse
from deployment.model_server.tools.websocket_policy_server import WebsocketPolicyServer
from starVLA.model.framework.base_framework import baseframework
import numpy as np
import torch, os


def main(args) -> None:
    # Example usage:
    # policy = YourPolicyClass()  # Replace with your actual policy class
    # server = WebsocketPolicyServer(policy, host="localhost", port=10091)
    # server.serve_forever()

    vla = baseframework.from_pretrained( # TODO should auto detect framework from model path
        args.ckpt_path,
    )

    if args.use_bf16: # False
        vla = vla.to(torch.bfloat16)
    vla = vla.to("cuda").eval()

    if args.attn:  # e.g. sdpa: about 10% faster than flash_attention_2 for one short request
        hf_config = vla.qwen_vl_interface.model.config
        for config in (hf_config, getattr(hf_config, "text_config", None), getattr(hf_config, "vision_config", None)):
            if config is not None:
                config._attn_implementation = args.attn
        logging.info("Attention implementation: %s", args.attn)

    if args.compile:  # fuse the elementwise work of the VLM layers (about 25% less GPU time per call)
        torch._dynamo.config.cache_size_limit = max(torch._dynamo.config.cache_size_limit, 64)
        for name, layers in vla.qwen_vl_interface.model.named_modules():
            if isinstance(layers, torch.nn.ModuleList) and name.endswith(("layers", "blocks")):
                for layer in layers:
                    layer.forward = torch.compile(layer.forward)
        # Compile now, before clients connect. Two prompt lengths, so the second compile makes the length dynamic.
        image = np.zeros((224, 224, 3), dtype=np.uint8)
        for lang in ("warm up", "warm up the compiled layers with a longer prompt"):
            vla.predict_action(examples=[{"image": [image, image], "lang": lang}], do_sample=False, use_ddim=True, num_ddim_steps=10)
        logging.info("Compiled the VLM layers")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)

    # start websocket server
    server = WebsocketPolicyServer(
        policy=vla,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
        metadata={"env": "simpler_env"},
    )
    logging.info("server running ...")
    server.serve_forever()


def build_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct")
    parser.add_argument("--port", type=int, default=10093)
    parser.add_argument("--use_bf16", action="store_true")
    parser.add_argument("--attn", type=str, default=None, choices=["flash_attention_2", "sdpa"],
                        help="Override the VLM attention implementation (default: as built)")
    parser.add_argument("--compile", action="store_true",
                        help="torch.compile the VLM layers at startup (about 25%% less GPU time; actions change slightly)")
    parser.add_argument("--idle_timeout" , type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    return parser


def start_debugpy_once():
    """start debugpy once"""
    import debugpy
    if getattr(start_debugpy_once, "_started", False):
        return
    debugpy.listen(("0.0.0.0", 10095))
    print("🔍 Waiting for VSCode attach on 0.0.0.0:10095 ...")
    debugpy.wait_for_client()
    start_debugpy_once._started = True


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    if os.getenv("DEBUG", False):
        print("🔍 DEBUGPY is enabled")
        start_debugpy_once()
    main(args)