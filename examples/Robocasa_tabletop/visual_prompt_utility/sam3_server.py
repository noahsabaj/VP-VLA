# SAM3 Segmentation Server
# Provides segmentation masks via websocket for the evaluation pipeline

import asyncio
import logging
import traceback
import time
import argparse
import collections
import hashlib
import socket

import numpy as np
import torch
import websockets.asyncio.server
import websockets.frames

from msgpack_utils import packb, unpackb


def _to_float32(x):
    """Cast the floating tensors in a model output (or tuple of them) to fp32."""
    if torch.is_tensor(x):
        return x.float() if x.is_floating_point() else x
    if isinstance(x, (tuple, list)):
        return type(x)(_to_float32(v) for v in x)
    if hasattr(x, "keys"):
        for k in list(x.keys()):
            x[k] = _to_float32(x[k])
    return x


class SAM3Model:
    """Wrapper for SAM3 model to handle segmentation requests."""

    def __init__(self, model_path: str, device: str = "cuda", fp16: bool = False):
        from transformers import Sam3Processor, Sam3Model

        self.device = device

        logging.info(f"Loading SAM3 model from {model_path}")
        self.model = Sam3Model.from_pretrained(model_path).to(device)
        self.processor = Sam3Processor.from_pretrained(model_path)
        self.model.eval()
        # fp16: run the image encoder in fp16 and the DETR encoder under fp16 autocast (together
        # about 2/3 of the cost), each handing fp32 back, so the decoder and the box coordinates
        # stay fp32. Masks and boxes can differ by about a pixel from all-fp32.
        self._fp16 = fp16
        if fp16:
            self.model.vision_encoder.half()
            detr_encoder_forward = self.model.detr_encoder.forward

            def detr_encoder_fp16(*args, **kwargs):
                with torch.autocast("cuda", dtype=torch.float16):
                    return _to_float32(detr_encoder_forward(*args, **kwargs))

            self.model.detr_encoder.forward = detr_encoder_fp16
        # The eval segments each frame twice (object, then target): keep the image features of the
        # last few frames so the second prompt skips the image encoder, also when several eval
        # clients share this server. The prompts repeat from step to step, so keep their text
        # features too. Same computation, same output.
        self._vision_cache = collections.OrderedDict()
        self._text_cache = collections.OrderedDict()
        logging.info(f"SAM3 model loaded successfully (fp16: {fp16})")

    @staticmethod
    def _cached(cache: collections.OrderedDict, key, make, keep: int):
        if key in cache:
            cache.move_to_end(key)
        else:
            cache[key] = make()
            while len(cache) > keep:
                cache.popitem(last=False)
        return cache[key]

    def _vision(self, image: np.ndarray):
        """Image features and original size of a frame."""
        from PIL import Image

        def make():
            inputs = self.processor(images=Image.fromarray(image).convert("RGB"), return_tensors="pt").to(self.device)
            pixel_values = inputs["pixel_values"]
            if not self._fp16:
                return self.model.get_vision_features(pixel_values=pixel_values), inputs["original_sizes"].tolist()
            features = self.model.get_vision_features(pixel_values=pixel_values.half())
            return _to_float32(features), inputs["original_sizes"].tolist()

        key = (image.shape, hashlib.blake2b(np.ascontiguousarray(image).tobytes(), digest_size=16).digest())
        return self._cached(self._vision_cache, key, make, keep=4)

    def _text(self, text_prompt: str):
        """Attention mask and text features of a prompt."""
        def make():
            t = self.processor(text=text_prompt, return_tensors="pt").to(self.device)
            features = self.model.get_text_features(input_ids=t["input_ids"], attention_mask=t["attention_mask"], return_dict=True)
            return t["attention_mask"], features

        return self._cached(self._text_cache, text_prompt, make, keep=64)
    
    def segment(
        self, 
        image: np.ndarray, 
        text_prompt: str,
        threshold: float = 0.5,
        mask_threshold: float = 0.5,
    ) -> dict:
        """
        Perform segmentation on the image using the text prompt.
        
        Args:
            image: np.ndarray of shape (H, W, 3), uint8 RGB image
            text_prompt: str, text description of object to segment
            threshold: float, confidence threshold for instance detection
            mask_threshold: float, threshold for mask binarization
            
        Returns:
            dict with:
                - masks: np.ndarray of shape (N, H, W), bool masks
                - scores: np.ndarray of shape (N,), confidence scores
                - num_masks: int, number of detected masks
        """
        # Run inference
        with torch.no_grad():
            vision_embeds, original_sizes = self._vision(image)
            attention_mask, text_embeds = self._text(text_prompt)
            outputs = self.model(vision_embeds=vision_embeds, text_embeds=text_embeds, attention_mask=attention_mask)

        # Post-process results
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=threshold,
            mask_threshold=mask_threshold,
            target_sizes=original_sizes
        )[0]
        
        masks = results.get("masks", torch.zeros((0, image.shape[0], image.shape[1]), dtype=torch.bool))
        scores = results.get("scores", torch.zeros((0,)))
        boxes = results.get("boxes", torch.zeros((0, 4), dtype=torch.float32))
        
        # Convert to numpy
        if isinstance(masks, torch.Tensor):
            masks = masks.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        
        return {
            "masks": masks.astype(np.uint8),  # (N, H, W)
            "boxes": boxes.astype(np.float32),  # (N, 4) in format (x1, y1, x2, y2)
            "scores": scores.astype(np.float32),  # (N,)
            "num_masks": len(masks),
        }


class SAM3Server:
    """Websocket server for SAM3 segmentation."""
    
    def __init__(
        self,
        model: SAM3Model,
        host: str = "0.0.0.0",
        port: int = 10094,
        idle_timeout: int = -1,
    ):
        self._model = model
        self._host = host
        self._port = port
        self._idle_timeout = idle_timeout
        self._last_active = time.time()
        self._metadata = {"service": "sam3_segmentation"}
        logging.getLogger("websockets.server").setLevel(logging.INFO)
    
    def serve_forever(self) -> None:
        asyncio.run(self.run())
    
    async def run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            logging.info(f"SAM3 server listening on {self._host}:{self._port}")
            if self._idle_timeout > 0:
                await self._idle_watchdog(server)
            else:
                await server.serve_forever()
    
    async def _idle_watchdog(self, server):
        while True:
            await asyncio.sleep(5)
            if time.time() - self._last_active > self._idle_timeout:
                logging.info(f"Idle timeout ({self._idle_timeout}s) reached, shutting down server.")
                server.close()
                await server.wait_closed()
                break
    
    async def _handler(self, websocket: websockets.asyncio.server.ServerConnection):
        logging.info(f"Connection from {websocket.remote_address} opened")
        
        # Send metadata on connection
        await websocket.send(packb(self._metadata))
        
        while True:
            try:
                msg = unpackb(await websocket.recv())
                self._last_active = time.time()
                ret = self._route_message(msg)
                await websocket.send(packb(ret))
            except websockets.ConnectionClosed:
                logging.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error.",
                )
                raise
    
    def _route_message(self, msg: dict) -> dict:
        req_id = msg.get("request_id", "default")
        mtype = msg.get("type", "segment")
        
        if mtype == "ping":
            return {"status": "ok", "ok": True, "type": "ping", "request_id": req_id}
        
        elif mtype == "segment":
            try:
                image = msg.get("image")  # np.ndarray (H, W, 3), uint8
                text_prompt = msg.get("text_prompt", "")
                threshold = msg.get("threshold", 0.5)
                mask_threshold = msg.get("mask_threshold", 0.5)
                
                if image is None:
                    return {
                        "status": "error",
                        "ok": False,
                        "request_id": req_id,
                        "error": {"message": "Missing 'image' in request"},
                    }
                
                result = self._model.segment(
                    image, 
                    text_prompt, 
                    threshold=threshold, 
                    mask_threshold=mask_threshold,
                )
                
                return {
                    "status": "ok",
                    "ok": True,
                    "type": "segment_result",
                    "request_id": req_id,
                    "data": result,
                }
            except Exception as e:
                logging.exception("SAM3 segmentation error (request_id=%s)", req_id)
                return {
                    "status": "error",
                    "ok": False,
                    "type": "segment_result",
                    "request_id": req_id,
                    "error": {"message": str(e)},
                }
        
        else:
            return {
                "status": "error",
                "ok": False,
                "type": "unknown",
                "request_id": req_id,
                "error": {"message": f"Unsupported message type '{mtype}'"},
            }


def build_argparser():
    parser = argparse.ArgumentParser(description="SAM3 Segmentation Server")
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="playground/Pretrained_models/sam3",
        help="Path to SAM3 model checkpoint"
    )
    parser.add_argument("--port", type=int, default=10094, help="Server port")
    parser.add_argument("--idle-timeout", type=int, default=1800, help="Idle timeout in seconds, -1 means never close")
    parser.add_argument("--tf32", action="store_true",
                        help="TF32 matmuls (about 1.5x faster on Ampere); masks can differ by a few pixels from fp32")
    parser.add_argument("--fp16", action="store_true",
                        help="Image encoder and DETR encoder in fp16 (about 2x faster); masks and boxes can differ by about a pixel from fp32")
    return parser


def main():
    logging.basicConfig(level=logging.INFO, force=True)
    parser = build_argparser()
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    logging.info(f"TF32 matmuls: {args.tf32}")

    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info(f"Starting SAM3 server (host: {hostname}, ip: {local_ip})")
    
    # Load model
    model = SAM3Model(
        model_path=args.model_path,
        device=device,
        fp16=args.fp16,
    )
    
    # Start server
    server = SAM3Server(
        model=model,
        host="0.0.0.0",
        port=args.port,
        idle_timeout=args.idle_timeout,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()


