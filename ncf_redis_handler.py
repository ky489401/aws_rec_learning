# ncf_redis_handler.py
import os
import torch
from ts.torch_handler.base_handler import BaseHandler
import redis

TOP_K = 5
MAX_RECENT = 5       # how many recent items to pull from Redis for scoring

class NCFRedisHandler(BaseHandler):
    """
    TorchServe handler for NCF model.
    Expects input JSON like {"user_id": ..., "item_id": ...}
    and returns the predicted probability.
    """

    def initialize(self, ctx):
        super().initialize(ctx)
        self.device = next(self.model.parameters()).device

        # --- Redis connection ---------------------------------------------
        self.redis = redis.Redis(
            host=os.getenv("REDIS_HOST", "localhost"),
            port=int(os.getenv("REDIS_PORT", "6379")),
            password=os.getenv("REDIS_AUTH", ""),
            decode_responses=True,              # return str not bytes
            socket_timeout=2,
            socket_connect_timeout=2,
        )

    def preprocess(self, requests):
        payload = requests[0]["body"]
        user_id = int(payload["user_id"])
        item_id = int(payload["item_id"])
        user = torch.tensor([user_id], dtype=torch.long).to(self.device)
        item = torch.tensor([item_id], dtype=torch.long).to(self.device)
        self._fetch_recent_items(user_id)
        return user, item

    def _fetch_recent_items(self, user_id: int):
        """Return a list[int] of at most MAX_RECENT item_ids the user recently touched.
        Fallback to an all‑negative list if Redis is down or empty."""
        try:
            key = f"user_browsed:{user_id}"
            recent = self.redis.lrange(key, 0, MAX_RECENT - 1) or []
            recent = list(map(int, recent))
            print("#####recent####", recent)
        except Exception:
            recent = []
            print("#####recent####", recent)

        # pad with -1 so tensor has fixed length
        if len(recent) < MAX_RECENT:
            recent += [-1] * (MAX_RECENT - len(recent))

        self.recent = recent[:MAX_RECENT]
        return self.recent

    def inference(self, inputs):
        users, items = inputs
        logits = self.model(users, items)
        probs = torch.sigmoid(logits)
        return probs

    def postprocess(self, scores):
        """
        `scores` can be 0-D when only one (user,item) was fed through the model.
        Make it 1-D, then pick top-k.
        """
        scores = scores.flatten()                 # now 1-D whatever the input
        k = min(TOP_K, scores.numel())
        top_idx = torch.topk(scores, k=k).indices
        return [{"recommendations": top_idx.cpu().tolist(), "recent":self.recent}]