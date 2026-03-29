# 9-train_reflection_grpo.py  —— GRPO (Group Relative Policy Optimization)
# 说明：在保留原有工程约定（路径/超参/量化/LoRA/强制1号卡）的基础上，
#       将 DAPO（优势加权 SFT）改为 GRPO（组内相对优化）。
#       具体做法：
#       1) 以 qid 为“组”，每个 batch 只包含同一 qid 的若干候选（GroupBatchSampler）。
#       2) 对组内样本，按 reward 形成有序成对 (i, j)，当 r_i > r_j 时优化：
#            L_pair = -log σ((lp_i - lp_j) / τ)
#          其中 lp_* 为“response 段平均 logprob（给定已生成的 response）”。
#       3) 额外加入一个轻量 KL 正则：KL(new || ref)（ref 为冻结参考策略：base+LoRA init），
#          仅在 response 段上计算，系数 KL_COEF 很小（例如 0.02）。
#       4) 仍使用 4bit 量化 + LoRA，可在单卡上训练。
#
# 仍然读取 reflection_data_v2/v1.json，仍然输出 gemma-2-2b-reflection-final
# 仍然用 gemma-2-2b-lora-init 作为 LoRA 初始化；强制使用物理 1 号卡

import os, re, json, math, random
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Iterator, Tuple

# ==== GPU 卡选择 & 稳定内存 ====
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model

# ======= 硬编码路径/超参（与现有管线保持一致）=======
BASE_MODEL       = "google/gemma-2-2b-it"
PEFT_INIT_DIR    = "gemma-2-2b-lora-init"         # 第 8 步产物（LoRA 初值）
OUTPUT_DIR       = "gemma-2-2b-reflection-final"  # 第 10/11 步读取的目录名保持不变
INPUT_CANDIDATES = ["reflection_data_v2.json", "reflection_data_v1.json"]

# 重要长度策略：为响应预留预算，避免响应尾部被截断
MAX_LENGTH_PROMPT   = 128
MAX_LENGTH_RESPONSE = 256
MAX_LENGTH_COMBINED = MAX_LENGTH_PROMPT + MAX_LENGTH_RESPONSE  # 384

# 批/优化
BATCH_SIZE   = 16  # 你已更正为 16
GRAD_ACCUM   = 1
EPOCHS       = 1
LR           = 1e-5
WARMUP_STEPS = 100
SEED         = 42

# GRPO 相关
TAU      = 1.0    # pairwise 温度
KL_COEF  = 0.02   # 轻量 KL 系数（new || ref），仅作用在 response 段

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def set_seed(s=42):
    random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

# ============== 工具函数 ==============

def _strip_code_fence(s: str) -> str:
    s = str(s or "").strip()
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.I)
    s = re.sub(r"\s*```$", "", s)
    return s.strip()

@dataclass
class Sample:
    qid: str
    prompt: str
    response: str
    reward: float

class ReflDataset(Dataset):
    def __init__(self, path: str):
        raw = json.load(open(path, "r", encoding="utf-8"))
        self.items: List[Sample] = []
        for it in raw:
            qid = str(it.get("qid") or it.get("query_id") or "")
            p = it.get("reflection_prompt") or it.get("prompt") or ""
            r = _strip_code_fence(it.get("reflection_output", it.get("output","")))
            rew = float(it.get("reward", it.get("delta_f1", 0.0)))
            if p and r:
                self.items.append(Sample(qid=qid, prompt=p, response=r, reward=rew))
        # 建立按 qid 的倒排，供分组采样器使用
        self.qid_to_indices: Dict[str, List[int]] = {}
        for idx, it in enumerate(self.items):
            self.qid_to_indices.setdefault(it.qid, []).append(idx)

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

class GroupBatchSampler(Sampler[List[int]]):
    """按 qid 分组采样：每个 batch 只包含同一 qid 的若干候选。
    若某 qid 的候选数 > BATCH_SIZE，则切分成多个 batch；不足的情况下直接用当前数量（不做跨 qid 填充，保证纯组内比较）。
    """
    def __init__(self, ds: ReflDataset, batch_size: int, shuffle: bool = True):
        self.ds = ds
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.qids = list(ds.qid_to_indices.keys())

    def __iter__(self) -> Iterator[List[int]]:
        qids = self.qids[:]
        if self.shuffle:
            random.shuffle(qids)
        for q in qids:
            idxs = self.ds.qid_to_indices[q][:]
            if self.shuffle:
                random.shuffle(idxs)
            # 按 batch_size 切块
            for b in range(0, len(idxs), self.batch_size):
                yield idxs[b:b+self.batch_size]

    def __len__(self) -> int:
        # 估计 batch 数：每个 qid 的 ceil(n/B)
        total = 0
        for q in self.qids:
            n = len(self.ds.qid_to_indices[q])
            total += max(1, math.ceil(n / self.batch_size))
        return total

class Collator:
    def __init__(self, tok: AutoTokenizer, max_prompt: int, max_resp: int, max_combined: int):
        self.tok = tok
        self.max_prompt = max_prompt
        self.max_resp = max_resp
        self.max_combined = max_combined

    def __call__(self, batch: List[Sample]) -> Dict[str, Any]:
        prompts   = [b.prompt for b in batch]
        responses = [b.response for b in batch]
        rewards   = torch.tensor([b.reward for b in batch], dtype=torch.float32)
        qids      = [b.qid for b in batch]

        # 先分别限制 prompt/response 的最大长度，再拼接，最后再兜底截断到 max_combined
        enc_p = self.tok(prompts, padding=False, truncation=True, max_length=self.max_prompt, return_tensors=None)
        enc_r = self.tok(responses, padding=False, truncation=True, max_length=self.max_resp,   return_tensors=None)

        input_ids_list, attn_list, labels_list, resp_mask_list = [], [], [], []
        pad_id = self.tok.pad_token_id

        for i in range(len(prompts)):
            p_ids = torch.tensor(enc_p["input_ids"][i], dtype=torch.long)
            r_ids = torch.tensor(enc_r["input_ids"][i], dtype=torch.long)
            if len(r_ids) == 0:
                # 保障至少有一个 token（eos）
                r_ids = torch.tensor([self.tok.eos_token_id], dtype=torch.long)
            # 去掉 response 的 BOS（若存在）
            if len(r_ids) > 0 and r_ids[0] == self.tok.bos_token_id:
                r_ids = r_ids[1:]
            ids = torch.cat([p_ids, r_ids])
            # 兜底截断到 max_combined
            ids = ids[:self.max_combined]

            # attention、labels、resp_mask
            am = torch.ones_like(ids)
            lab = torch.full_like(ids, -100)
            # response 段起始位置
            resp_start = min(len(p_ids), len(ids))
            lab[resp_start:] = ids[resp_start:]  # 只监督 response 段
            rm = torch.zeros_like(ids, dtype=torch.bool)
            rm[resp_start:] = True

            input_ids_list.append(ids)
            attn_list.append(am)
            labels_list.append(lab)
            resp_mask_list.append(rm)

        # pad 到同长度
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        attn      = torch.nn.utils.rnn.pad_sequence(attn_list,      batch_first=True, padding_value=0)
        labels    = torch.nn.utils.rnn.pad_sequence(labels_list,    batch_first=True, padding_value=-100)
        resp_mask = torch.nn.utils.rnn.pad_sequence(resp_mask_list, batch_first=True, padding_value=0)

        return {
            "input_ids": input_ids,
            "attention_mask": attn,
            "labels": labels,
            "resp_mask": resp_mask,
            "rewards": rewards,
            "qids": qids,
        }

# ============== 模型加载 ==============

def load_model_and_tokenizer():
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    # 训练模型（可训练 LoRA）
    train_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        low_cpu_mem_usage=True, attn_implementation="eager"
    )
    train_model.config.use_cache = False
    train_model.gradient_checkpointing_enable()
    torch.backends.cuda.matmul.allow_tf32 = True

    # 参考模型（冻结）：与 train_model 结构一致，加载同样的 LoRA 初始化，以便 KL(new||ref)
    ref_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb, device_map="auto",
        low_cpu_mem_usage=True, attn_implementation="eager"
    )
    ref_model.config.use_cache = False

    # 训练模型挂 LoRA
    if os.path.isdir(PEFT_INIT_DIR):
        train_model = PeftModel.from_pretrained(train_model, PEFT_INIT_DIR, is_trainable=True)
    else:
        lora = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                          target_modules=["q_proj","v_proj"])  # 如需更强可扩到 k/o
        train_model = get_peft_model(train_model, lora)
        train_model.print_trainable_parameters()

    # 参考模型也挂相同的 LoRA 权重，但冻结
    if os.path.isdir(PEFT_INIT_DIR):
        ref_model = PeftModel.from_pretrained(ref_model, PEFT_INIT_DIR, is_trainable=False)
    else:
        # 若没有 init 目录，就让参考模型维持 base（无 LoRA），也可；KL 仍然有意义
        pass
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model.eval()

    return train_model, ref_model, tok

# ============== 计算函数 ==============

def avg_logprob_on_response(logits: torch.Tensor, labels: torch.Tensor, resp_mask: torch.Tensor) -> torch.Tensor:
    """给定模型 logits 和标签，在 response 段上计算 per-sample 平均 logprob。
    logits: [B, T, V]; labels: [B, T]（response 段为 token id，prompt 段为 -100）；resp_mask: [B, T] bool
    返回：lp_avg [B]
    """
    logp = torch.log_softmax(logits, dim=-1)  # [B, T, V]
    tgt = labels.clone()
    tgt[tgt < 0] = 0  # 占位，随后通过 mask 过滤
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, T]
    tok_lp = tok_lp * resp_mask
    lengths = resp_mask.sum(dim=1).clamp(min=1)
    lp_avg = tok_lp.sum(dim=1) / lengths
    return lp_avg

@torch.no_grad()
def kl_on_response(new_logits: torch.Tensor, ref_logits: torch.Tensor, resp_mask: torch.Tensor) -> torch.Tensor:
    """在 response 段 token 上计算 KL(new || ref) 的 per-sample 平均值。返回 [B]。"""
    new_logp = torch.log_softmax(new_logits, dim=-1)  # [B, T, V]
    ref_logp = torch.log_softmax(ref_logits, dim=-1)
    new_p = new_logp.exp()
    # KL = sum p_new * (log p_new - log p_ref)
    kl_tok = (new_p * (new_logp - ref_logp)).sum(dim=-1)  # [B, T]
    kl_tok = kl_tok * resp_mask
    lengths = resp_mask.sum(dim=1).clamp(min=1)
    kl_avg = kl_tok.sum(dim=1) / lengths
    return kl_avg

# 组内成对索引

def make_pairs_within_group(rewards: torch.Tensor) -> List[Tuple[int, int]]:
    """给定同一组（batch）的 reward，返回所有严格有序对 (i, j) 使 r_i > r_j。
    若全相等或只有 1 条，返回空列表。"""
    pairs: List[Tuple[int, int]] = []
    B = rewards.size(0)
    for i in range(B):
        for j in range(B):
            if rewards[i] > rewards[j]:
                pairs.append((i, j))
    return pairs

# ============== 训练主流程 ==============

def train():
    set_seed(SEED)
    data_path = next((p for p in INPUT_CANDIDATES if os.path.exists(p)), None)
    if not data_path:
        raise FileNotFoundError("reflection_data_v2.json / v1.json not found.")

    model, ref_model, tok = load_model_and_tokenizer()
    collate = Collator(tok, MAX_LENGTH_PROMPT, MAX_LENGTH_RESPONSE, MAX_LENGTH_COMBINED)
    ds = ReflDataset(data_path)

    # 关键：按 qid 分组采样，保证每个 batch 都是同一问题的多个候选
    sampler = GroupBatchSampler(ds, batch_size=BATCH_SIZE, shuffle=True)
    dl = DataLoader(ds, batch_sampler=sampler, collate_fn=collate)

    optim = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

    # 正确的 total_steps 计算（含 BATCH_SIZE / 累积）
    steps_per_epoch = len(dl) // max(1, GRAD_ACCUM)
    if len(dl) % max(1, GRAD_ACCUM) != 0:
        steps_per_epoch += 1
    total_steps = max(1, EPOCHS * steps_per_epoch)
    sched = get_cosine_schedule_with_warmup(optim, WARMUP_STEPS, total_steps)

    print(f"Dataset: {len(ds)} | epochs={EPOCHS} | batch(B={BATCH_SIZE}) | steps≈{total_steps}")

    model.train()
    gstep = 0

    for ep in range(EPOCHS):
        for step, batch in enumerate(dl):
            ids   = batch["input_ids"].to(DEVICE)
            attn  = batch["attention_mask"].to(DEVICE)
            labs  = batch["labels"].to(DEVICE)
            rmask = batch["resp_mask"].to(DEVICE)
            rwd   = batch["rewards"].to(DEVICE)  # [B]

            # 前向：训练模型
            out_new = model(input_ids=ids, attention_mask=attn)
            lp_avg_new = avg_logprob_on_response(out_new.logits, labs, rmask)  # [B]

            # 前向：参考模型（冻结）
            with torch.no_grad():
                out_ref = ref_model(input_ids=ids, attention_mask=attn)
            # KL(new||ref) on response
            kl_avg = kl_on_response(out_new.logits, out_ref.logits, rmask)  # [B]

            # 组内成对
            pairs = make_pairs_within_group(rwd)
            if len(pairs) == 0:
                # 若该组无法成对（例如只有 1 条或全相等），退回到“去中心化加权”以避免空梯度
                lp_centered = lp_avg_new - lp_avg_new.mean()
                loss_pair = -(lp_centered).mean() * 0.0  # 显式 0，不影响总损失
            else:
                diffs = []
                for i, j in pairs:
                    diffs.append((lp_avg_new[i] - lp_avg_new[j]) / TAU)
                diffs = torch.stack(diffs)  # [P]
                loss_pair = torch.nn.functional.softplus(-diffs).mean()  # -log σ(x) = softplus(-x)

            loss_kl = KL_COEF * kl_avg.mean()
            loss = loss_pair + loss_kl

            loss.backward()

            if (step + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optim.step(); sched.step(); optim.zero_grad()
                gstep += 1
                if gstep % 10 == 0:
                    print(f"step {gstep:05d} | loss={loss.item():.4f} | pair={loss_pair.item():.4f} | kl={loss_kl.item():.4f} | pairs={len(pairs)}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # 始终保存 PEFT 适配器；不要回退保存 base_model 以免覆盖
    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    print(f"✅ GRPO adapter saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    train()
