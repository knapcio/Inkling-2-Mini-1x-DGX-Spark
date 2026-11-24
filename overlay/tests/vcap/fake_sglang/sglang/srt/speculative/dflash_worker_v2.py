"""CPU stand-in for the DFLASH worker (sglang.srt.speculative.dflash_worker_v2): the hooked methods' names,
signatures and call order in forward_batch_generation, a toy selector walk (the image's CPU torch path), a toy
target with Inkling2MoE layers run in TARGET_VERIFY mode, and the stock greedy accept. Every step's reference
values (walk inputs/outputs, accept lengths, expert sets per MoE visit) are kept in ``self.history`` so a test can
check what the capture wrote."""

from types import SimpleNamespace

import torch
import torch.nn.functional as F
from torch import nn

from sglang.srt.models import dflash as mdf
from sglang.srt.models.inkling_common import inkling2_moe

SIMULATE_ACC_LEN = 0.0
V, H, K = 64, 16, 16


class _Mode:
    def __init__(self, verify):
        self._verify = verify

    def is_target_verify(self):
        return self._verify


class CandidateSelector(nn.Module):
    def __init__(self):
        super().__init__()
        self.top_k = K

    def sample_path(self, *, candidate_ids, scores, uniforms, temperatures, greedy_mask):
        temps = temperatures.view(-1, 1)
        initial_probs = torch.softmax(scores[:, 0, 0].float() / temps, dim=-1)
        initial = uniforms[:, :1].ge(initial_probs.cumsum(dim=-1)).sum(dim=-1).clamp_max(K - 1)
        transition = torch.softmax(scores[:, 1:].float() / temps[:, :, None, None], dim=-1)
        local = uniforms[:, 1:, None, None].ge(transition.cumsum(dim=-1)).sum(dim=-1).clamp_max(K - 1)
        initial = torch.where(greedy_mask, scores[:, 0, 0].argmax(dim=-1), initial)
        local = torch.where(greedy_mask[:, None, None], scores[:, 1:].argmax(dim=-1), local)
        index = initial
        path = [index]
        for edge in range(scores.shape[1] - 1):
            index = local[:, edge].gather(-1, index[:, None])[:, 0]
            path.append(index)
        path = torch.stack(path, dim=1)
        tokens = candidate_ids.gather(-1, path.unsqueeze(-1))[:, :, 0]
        realized = transition.gather(2, path[:, :-1, None, None].expand(-1, -1, 1, K))[:, :, 0]
        q_rows = torch.cat((initial_probs.unsqueeze(1), realized), dim=1)
        q_rows = torch.where(greedy_mask[:, None, None], F.one_hot(path, K).float(), q_rows)
        return tokens, q_rows


class _SelectorDraftSampler:
    def __init__(self, selector):
        self.selector = selector

    def __call__(self, candidate_ids, scores, uniforms, temperatures, greedy_mask):
        return self.selector.sample_path(
            candidate_ids=candidate_ids, scores=scores, uniforms=uniforms, temperatures=temperatures,
            greedy_mask=greedy_mask,
        )


class ToyTarget(nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(7)
        self.emb = torch.randn(V, H, generator=g, dtype=torch.float64)
        self.unemb = torch.randn(V, H, generator=g, dtype=torch.float64)
        self.layers = nn.ModuleList(inkling2_moe.Inkling2MoE(H, 16, 2, s) for s in (1, 2))
        self.schedule = [0, 1, 1]  # layer 1 looped: 3 MoE visits per forward

    def verify(self, ids):
        x = self.emb[ids]
        fb = SimpleNamespace(forward_mode=_Mode(True))
        for li in self.schedule:
            x = x + self.layers[li](x, fb)
        return x @ self.unemb.T


class _DraftRunner:
    def __init__(self, worker):
        self.worker = worker

    def forward(self, forward_batch):
        return SimpleNamespace(logits_output=None, can_run_graph=True, forward_batch=forward_batch)


class _TargetWorker:
    def __init__(self, model):
        self.model_runner = SimpleNamespace(model=model)

    def forward_batch_generation(self, *, batch, forward_batch, is_verify, skip_attn_backend_init=None):
        return SimpleNamespace(logits_output=SimpleNamespace(next_token_logits=self.model_runner.model.verify(
            forward_batch.reshape(-1)).float()))


class DFlashWorkerV2:
    _hosts_draft = True

    def __init__(self, *, seed=0, accept_bias=40.0):
        self.block_size = 8
        self.device = torch.device("cpu")
        self.selector = CandidateSelector()
        self.draft_model = mdf.DFlash2DraftModel(V, H, K)
        self.lilicorr = None
        self._is_domino = False
        self._verify_planner = None
        self._megakernel_accept = None
        self._megakernel_sampled = None
        self._block_verification = False
        # every Inkling target has a LogitsProcessor; at TP1 it never defers the gather
        self._target_logits_processor = SimpleNamespace(do_tensor_parallel_all_gather=False,
                                                        do_tensor_parallel_all_gather_dp_attn=False)
        self.uncapped_accept = None
        self.model = ToyTarget()
        self._target_worker = _TargetWorker(self.model)
        self.target_worker = self._target_worker
        self.draft_model_runner = _DraftRunner(self)
        self._draft_sampler = _SelectorDraftSampler(self.selector)
        self.gen = torch.Generator().manual_seed(seed)
        self.accept_bias = accept_bias
        self.history = []
        self.skip_accept_steps = set()  # steps that leave through a "megakernel" path (no _accept_block)
        self._step = 0

    # --- hooked methods (image signatures) -------------------------------------------------------
    def _stage_draft_sampler(self, *, bs, sampling_info):
        self._staged = (bs, sampling_info)

    def _plan_verify(self, *, batch, draft_input, prefix_lens):
        if self._verify_planner is None:
            return None, None, None
        raise NotImplementedError

    def _propose_selector_block(self, *, draft_logits_output, bs, lm_head, anchor_token_ids, sampling_info):
        return self.selector.sample_path(candidate_ids=None, scores=None, uniforms=None, temperatures=None,
                                         greedy_mask=None)

    def _accept_block(self, *, candidates, next_token_logits, sampling_info, draft_input, prefix_lens, bs,
                      target_predict=None, cutoff_verify_lens=None, verify_shard=None):
        cap_trim_lens = None
        if target_predict is None:
            target_predict = torch.argmax(next_token_logits, dim=-1)
        target_predict = target_predict.view(bs, int(self.block_size))
        match = (candidates[:, 1:] == target_predict[:, :-1]).to(torch.int64)
        accept_len = match.cumprod(dim=1).sum(dim=1).to(torch.int32)
        self.uncapped_accept = accept_len.clone()
        if cutoff_verify_lens is not None:  # the stock AcceptGreedy cutoff: at most L - 1 drafts, bonus from row
            capped = torch.minimum(accept_len, (cutoff_verify_lens[:bs] - 1).to(accept_len.dtype)).clamp_(min=0)
            cap_trim_lens = accept_len - capped
            accept_len = capped
        bonus = target_predict[torch.arange(bs), accept_len.long()]
        commit_lens = accept_len + 1
        out_tokens = candidates.clone()
        new_seq_lens = prefix_lens + commit_lens.to(prefix_lens.dtype)
        return accept_len, commit_lens, bonus, out_tokens, new_seq_lens, cap_trim_lens

    def _selector_sampling_accept(self, *, candidates, next_token_logits, candidate_ids, q_rows, sampling_info,
                                  draft_input, cutoff_verify_lens=None):
        """Image shape only (the toy loop is greedy); records the q it was handed."""
        self.last_sampled_q = q_rows
        draft_probs = torch.zeros(candidates.shape[0], candidates.shape[1] - 1, V)
        draft_probs.scatter_(-1, candidate_ids, q_rows.float())
        out = dict(block_verification=self._block_verification)
        draft_probs.scatter_(-1, candidate_ids, 0.0)
        return out

    # --- one decode step, in the image's order ----------------------------------------------------
    def forward_batch_generation(self, batch):
        bs = len(batch.seq_lens)
        gamma, g = self.block_size - 1, self.gen
        prefix_lens = batch.seq_lens.clone()
        self._stage_draft_sampler(bs=bs, sampling_info=batch.sampling_info)
        forward_batch = None
        draft_out = self.draft_model_runner.forward(forward_batch)
        hidden = torch.randn(bs * gamma, H, generator=g)
        with torch.no_grad():
            candidate_ids, unary = self.draft_model.compute_candidates(hidden)
            plane = mdf._project_candidate_logits.__wrapped__(hidden, self.draft_model.lm_head, num_org=V) \
                if hasattr(mdf._project_candidate_logits, "__wrapped__") else \
                mdf._project_candidate_logits(hidden, self.draft_model.lm_head, num_org=V)
        candidate_ids = candidate_ids.view(bs, gamma, K)
        unary = unary.view(bs, gamma, K)
        scores = unary[:, :, None, :] + torch.randn(bs, gamma, K, K, generator=g) * 2.0
        temps = batch.sampling_info.temperatures.view(-1).float().clamp_min(1e-5)
        greedy = batch.sampling_info.temperatures.view(-1) <= 0
        uniforms = torch.rand(bs, gamma, generator=g)
        tokens, q_rows = self._draft_sampler(candidate_ids, scores, uniforms, temps, greedy)
        draft_tokens = torch.cat((batch.anchor[:, None], tokens), dim=1)
        confidence, verify_token_budget, layout = self._plan_verify(
            batch=batch, draft_input=None, prefix_lens=prefix_lens
        )
        touched_before = [len(layer.experts.touched) for layer in self.model.layers]
        target_out = self.target_worker.forward_batch_generation(
            batch=None, forward_batch=draft_tokens, is_verify=True
        )
        logits = target_out.logits_output.next_token_logits.view(bs, self.block_size, V)
        # Make the target agree with a random-length prefix of the draft (a toy acceptance process).
        agree = torch.rand(bs, gamma, generator=g) < 0.8
        bias = torch.zeros_like(logits)
        bias[:, :gamma].scatter_(2, tokens[:, :, None], (agree.float() * self.accept_bias)[:, :, None])
        logits = (logits + bias).view(bs * self.block_size, V)
        visits = []
        for li in self.model.schedule:  # the order of the MoE visits of this forward
            visits.append(li)
        touched = []
        cursor = list(touched_before)
        for li in visits:
            touched.append(self.model.layers[li].experts.touched[cursor[li]])
            cursor[li] += 1
        record = dict(bs=bs, candidate_ids=candidate_ids, scores=scores, tokens=tokens, temps=temps,
                      unary=unary.clone(), plane=plane.clone(), logits=logits.clone(),
                      greedy=greedy, prefix_lens=prefix_lens, rids=[r.rid for r in batch.reqs],
                      req_temps=batch.sampling_info.temperatures.view(-1).float(), touched=touched,
                      draft_tokens=draft_tokens)
        if self._step in self.skip_accept_steps:
            self._step += 1
            record["accept_len"] = None
            self.history.append(record)
            return draft_tokens[:, 0]
        (
            accept_len,
            commit_lens,
            bonus,
            out_tokens,
            new_seq_lens,
            cap_trim_lens,
        ) = self._accept_block(
            candidates=draft_tokens,
            next_token_logits=logits,
            sampling_info=batch.sampling_info,
            draft_input=None,
            prefix_lens=prefix_lens,
            bs=bs,
            target_predict=None,
            cutoff_verify_lens=None if layout is None else layout.verify_lens,
            verify_shard=None,
        )
        record["accept_len"] = accept_len.clone()
        record["uncapped_accept"] = self.uncapped_accept.clone()
        record["live_lens"] = None if layout is None else layout.verify_lens.clone()
        record["bonus"] = bonus.clone()
        record["out_tokens"] = out_tokens.clone()
        record["commit_lens"] = commit_lens.clone()
        self.history.append(record)
        self._step += 1
        batch.seq_lens = new_seq_lens
        batch.anchor = bonus
        return bonus
