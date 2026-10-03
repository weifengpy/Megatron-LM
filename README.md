# FlexShard data-parallel backend for Megatron-LM

This fork adds [FlexShard](https://github.com/meta-pytorch/flex_shard) as a Megatron data-parallel backend behind `--use-flex-shard`, alongside DDP, the distributed optimizer, torch FSDP2 and Megatron-FSDP. With the flag off, Megatron behaves exactly like upstream, so one `pretrain_gpt.py` command compares Megatron's DDP / DistributedOptimizer with FlexShard by flipping a flag.

Base: upstream NVIDIA/Megatron-LM `16251ac12` plus one commit, "Add FlexShard data-parallel backend (--use-flex-shard)".

## Usage

### Requirements

- PyTorch with CUDA and NCCL. Tested with a PyTorch 2.15 dev build on CUDA 13 (flex_shard declares `torch>=2.14,<2.15`, but its test suite passes on 2.15).
- `flex_shard` importable (`pip install --no-deps -e <flex_shard>` or `PYTHONPATH=<flex_shard>/src`). It also needs `torchao`, which flex_shard declares since [#30](https://github.com/meta-pytorch/flex_shard/pull/30); `--no-deps` and `PYTHONPATH` skip it, so install it separately.
- TransformerEngine layers (`--transformer-impl transformer_engine`) need flex_shard `main` at or after [#16](https://github.com/meta-pytorch/flex_shard/pull/16) (persistent unsharded parameters, which superseded drafts #13–#15). `--flex-shard-no-sync` and `--flex-shard-no-reshard-after-backward` need `main` at or after [#18](https://github.com/meta-pytorch/flex_shard/pull/18) (no-sync gradient accumulation). fp32 local-shard gradients need `main` at or after [#23](https://github.com/meta-pytorch/flex_shard/pull/23) (per-parameter `grad_dtype`, D121468586); without it, local-shard gradients stay bf16. [#20](https://github.com/meta-pytorch/flex_shard/pull/20) (casts fused into the reduce-scatter copy-in) and [#21](https://github.com/meta-pytorch/flex_shard/pull/21) (deferred upcasts) keep their casts as cheap as before #23.
- Gradient accumulation fusion (Megatron's default; `--no-gradient-accumulation-fusion` turns it off) needs [#25](https://github.com/meta-pytorch/flex_shard/pull/25) (`BucketSpec` pre-backward and post-reduce hooks). Megatron's own linear layers, including the GPT output layer under the TransformerEngine spec, also need APEX's `fused_weight_gradient_mlp_cuda` extension for fusion, with or without FlexShard.
- Expert parallelism needs [#27](https://github.com/meta-pytorch/flex_shard/pull/27) (`BucketSpec.gradient_divide_factor`).
- Delayed expert weight gradients (`--overlap-dispatch-backward-with-experts-wgrad`) need [#33](https://github.com/meta-pytorch/flex_shard/pull/33) (`BucketSpec.defer_post_backward`).
- The EP all-to-all overlap (`--overlap-moe-expert-parallel-comm`) needs [#34](https://github.com/meta-pytorch/flex_shard/pull/34) (`FlexShardModule.unshard`).
- Blockwise FP8 parameter all-gather (`--fp8-param-gather --fp8-recipe blockwise`) needs [#35](https://github.com/meta-pytorch/flex_shard/pull/35) (a pluggable `BlockwiseFp8Quantizer` for `Fp8BucketedBlockShard`).
- Pipeline parallelism with `--flex-shard-no-sync` needs [#32](https://github.com/meta-pytorch/flex_shard/pull/32) (`finalize_backward`) for the default `--align-grad-reduce`.

### Flags

| Flag | Effect |
| --- | --- |
| `--use-flex-shard` | Shard parameters over the data-parallel group with FlexShard. Default `reshard_after_forward=True` (ZeRO-3). |
| `--flex-shard-no-reshard-after-forward` | Keep gathered parameters from forward until backward (ZeRO-2). |
| `--flex-shard-no-sync` | With gradient accumulation, reduce-scatter only in the last microbatch's backward. Earlier microbatches accumulate full gradients (fp32 with `--accumulate-allreduce-grads-in-fp32`, the bf16 default), at the memory cost of one full gradient copy. |
| `--flex-shard-no-reshard-after-backward` | With `--flex-shard-no-sync`, keep gathered parameters between microbatches, so only the first microbatch all-gathers (without reshard-after-forward). |

`validate_args` rejects combining `--use-flex-shard` with any of:
- `--overlap-moe-expert-parallel-comm` without `--flex-shard-no-reshard-after-forward --flex-shard-no-sync --flex-shard-no-reshard-after-backward`
- `--delay-wgrad-compute` without gradient accumulation fusion, and `--overlap-dispatch-backward-with-experts-wgrad` with `--flex-shard-no-sync` but without it
- TransformerEngine single grouped MoE weights or biases (`--moe-single-grouped-weight`, `--moe-single-grouped-bias`)
- `--use-distributed-optimizer`, `--overlap-param-gather`
- fp16
- optimizers other than Adam / SGD
- `--use-torch-fsdp2` or `--use-megatron-fsdp`

### Example

```bash
torchrun --nproc-per-node 8 pretrain_gpt.py \
  --num-layers 24 --hidden-size 2048 --ffn-hidden-size 5632 --num-attention-heads 16 \
  --seq-length 2048 --max-position-embeddings 2048 \
  --micro-batch-size 1 --global-batch-size 8 --train-iters 60 \
  --lr 3e-4 --min-lr 3e-5 --lr-decay-style cosine --lr-warmup-iters 5 --clip-grad 1.0 \
  --bf16 --swiglu --normalization RMSNorm --position-embedding-type rope \
  --untie-embeddings-and-output-weights --disable-bias-linear \
  --transformer-impl transformer_engine --no-gradient-accumulation-fusion \
  --mock-data --tokenizer-type NullTokenizer --vocab-size 32000 \
  --log-throughput --timing-log-level 1 --eval-iters 0 \
  --use-flex-shard --flex-shard-no-reshard-after-forward
```

For the Megatron baseline, replace the last line with `--use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather`.

Without TransformerEngine, use `--transformer-impl local` together with:
- `--no-rope-fusion`
- `--no-bias-swiglu-fusion`
- `--no-masked-softmax-fusion`
- `--no-bias-dropout-fusion`
- `--no-persist-layer-norm`

## Design

`FlexShardDataParallel` (`megatron/core/distributed/flex_shard/flex_shard_data_parallel.py`) subclasses `_BaseDataParallel`, like the torch FSDP2 wrapper.

- **Buckets**, in forward order: the embedding, one bucket per `TransformerLayer`, any remaining parameter-owning modules (e.g. `final_layernorm`), then `output_layer`. Each bucket is one all-gather before use and one reduce-scatter after backward. Every parameter is `Shard(0)` (`per_param_placements`) over the `dp_cp` group, except expert parameters (see expert parallelism below). Buckets are split by dtype, because FlexShard requires one dtype per bucket.
- **Reshard-after-forward** follows `FlexShardDataParallelConfig.reshard_after_forward`. The last bucket never reshards, because its backward runs immediately (like the FSDP2 root).
- **Parameter attributes.** FlexShard replaces each parameter with a local-shard tensor. The wrapper saves Megatron's per-parameter attributes (`tensor_model_parallel`, `allreduce`, ...) before `flex_shard()` and restores them afterwards.
- **Optimizer.** After wrapping, `module.parameters()` yields local shards, so Megatron's existing `Float16OptimizerWithFloat16Params` (Adam, fp32 main params) updates only this rank's shard. With `--accumulate-allreduce-grads-in-fp32` (the bf16 default), each bf16 parameter gets `grad_dtype=torch.float32` before `flex_shard()`, so FlexShard stores fp32 local-shard gradients, as the distributed optimizer keeps them, and the optimizer uses them as its main gradients without a copy. Each gradient element lives on exactly one data-parallel rank, so grad stats (norm, zero count) are reduced over WORLD (`megatron/core/optimizer/__init__.py`).
- **Grad sync.** FlexShard reduce-scatters during backward and waits at the end of backward, so `finish_grad_sync` is a no-op. `scale_gradients` scales the local shards. Buckets average over their group, as Megatron DDP scales gradients by 1/DP; with `--calculate-per-token-loss` they sum instead, since `finalize_model_grads` divides every gradient by the global token count.
- **No-sync.** With `--flex-shard-no-sync`, `train()` puts `FlexShardDataParallel.no_sync` in `no_sync_func`. It turns FlexShard's `set_requires_gradient_sync` off on entry and back on at exit, as Megatron DDP's `no_sync()` does with `is_last_microbatch`; FlexShard, like FSDP2, has no context manager of its own. Backwards of all but the last microbatch keep full gradients on FlexShard's persistent unsharded parameters, and autograd accumulates into them. The last microbatch's backward reduce-scatters them, including buckets it did not use. FlexShard always reshards after that syncing backward, so `--flex-shard-no-reshard-after-backward` cannot leave stale parameters after the optimizer step.
- **Selection.** `get_megatron_ddp_config` returns a `FlexShardDataParallelConfig` when `--use-flex-shard` is set. Both `get_model()` and the ModelBuilder path that `pretrain_gpt.py` uses (`megatron/training/models/dist_utils.py:_ddp_wrap`) pick the wrapper from that config type.
- **Process groups** come from `pg_collection.dp_cp`, with a fallback to `parallel_state` for callers that don't pass groups.
- **Tensor parallelism.** With TP, each rank's parameters are its TP slices, and FlexShard shards them over the rank's `dp_cp` group, which excludes its TP peers. Megatron's layers keep their own TP communication: column- and row-parallel linears, sequence-parallel all-gathers and reduce-scatters, and TransformerEngine's `--tp-comm-overlap`. The restored `tensor_model_parallel` attributes keep the grad-norm filter (`param_is_not_tensor_parallel_duplicate`) counting TP-replicated parameters, such as layer norms, once. `finalize_model_grads` all-reduces sequence-parallel and `--qk-layernorm` layer-norm gradients over TP on `param.grad`, FlexShard's local-shard gradient, since there is no `main_grad`. On Hopper, Megatron requires `CUDA_DEVICE_MAX_CONNECTIONS=1` with TP, and FlexShard runs with it.
- **Gradient accumulation fusion.** With it, TransformerEngine's and Megatron's linear layers add weight gradients straight into `param.main_grad` and give autograd none. For buckets with such layers, the wrapper passes flex_shard two per-bucket hooks. The pre-backward hook allocates each gathered parameter's gradient, zeroed and fp32 with `--accumulate-allreduce-grads-in-fp32`, and aliases it as `main_grad`; the post-reduce hook drops the alias once the reduce-scatter has taken the gradient. flex_shard itself knows nothing about `main_grad`. The fused GEMMs therefore add into the gradient FlexShard reduce-scatters, across microbatches with no-sync, with no separate buffer or copy. Megatron's own linear layer captures `main_grad` at forward, so its backward now keeps a `main_grad` attached after forward instead of resetting it to forward's `None`. The embedding and final-norm buckets keep ordinary autograd gradients.
- **Tied embeddings.** The output layer reuses the embedding's weight at call time and registers none, so FlexShard cannot see that use from parameter names. When the model ties them, the wrapper puts the final norm's parameters into the embedding bucket, as torchtitan groups `[tok_embeddings, norm, lm_head]` for FSDP2. Their deepest common module is the model root, which also runs the output layer, so FlexShard hooks the bucket there: it gathers before the embedding, stays gathered through the output layer, re-gathers at the start of backward, and reduce-scatters after both uses' gradients have accumulated. The wrapper asserts that anchor, since FlexShard, like FSDP2, cannot detect call-time tying. The bucket skips reshard-after-forward, which at the root would free the weight right before backward re-gathers it. With fusion, the bucket gets the `main_grad` hooks, whose pre-backward hook now runs before the output layer's fused GEMM.
- **Expert parallelism.** Expert parameters (`allreduce=False`, Megatron's marker for the expert topology) exist only on their EP rank and are replicated over the expert data-parallel group (`pg_collection.expt_dp`), so the wrapper gives each MoE layer's experts their own bucket on that group, after the layer's other parameters, as Megatron DDP keeps them in separate buffers. FlexShard hooks it on the experts module, so it gathers after the token dispatch and its all-gather overlaps attention. Each expert's gradient already sums the tokens its EP peers routed to it, so the sum over the expert data-parallel group is divided by the dense data-parallel size (`gradient_divide_factor`), as Megatron DDP, Megatron-FSDP and torchtitan's FSDP2 scale expert gradients. Keying on `allreduce` matches DDP: at EP 1 with ETP = TP, experts are ordinary dense parameters. Megatron builds separate dense and expert optimizers from the restored `allreduce` attribute; both reduce grad stats over WORLD, and `param_is_not_tensor_parallel_duplicate` filters expert duplicates over ETP. The token dispatchers and router buffers (`expert_bias`, token counts) are untouched. With `--overlap-dispatch-backward-with-experts-wgrad`, TransformerEngine leaves the experts' weight gradients to `backward_dw()`, which the token dispatch's backward runs on a side stream after the experts' backward, overlapping the dispatch all-to-all. TransformerEngine marks those weights `skip_backward_post_hook`, so the wrapper gives their buckets flex_shard's `defer_post_backward` and sets each weight's `post_wgrad_grad_acc_hook`, which Megatron calls after `backward_dw()`, to finish the bucket with `finish_deferred_backward`. Megatron DDP waits for those gradients the same way, through TransformerEngine's weight-gradient hooks. Without fusion, `backward_dw()` assigns `param.grad` instead of adding to it, which would overwrite gradients accumulated without sync, so `--flex-shard-no-sync` needs fusion here. With `--overlap-moe-expert-parallel-comm`, Megatron's schedule runs one microbatch's forward alongside another's backward, splits each layer into schedule steps (attention and router, dispatch, experts, combine) whose backwards are separate autograd calls, and calls the layers' sub-modules directly, so FlexShard's forward hooks on `TransformerLayer` and at the model root never run, as Megatron notes for Megatron-FSDP. The schedule therefore gathers every bucket before the step with flex_shard's `unshard()` (setting the fused layers' `main_grad` aliases the pre-backward hook would set), FlexShard runs with manual backward finalization so a backward call finishes nothing, each TransformerLayer's buckets (its dense parameters and its experts) defer their post-backward and finish from the schedule's per-layer post-backward hook (`set_fsdp_reshard_hooks`, which Megatron-FSDP also uses), after the layer's last backward step or its `backward_dw()`. In the step's last backward, that reduce-scatters each layer while the layers before it run their backward, as Megatron DDP overlaps its bucket reduce-scatters; earlier backwards keep accumulating without sync. `start_grad_sync` reduce-scatters the rest after the last backward (per model chunk with virtual PP, where the schedules already call it, and where sync stays off during backward), with `finish_grad_sync` waiting for it. Buckets on modules the schedule still calls (the experts, the embedding) keep their own triggers. This needs FlexShard's apples-to-apples setup (no reshard-after-forward, no-sync, no reshard-after-backward), which keeps full parameters and gradients for the step, as Megatron DDP does. With `--delay-wgrad-compute`, which requires the overlap, TransformerEngine also delays the attention and expert weight gradients to the schedule's `backward_dw()` steps: the expert buckets defer to their weights' `post_wgrad_grad_acc_hook`, which the schedule runs after each `backward_dw()`, and the other buckets reduce-scatter after the step, after every `backward_dw()`. It needs fusion, since without it `backward_dw()` assigns `param.grad`, which would overwrite the gradients accumulated without sync.
- **Pipeline parallelism.** Each model chunk, one per virtual pipeline stage, is its own `FlexShardDataParallel`, with its own buckets, communication streams and learned bucket order. Megatron's schedules run each microbatch's backward as a separate backward call, which FlexShard finishes with its end-of-backward callback. With `--flex-shard-no-sync`, `train()` hands the schedules one `no_sync` per chunk, and they re-enable sync before each chunk's last microbatch backward, which reduce-scatters. With `--align-grad-reduce` (the default), the schedules re-enable sync before the last backward only on the first stage, and the other stages call `start_grad_sync` (as `grad_sync_func`) after a chunk's last microbatch. It reduce-scatters the accumulated gradients outside backward with flex_shard's `finalize_backward(async_op=True)`, so the reduce-scatter overlaps the pipeline bubble, as Megatron DDP's does, and `finish_grad_sync` waits for it. Tied embeddings across stages need nothing new: the last stage's output weight is a separate copy (`shared = True`) that Megatron sets equal to the first stage's embedding at initialization, and `finalize_model_grads` all-reduces the two copies' gradients over the embedding group on the local shards, which both stages shard identically; the grad-norm filter skips the copy. On every stage with the output layer, the model fetches the tied weight before calling the output layer: the embedding's on a stage that holds the embedding (the MTP stage), the output layer's own copy otherwise. The wrapper groups that weight's bucket with the final norm at the model root, as it does without pipeline parallelism, so the weight is gathered before the model fetches it. For an apples-to-apples comparison with Megatron DDP + distributed optimizer, use `--flex-shard-no-reshard-after-forward --flex-shard-no-sync --flex-shard-no-reshard-after-backward`: like DDP, it keeps a stage's full parameters and gradients across the schedule and reduces once per step, and it is torchtitan's default with PP. With reshard-after-forward, FlexShard all-gathers once per microbatch.
- **Context parallelism.** CP splits each sequence over the CP group and leaves the parameters replicated across it, so it needs nothing new in the wrapper. Dense buckets shard over `dp_cp`, which includes the CP ranks, as the distributed optimizer does, and Megatron folds CP into the expert data-parallel group that expert buckets shard over. Each CP rank normalizes its loss by its own token count, and the buckets average over `dp_cp`, as Megatron DDP scales gradients by 1/`dp_cp`; expert buckets divide by the `dp_cp` size, so CP counts there too. With `--calculate-per-token-loss`, the buckets sum and `finalize_model_grads` divides by the token count all-reduced over `dp_cp`. That is torchtitan's normalization: its FSDP2 shards over `(dp_shard, cp)`, sums gradients, and divides the loss by the global token count. TransformerEngine's attention does all of CP's communication, on the CP group inside its own forward and backward, so CP needs `--transformer-impl transformer_engine`. On Hopper, Megatron requires `CUDA_DEVICE_MAX_CONNECTIONS=1` with CP, as with TP, and FlexShard runs with it.
- **Blockwise FP8 parameter all-gather.** With `--fp8-param-gather` and `--fp8-recipe blockwise` (TransformerEngine's `Float8BlockScaling`), FlexShard all-gathers the weights of TransformerEngine's linear layers in FP8, halving their all-gather bytes and gathered memory, while their local shards, gradients and optimizer stay bf16/fp32: `validate_args` keeps `fp8_param` off, so the model has bf16 parameters, unlike Megatron's FP8 primary weights. Weights whose dims are multiples of 128 (`weight`, or a grouped linear's `weight<i>`) use flex_shard's `Fp8BucketedBlockShard`, which cuts their bucket into 128-row block rows so a 128 x 128 block never straddles ranks; the rest of the bucket (norms, biases) shares its one all-gather through `MixedBucketPlacement`. Each rank quantizes its block rows with TransformerEngine's weight quantizer for the recipe (`te_fp8.py`), so the gathered FP8 data and scales equal TransformerEngine quantizing the full bf16 weight, as its FSDP2 hook gathers them along dim 0. The weight factory builds a `Float8BlockwiseQTensor` over the gathered buffer, which TransformerEngine's layers use as is instead of quantizing the weight. TransformerEngine derives the column-wise copy backward needs from the row-wise data, and the post-reduce hook frees it, since the next all-gather refills the row-wise data with the updated weights. Numerics equal Megatron's blockwise FP8 training without `--fp8-param-gather`; Megatron's flag instead quantizes fp32 main parameters straight to FP8.

### Limitations

- **Checkpoint save/load** is not wired up.
- **`torch.compile`:** FlexShard falls back to synchronous unshard under compile, so it is not used here.

## Benchmark: Megatron DDP + distributed optimizer vs FlexShard without reshard-after-forward

The goal is an apples-to-apples comparison between Megatron's data-parallel baseline and FlexShard configured to behave like it. AdamW comes first (Phase A), then Muon (Phase B). The [Plan](#plan) lists the steps; the results so far come first.

### Method

The baseline is **Megatron DDP with the distributed optimizer** (`--use-distributed-optimizer`, ZeRO-1):
- It keeps full bf16 params during forward and backward.
- It shards the optimizer state.
- Per step, it reduce-scatters the gradients once and all-gathers the params once.

The FlexShard side is **FlexShard without reshard-after-forward** (`--flex-shard-no-reshard-after-forward`, ZeRO-2):
- It also keeps full bf16 params during forward and backward.
- With gradient accumulation, `--flex-shard-no-sync --flex-shard-no-reshard-after-backward` makes it move the same bytes per step as the baseline.

A gap between the two therefore measures the implementation (scheduling and bucketing), not the sharding strategy.

| | Megatron DDP + distributed optimizer | FlexShard without reshard-after-forward |
| --- | --- | --- |
| Flags | `--use-distributed-optimizer --overlap-grad-reduce --overlap-param-gather` | `--use-flex-shard --flex-shard-no-reshard-after-forward`, plus `--flex-shard-no-sync --flex-shard-no-reshard-after-backward` with gradient accumulation |
| Communication per step | Reduce-scatter of grads + all-gather of params (after the optimizer, overlapped with the next forward) | All-gather per bucket in the first forward + reduce-scatter per bucket in the last backward |
| Optimizer | `DistributedOptimizer` (Adam, fp32 main params, 1/DP) | `Float16OptimizerWithFloat16Params` (Adam, fp32 main params) on local 1/DP shards |
| Grads | Persistent full fp32 grad buffer, reduce-scattered to fp32 shards | fp32 local-shard grads, plus full fp32 grads between microbatches only |
| Buckets | ~40M-param contiguous buckets | One per `TransformerLayer` + embedding / final norm / output |

Both sides match in:
- the model, data and seed;
- bf16 params, fp32 grads and fp32 main params;
- Adam with decoupled weight decay and global-norm clipping;
- gradient accumulation through `no_sync_func`.

FlexShard's fp32 shard grads need flex_shard #23 (see [Requirements](#requirements)).

Reference setups (not part of the goal):
- **Megatron DDP** without the distributed optimizer, to show what the distributed optimizer adds.
- **FlexShard with reshard-after-forward** (ZeRO-3, the `--use-flex-shard` default) and **Megatron-FSDP** (`optim_grads_params`). These trade communication for memory that the baseline doesn't save, so they are compared only with each other (see the [Roadmap](#roadmap)).

Setup:
- 8x H100 96 GB, DP = 8, TP = PP = 1.
- Mock data, bf16 params, MBS 1, seq 2048, lr 3e-4.
- Timing is the median over iterations ≥ 20.

Models:
- **1.4B model:** 24 layers, hidden 2048, ffn 5632, 16 heads (1.36B parameters).
- **7.2B model:** 32 layers, hidden 4096, ffn 14336, GQA 32/8, i.e. Llama-3-8B layers with a 32K vocabulary (7.24B parameters). It runs with `--recompute-granularity selective` in every setup, because Megatron with the distributed optimizer runs out of memory without it.

### Results (local spec, no TransformerEngine)

These were measured with FlexShard before flex_shard #16 and with bf16 FlexShard shard grads. Phase A re-measures them.

| Model | Megatron DDP + distributed optimizer, ms/it | FlexShard without reshard-after-forward, ms/it | Change | Max allocated, Megatron / FlexShard |
| --- | --- | --- | --- | --- |
| 1.4B | 157.7 | 156.6 | −1% | 21.5 / 16.5 GB |
| 7.2B | 594.9 | 538.1 | −10% | 59.4 / 33.6 GB |

- **Profile (1.4B model, rank 0, one step):**
  - Compute is the same in both setups (~117 ms).
  - FlexShard exposes 45.5 ms of NCCL time (all-gather 25.3, reduce-scatter 17.6), against 13.0 ms for Megatron.
  - FlexShard's stalls are ~1 ms gaps before `split_with_sizes_copy_out`: each layer waits for its own all-gather, and one-bucket-ahead prefetch does not hide it.
- **4 microbatches, before FlexShard had no-sync (not like-for-like):**
  - Megatron vs FlexShard: 495.8 vs 545.3 ms (+10%) on the 1.4B model, 1824.4 vs 2006.2 ms (+10%) on the 7.2B model.
  - FlexShard's extra cost per microbatch on the 1.4B model (+17 ms) matches its exposed reduce-scatter.
  - See [Gradient accumulation with no-sync](#gradient-accumulation-with-no-sync-transformerengine-14b-model-dp--4) for the fix.
- **ZeRO-3 pair, 1 microbatch:** FlexShard with reshard-after-forward vs Megatron-FSDP is 179.9 vs 164.1 ms on the 1.4B model and 649.6 vs 533.0 ms on the 7.2B model, with similar memory.
- **Correctness:**
  - Iteration-1 loss and grad norm are bit-identical across all five setups with the local spec: Megatron DDP, Megatron with the distributed optimizer, Megatron-FSDP, and FlexShard with and without reshard-after-forward.
  - With the TransformerEngine spec, they are bit-identical across Megatron with the distributed optimizer and both FlexShard setups.
  - A 4-layer model tracks over 20 iterations (loss 2.357658 vs 2.357765).
  - The 1.4B and 7.2B models diverge after ~iteration 5 in every setup, Megatron DDP vs Megatron with the distributed optimizer included, because lr 3e-4 is unstable for them.

### Gradient accumulation with no-sync (TransformerEngine, 1.4B model, DP = 4)

Setup:
- TransformerEngine spec, 1.4B model, 4x H100 with no other jobs, MBS 1.
- GBS 8 means 2 microbatches per step, and GBS 32 means 8.
- Two GPU sets ran the two GBS series at the same time. Each cell is the mean of two interleaved repetitions of the median ms/it over iterations ≥ 20.
- "No-sync" means `--flex-shard-no-sync`. "Params kept" means `--flex-shard-no-reshard-after-backward` on top of it.

| Setup | GBS 8 ms/it | GBS 32 ms/it | Max allocated |
| --- | --- | --- | --- |
| Megatron DDP + distributed optimizer | 192.6 | 640.7 (612.9 / 668.5) | 12.9 GB |
| FlexShard without reshard-after-forward | 216.2 | 795.9 | 9.0 GB |
| … with no-sync | 206.7 | 710.4 | 13.5 GB |
| … with no-sync and params kept | **193.2** | **606.3** | 13.5 GB |
| FlexShard with reshard-after-forward | 239.2 | 875.4 | 6.6 GB |
| … with no-sync | 228.4 | 792.1 | 11.0 GB |
| … with no-sync and params kept | 215.4 | 703.8 | 11.0 GB |

- **FlexShard without reshard-after-forward, with no-sync and params kept, matches Megatron.**
  - At 2 microbatches it is +0.3% against Megatron.
  - At 8 microbatches it is −5% against Megatron's mean, and −1% against its faster repetition.
  - Like Megatron, it does one reduce-scatter and one all-gather per bucket per step.
- **No-sync alone** speeds FlexShard without reshard-after-forward up by 4% (GBS 8) and 11% (GBS 32). Keeping params speeds it up by another 7% and 15%, because each later microbatch skips the forward all-gather.
- **FlexShard with reshard-after-forward** (ZeRO-3) still re-gathers every bucket in each backward. With no-sync and params kept, it is 12% behind Megatron at 2 microbatches and 10% at 8.
- **Memory:**
  - No-sync adds about 4.5 GB at the peak, for the full fp32 gradients it keeps between microbatches.
  - Keeping params adds nothing at the peak.
  - FlexShard without reshard-after-forward, with no-sync and params kept, uses 0.7 GB more than Megatron.
- **Correctness:** iteration-5 loss and grad norm agree across all setups within run-to-run noise. For example, at GBS 8:
  - Megatron: 10.09253 / 61.682.
  - FlexShard with no-sync: 10.09292 / 61.558.
  - FlexShard with no-sync and params kept: 10.09299 / 61.576.

### Plan

#### Phase A: AdamW

Megatron DDP + distributed optimizer vs FlexShard without reshard-after-forward.

1. **Matched configuration.** Use the [Method](#method) table. This needs flex_shard #23 for fp32 shard grads, and #20 and #21 to keep their casts cheap. The Megatron side is already in: `--accumulate-allreduce-grads-in-fp32` (the bf16 default) gives bf16 params fp32 shard grads.
2. **Correctness gate.**
   - Run the 1.4B model at DP = 4, then the 7.2B model at DP = 8, with at least 2 microbatches for about 500 steps.
   - Use a stable lr, e.g. 1e-4 with a 50-step warmup; 3e-4 diverges after about 5 iterations in every setup.
   - Run the Megatron baseline twice for the noise floor, since TransformerEngine kernels are not bit-deterministic.
   - FlexShard passes if its loss and grad norm stay within the spread between the two Megatron runs.
3. **Performance.**
   - Cover both models at 1, 2 and 8 microbatches with the TransformerEngine spec, and also seq 4096.
   - For each cell, take the median ms/it over iterations ≥ 20. Report the min and median of at least 3 interleaved repetitions, plus peak allocated memory.
   - Run on a quiet node, or report GPU kernel time when the node is busy.
   - Profile one step per setup for exposed NCCL time and gaps, including Megatron on the 7.2B model, where it was slower than both FlexShard setups.
4. **Close gaps and record.**
   - Make one targeted fix per FlexShard shortfall, e.g. prefetch depth, or bucket size against Megatron's `--ddp-bucket-size` 20/40/80M with nccl-tests at both sizes.
   - Record the final numbers here.

#### Phase B: Muon

Megatron layer-wise Muon vs FlexShard + DistMuon. The FlexShard side keeps `reshard_after_forward=False`.

| | Megatron layer-wise Muon | FlexShard + DistMuon |
| --- | --- | --- |
| Wiring | `--optimizer muon --use-distributed-optimizer --muon-scalar-optimizer adam` → `LayerWiseDistributedOptimizer` | owned buckets from flex_shard's `materialize_dist_muon_buckets` + `build_local_dist_muon(DistMuon)` (torchtitan `torchtitan/distributed/flex_shard/dist_muon.py`) |
| Ownership | whole matrices, LPT bin-packing per bucket | whole matrices / block groups, `assign_matrices` |
| Communication per step | Reduce-scatter of grads to the owning ranks + all-gather of params | All-gather of params in forward + reduce-scatter of grads to the owning ranks |
| Optimizer communication | none | none (storage layout == compute layout) |
| Uneven shards | padded to the largest owner | padded to the largest owner |
| Non-matrix params | Adam | Adam on `Shard(0)` buckets |

- Megatron's layer-wise Muon reduce-scatters to the owning ranks by default (`use_layer_wise_param_layout=True`); the class docstring's all-reduce flow is the legacy path. It needs `emerging_optimizers` `v0.3.0`.
- The two AdamW setups from Phase A serve as references. Comparing each stack's Muon-minus-AdamW difference isolates the cost of switching from Adam to Muon in that stack.

1. **Wire FlexShard + DistMuon in Megatron.**
   - Pick Muon params with Megatron's `is_managed_by_layer_wise_optimizer` (qkv, proj, fc1 and fc2 weights), so both stacks use the same set.
   - Add a `--flex-shard-dist-muon` mode: `assign_matrices` turns the Muon params into owned buckets per layer, and non-matrix params go on `Shard(0)` buckets.
   - Optimizer: a `ChainedOptimizer` of DistMuon on fp32 main copies of the owned shards, plus Megatron Adam on the rest. Take the global grad norm over WORLD, and handle ranks that own no matrices.
   - Relax the `--use-flex-shard` optimizer restriction.
   - Since flex_shard #23, DistMuon's local adapter requires real-param grads in the param dtype. `--accumulate-allreduce-grads-in-fp32` gives bf16 params fp32 shard grads, so either let DistMuon accept fp32 grads or keep bf16 grads for Muon params.
2. **Parity gate before timing.**

   | Knob | Megatron | DistMuon |
   | --- | --- | --- |
   | Newton-Schulz coefficients / steps | `--muon-coefficient-type`, `--muon-num-ns-steps` | `ns_coefficients` (3.4445, −4.7750, 2.0315), `ns_steps=5` |
   | Newton-Schulz precision | `--muon-fp32-matmul-prec` | bf16 |
   | Update scale | `--muon-scale-mode spectral` | `adjust_lr_fn` (`original` √max(1, r/c), `match_rms_adamw`, `spectral_unclamped`) |
   | Momentum / weight decay | `--muon-momentum`, `--muon-nesterov`, confirm decoupled weight decay | 0.95, Nesterov, decoupled |
   | QKV / fc1 split | per-head split by default (`--muon-no-split-qkv`) | whole matrix or `BlockShard`; start with no split on both sides |

   Checks:
   - A single-matrix update agrees within ~2e-2.
   - The Megatron and FlexShard Muon loss curves agree over 50 iterations at a stable lr.
   - In Megatron, the Muon and AdamW curves differ, confirming Muon is active.
3. **Performance.** Same protocol as Phase A, step 3, also reporting each stack's Muon-minus-AdamW time.
4. **Risks.**
   - Coarse ownership at DP = 8 (mitigate with block groups).
   - DistMuon requires a grad for every configured param.
   - Megatron's interleaved per-group QKV layout.
   - Keep `overlap_param_gather_with_optimizer_step` off.

## Roadmap

After Phases A and B, in order of benchmarking value:
1. **Tensor parallelism (Megatron vs FlexShard at TP × DP).** This is the most common Megatron configuration for dense models from about 8B up, so larger comparisons need it. `validate_args` no longer rejects it: FlexShard shards each TP rank's slices over that rank's data-parallel group (see [Design](#design)).
   - 117M model (4 layers, hidden 1024) at TP 2 × DP 2: iteration-1 loss and grad norm match Megatron exactly, with and without sequence parallelism, no-sync, `--qk-layernorm` and `--tp-comm-overlap`.
   - 1.4B model over 500 iterations at TP 2 × DP 2 and TP 2 × DP 4, with sequence parallelism: FlexShard's loss curves differ from Megatron's about as much as Megatron's two runs differ from each other.
   - 1.4B model over 500 iterations at TP 4 × DP 2, with sequence parallelism: Megatron's two runs are bit-identical here, so there is no noise floor. FlexShard matches at iteration 1, and its 50-iteration moving average of the loss stays within 0.02 of Megatron's (with and without no-sync), ending at 0.0115 vs 0.0117.
   - 7.2B model at TP 2 × DP 4 with sequence parallelism, measured with flex_shard at #21 and fusion off: FlexShard is 14% faster than Megatron at 1 microbatch (222 vs 259 ms/it, 16.8 vs 30.4 GB max allocated), 4.5% at 2 microbatches and 3% at 8 (with no-sync; 30.1 vs 30.4 GB). It doesn't need `CUDA_DEVICE_MAX_CONNECTIONS=1`: with it unset, step time stays within 2% (368 vs 366 ms/it at 2 microbatches, 1143 vs 1124 at 8).
2. **Gradient accumulation fusion.** With TransformerEngine, Megatron by default has the weight-gradient GEMM accumulate straight into an fp32 `main_grad` buffer. `--use-flex-shard` now supports it (see [Design](#design)), and on the 117M model iteration-1 loss and grad norm match Megatron exactly with fusion on both sides.
   - Still to do: benchmark Megatron and FlexShard with fusion on the 1.4B and 7.2B models. The benchmarks so far turned fusion off on both sides, partly because this environment lacks APEX's `fused_weight_gradient_mlp_cuda`, which Megatron's own linear layers need for fusion.
3. **Tied embeddings.** `--use-flex-shard` now supports Megatron's default tied embedding and output weights (see [Design](#design)). On the 117M model, iteration-1 loss and grad norm match Megatron exactly at DP 4 and at TP 2 × DP 2 with sequence parallelism, with no-sync, fusion and multi-token prediction. The 1.4B model tracks Megatron over 500 iterations within its run-to-run spread.
4. **Expert parallelism (MoE, Megatron vs FlexShard).** `--use-flex-shard` now supports EP (see [Design](#design)), with flex_shard [#27](https://github.com/meta-pytorch/flex_shard/pull/27).
   - Small MoE model (4 layers, hidden 1024, 8 experts, top-2, expert FFN 2048, grouped GEMM, all-to-all dispatcher, tied embeddings) on 8 GPUs: iteration-1 loss and grad norm match Megatron DDP + distributed optimizer exactly at EP 1, 2, 4 and 8, and at TP 2 with sequence parallelism, expert TP 2 and 1, and EP 2 and 4. At EP 4 they also match with reshard-after-forward, no-sync, per-token loss, expert bias, shared experts with and without overlap, a dense first layer, multi-token prediction, recompute of `moe_act` and of `moe`, the allgather dispatcher, gradient accumulation fusion, experts without grouped GEMM, and with 64 experts, top-1 routing and 32-token sequences, where many experts receive no tokens.
   - 10B MoE model (Qwen3-30B-A3B layers, 16 of its 48) over 500 iterations at EP 4 × expert data-parallel 2, with no-sync over 4 microbatches: FlexShard's loss and grad-norm curves differ from Megatron's about as much as Megatron's two runs differ from each other, with no bias. On the mock data every run reaches a loss of about 0.005, so only the first 150 iterations tell the runs apart.
   - Still to do: benchmark Megatron and FlexShard on the 10B MoE model.
5. **Pipeline parallelism.** `--use-flex-shard` now supports PP and virtual PP (see [Design](#design)).
   - 117M model with tied embeddings and 8 microbatches: iteration-1 loss and grad norm match Megatron DDP + distributed optimizer exactly at PP 2 × DP 4, PP 4 × DP 2, PP 2 × TP 2 × DP 2 with sequence parallelism, and PP 2 with 2 virtual pipeline chunks, for FlexShard with no-sync, with reshard-after-forward, and syncing every microbatch. At PP 2 × DP 4 they also match with MTP, per-token loss, full recompute, an uneven layer split, fusion and untied embeddings, and so does the small MoE model at PP 2 × EP 2, with and without TP 2.
   - 1.4B model over 500 iterations at PP 2 × DP 4 and PP 4 × DP 2: FlexShard's loss curves differ from Megatron's about as much as Megatron's two runs differ from each other.
   - Later stages reduce-scatter in the pipeline bubble, as Megatron DDP does, through flex_shard's `finalize_backward` ([#32](https://github.com/meta-pytorch/flex_shard/pull/32)). Iteration-1 loss and grad norm still match Megatron exactly at PP 2 × DP 2, with and without 2 virtual pipeline chunks and with MTP, and for the small MoE model at PP 2 × EP 2.
   - Still to do: a benchmark, with and without the overlap in the bubble.
6. **Context parallelism.** `--use-flex-shard` composes with CP without changes (see [Design](#design)).
   - 117M model with tied embeddings at an 8K sequence and 8 microbatches: iteration-1 loss and grad norm match Megatron DDP + distributed optimizer exactly at CP 2 × DP 4, with and without reshard-after-forward and with no-sync.
   - Still to do: CP 2 × TP 2 × DP 2 and CP 2 × PP 2 × DP 2; CP 4 and 8; the all-gather, all-to-all and hierarchical CP communication types; MTP, recompute, fusion, per-token loss and untied embeddings; the small MoE model with EP; loss curves on the 1.4B model; and a long-context benchmark.
7. **FP8 parameter all-gather.** `--use-flex-shard` all-gathers TransformerEngine's weights in FP8 with `--fp8-param-gather --fp8-recipe blockwise` (see [Design](#design)); with other recipes it still switches the flag off with a warning.
   - 117M model with the blockwise recipe at DP 8: iteration-1 loss and grad norm match Megatron DDP + distributed optimizer's blockwise FP8 training without `--fp8-param-gather` exactly, and peak allocated memory drops by 94 MB (1224 vs 1317 MB) against FlexShard's bf16 all-gather.
   - Still to do: the composition matrix (TP, EP, PP, the EP overlap, fusion, recompute), loss curves against Megatron with `--fp8-param-gather`, and a benchmark.
8. **Real training runs.**
   - **Distributed checkpoint save/load** for FlexShard shards and their optimizer state. Benchmarks don't save or load. flex_shard `426e2bf` adds DCP metadata for model tensors.
   - **Evaluation** (`--eval-iters > 0`) is untested, since every benchmark ran with evaluation off.
9. **ZeRO-3: FlexShard with reshard-after-forward vs Megatron-FSDP,** only for models that don't fit with full params resident. The numbers above for those two setups predate flex_shard #16.
