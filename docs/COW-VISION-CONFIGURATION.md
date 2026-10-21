# Enable image feedback on a prepared COW server

The language engine can consume image embeddings. A separate pinned image
producer turns uploaded PNG/JPEG files into those embeddings. Starting the
engine with --vision does not configure that producer.

The normal text-only configuration stays unchanged. To enable new image
uploads, the operator supplies an existing vision-config.json using the native
strata-vision executable, the matching projector, the selected language-model
first shard, and their measured SHA256 pins. Use the existing producer schema
in serve/cow_http.py: schema, executable, projector, pinned_sha256,
embedding_width and original_bytes_limit. Optional fields include threads,
max_rows, flash_attn and timeout. The selected mixed-input execution profile
must match the embedding width. All language-model shards remain pinned by
that profile; the producer additionally pins the first shard it reads.

Pass the configuration when creating a new server with the advanced
 tools/cow_server_setup.py command:

    --vision-config /absolute/path/to/vision-config.json

The saved server.json contains that path. cow-start forwards it to the existing
HTTP service. The service validates producer inputs before GPU startup.
The operator can also set the same vision_config field in an existing server's
server.json while that server is stopped. Do not edit its execution profile
or pretend that an incompatible checkpoint is compatible.

This option reuses the existing native producer. It does not download a
projector or build a vision executable automatically. Those prepared resources
are still needed. Restore-only operation without a producer remains useful for
compatible retained image state; new image uploads require the producer.

The lower-quant domain campaign found an HTTP404 at this boundary before
configuration, followed by a separate 32K context-capacity rejection. Neither
is a quantization result. The corrected common CAD trials use 98,304 context
tokens, 8,192-token native prefill batches, FP16 KV and the same producer/projector
with separate model pins. They are bounded functional trials, not numerical
G1-G5 certification. The shared BF16 vision projector is not a BF16
language-model trial.
