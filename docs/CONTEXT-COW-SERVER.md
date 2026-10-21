# Run the Context COW server

For a first installation, follow **[Desktop server + laptop client: start here](TWO-COMPUTER-SETUP.md)**.
It explains the hardware, automatic Q4 download, model preparation, server setup,
LAN address, API key and BranchWrite laptop installation. It uses ordinary Linux
Terminal commands and does not require an AI assistant.

The server setup is for the documented Q4/NVIDIA reference configuration. The
client needs no GPU. This source-built server requires a working CUDA compiler;
it does not automatically install a graphics driver or qualify every gaming PC.

## Already have the model files?

The resource command can verify an existing full Q4 download without copying it
or downloading MTP:

```sh
.venv/bin/python tools/cow_prepare_model.py --existing-models /your/model/folder --yes
```

Use the first model path printed by that command when running
`tools/cow_server_setup.py`. The pack path is printed too. Model files are checked
against the pinned hashes; existing inputs are not changed. A matching prepared
pack is reused only after its receipt and actual bytes pass verification.

## Settings and supported scope

`tools/cow_server_setup.py --help` lists context capacity, batch size, resident
expert budget and RAM/disk pool sizes. Pool budgets are additional to model and OS
memory. The default preview expires idle checkpoint copies after 30 minutes.
Retained canonical history remains available for explicit reconstruction.

Profiles bind model, executable, source and settings. Do not import saved native
state across incompatible profiles. Exact continuation is scoped to matching
execution arrangements; arbitrary batch-boundary equivalence remains unqualified.

This launcher starts text and server-side GBNF/JSON Schema support. An image encoder,
CAD, Blender and remote administration targets require their separately configured
capabilities. Native image-row support alone is not an image encoder installation.

See [executed LAN install/restart evidence](../validation/linux-lan-install-20261020/)
and the resource-preparation evidence for what was actually run. Current bounded
functional results do not establish cumulative G1–G5 certification.
