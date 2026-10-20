# BranchWrite connection discovery

GET /cow/info uses normal COW authentication and Origin checks. It returns schema 1, service context-cow, startup model and execution-profile identities, EOS token IDs, and whether text serving and durable inline grammar admission are configured. It does not admit work, take a GPU lease, create history or export model paths.

Normal serve.cow_http startup supplies the public fields from its validated active profile. An embedded server without these fields returns 503. EOS-less token-only configurations retain their previous behavior and do not offer discovery. The fields are copied at startup, preventing mutation through the caller's dictionary.

BranchWrite dev29 uses this for branchwrite connect ADDRESS. An older client still uses its explicit connection file. This change does not relax source-closure validation or change an existing checkpoint execution identity. A live startup from changed source still needs its exact runtime source closure validated.

Tests: serve/test_cow_connection_info.py and BranchWrite tests/test_client_setup_cow.py. These are transport/setup checks, not native numerical or GPU qualification.
