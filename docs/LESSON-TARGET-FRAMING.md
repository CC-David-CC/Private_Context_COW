# Template framing in exact lesson targets

Schema2 lessons may label a rendered template-owned assistant span as
`kind: "framing"`, `role: "assistant"`, `verified: false`, `scope: "practice"`.
It remains in the exact rendered token stream and is always masked from loss,
under both existing target policies. Verified answer/action/reasoning rules stay
the same. A framing span cannot supply the required verified target.

This lets an artifact renderer select the exact checked answer and its native
end token while retaining a template newline after that token as context.
Labelling a span does not establish that it came from the template: the owner
provider must independently verify rendering, artifact bytes, EOS, framing and
mask boundaries before registering a lesson. The renderer version and installed
validator source must be included in the provider/policy identity.

Existing lessons without framing retain the same token and loss-mask behavior.
A server without this addition refuses the new kind. The addition does not grant
lesson approval, execute training or change any model/candidate automatically.
