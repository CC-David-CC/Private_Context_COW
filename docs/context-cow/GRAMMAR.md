# Named GBNF states in COW serving

Grammar support is opt-in and uses a startup-owned inline registry. Install the optional parser dependency with the server environment:

```sh
python -m pip install -r serve/requirements-grammar.txt
```

Start the text service with both `--text` and `--grammar-registry PATH`. The registry is strict UTF-8 JSON with schema 1 and named states whose definitions contain `format: "gbnf"` and inline `source` text. It is loaded and compiled before the coordinator starts model workers. Client requests can name a configured state with `--grammar-state READ`; they cannot send grammar text or server paths.

```json
{
  "schema": 1,
  "states": {
    "READ": {"format": "gbnf", "source": "root ::= \"read\""},
    "EDIT": {"format": "gbnf", "source": "root ::= \"edit\""}
  }
}
```

For example, `cow-cli.py --url http://127.0.0.1:8099 answer --branch assistant --grammar-state READ` selects READ for that generation. A `continue` request restores the parser and state saved in canonical history; it cannot switch an active parser to EDIT. After a grammar turn completes, the host can select a new grammar on the next user turn with `append-user --grammar-state EDIT`.

The grammar constrains token syntax only. The host still validates meaning, authorizes any proposed action, and performs tool execution. Grammar support requires a model profile with the checked consumed-token protocol and matching EOS/tokenizer identity. Without `--grammar-registry`, normal service startup does not import llguidance.