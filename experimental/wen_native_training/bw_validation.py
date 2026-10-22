"""Execute owned BW cases in isolated workspaces; no bundled task answers."""
import ast
import hashlib


def verify(case, text, directory):
    from branchwrite.execution import Executor, Grant
    from branchwrite.tool_registry import ToolRegistry
    directory.mkdir(parents=True,exist_ok=False)
    source = directory/'source'; source.mkdir()
    if case['kind'] == 'edit':
        (source/case['path']).write_text(case['original'],encoding='utf-8',newline='')
    elif case['kind'] == 'read':
        (source/case['path']).write_text('value = 1\n',encoding='utf-8',newline='')
    executor = Executor.create(source,directory/'execution')
    grant = Grant((case['path'],) if case['kind'] == 'edit' else (),False,
                  create_paths=(case['path'],) if case['kind'] == 'create' else (),
                  read_paths=(case['path'],))
    registry = ToolRegistry(executor,grant)
    observations = ()
    try:
        if case['kind'] == 'edit':
            import json
            receipt = registry.execute(registry.freeze(json.dumps(dict(tool='read_file',arguments=dict(path=case['path'])))),'initial-read')
            observations = (receipt.observation,)
        tool,args = registry.parse(text)
        if case['kind'] == 'read':
            assert tool == 'read_file' and args['path'] == case['path']
            result = registry.execute(registry.freeze(text),'candidate-read')
            assert result.exit_code == 0 and result.stdout == b'value = 1\n'
            return dict(passed=True,tool=tool)
        assert tool in (('create_file','create_text_file') if case['kind'] == 'create' else ('replace','replace_text_file'))
        result = registry.execute(registry.freeze(text,observations=observations),'candidate-action')
        actual = (executor.workspace/case['path']).read_bytes()
        if case['kind'] == 'edit':
            assert ast.dump(ast.parse(actual)) == ast.dump(ast.parse(case['expected']))
            assert '# keep this marker' in actual.decode('utf-8')
        elif 'word_count' in case:
            value = actual.decode('utf-8')
            assert len(value.split()) == case['word_count'], 'wrong whitespace-delimited word count'
            assert '\n' not in value and '\r' not in value
            assert all(word in value.lower() for word in case['topic_words'])
        else:
            assert actual == case['expected'].encode('utf-8')
        return dict(passed=True,tool=tool,bytes_sha256=hashlib.sha256(actual).hexdigest())
    except (ValueError,AssertionError,SyntaxError,KeyError,TypeError) as error:
        return dict(passed=False,error=type(error).__name__+': '+str(error)[:500])
