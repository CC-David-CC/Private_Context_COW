"""Private-input CLI comparison with fresh workspaces and independent file oracles.

Linux only. The model gets user tasks and fixtures, never the oracle answers.
All runner commands execute inside bubblewrap with the user's home hidden.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def task_prompt(case):
    task = case['task']
    for phrase in ('Choose the write action now; read it back afterward.',
                   'The current file has already been read in the actual feedback below. Choose the edit action now.',
                   'Choose the file-write action now.'):
        task = task.replace(phrase, '')
    return ('Work only in the current workspace. Use your available file or terminal tools to complete '
            'the task, verify the actual result, and then report briefly. Do not merely propose a change.\n\n'+task.strip())


def verdict(case, workspace, log):
    path = workspace/case['path']
    if not path.is_file() or path.is_symlink(): return dict(passed=False, error='Missing regular task file')
    actual = path.read_bytes()
    if case['kind'] == 'read':
        # A read task is accepted only if the response reports the observed
        # fixture, not merely because our fixture exists. Runner traces remain
        # available to distinguish explicit tool reads from preloaded context.
        passed = actual == b'value = 1\n' and 'value' in log and '1' in log
    elif case['kind'] == 'edit':
        try:
            passed = (ast.dump(ast.parse(actual)) == ast.dump(ast.parse(case['expected']))
                      and b'# keep this marker' in actual)
        except SyntaxError: passed = False
    elif 'word_count' in case:
        try:
            value = actual.decode('utf-8')
            passed = (len(value.split()) == case['word_count'] and '\n' not in value and '\r' not in value
                      and all(word in value.lower() for word in case['topic_words']))
        except UnicodeError: passed = False
    else: passed = actual == case['expected'].encode('utf-8')
    return dict(passed=passed, bytes=len(actual), sha256=hashlib.sha256(actual).hexdigest(),
                word_count=len(actual.decode('utf-8', errors='replace').split()))


def sandbox(root, trial, environment, command):
    argv = ['bwrap', '--die-with-parent', '--unshare-user', '--unshare-pid', '--unshare-ipc', '--unshare-uts',
            '--ro-bind', '/', '/', '--tmpfs', '/home', '--tmpfs', '/root', '--tmpfs', '/tmp', '--tmpfs', '/run',
            '--proc', '/proc', '--dev', '/dev', '--dir', str(Path.home()),
            '--ro-bind', str(root/'tools'), str(root/'tools'),
            '--ro-bind', str(root/'bin'), str(root/'bin'),
            '--bind', str(trial/'home'), '/runner-home', '--bind', str(trial/'workspace'), '/work']
    managed_python = Path.home()/'.local/share/uv/python'
    if managed_python.exists(): argv += ['--ro-bind', str(managed_python), str(managed_python)]
    argv += ['--chdir', '/work', '--clearenv']
    base = dict(HOME='/runner-home', XDG_CONFIG_HOME='/runner-home/.config', XDG_DATA_HOME='/runner-home/.local/share',
                XDG_CACHE_HOME='/runner-home/.cache', PATH=str(root/'bin')+':/usr/local/bin:/usr/bin:/bin',
                LANG='C.UTF-8', TERM='dumb', NO_COLOR='1', SHELL='/bin/bash')
    for key, value in {**base, **environment}.items(): argv += ['--setenv', key, str(value)]
    return argv+['--']+command


def command_for(runner, root, trial, model, endpoint, key, case):
    home = trial/'home'; home.mkdir()
    (home/'task.txt').write_text(task_prompt(case), encoding='utf-8')
    env = dict(WEN_COMPAT_API_KEY=key)
    if runner == 'codex':
        config = home/'codex'; config.mkdir()
        (config/'config.toml').write_text(f'''model = "{model}"
model_provider = "wen"
model_context_window = 65536
model_auto_compact_token_limit = 55000
model_supports_reasoning_summaries = false
web_search = "disabled"
approval_policy = "never"
[features]
multi_agent = false
[model_providers.wen]
name = "Wen local COW"
base_url = "{endpoint}"
env_key = "WEN_COMPAT_API_KEY"
wire_api = "responses"
requires_openai_auth = false
supports_websockets = false
request_max_retries = 0
stream_max_retries = 0
''')
        env['CODEX_HOME'] = '/runner-home/codex'
        cmd = [str(root/'bin/codex'), 'exec', '--json', '--ephemeral', '--skip-git-repo-check',
               '--sandbox', 'danger-full-access', '-']
    elif runner == 'opencode':
        config = dict(model='wen/'+model, enabled_providers=['wen'], share='disabled', autoupdate=False,
            provider=dict(wen=dict(npm='@ai-sdk/openai-compatible', name='Wen local COW',
                options=dict(baseURL=endpoint, apiKey=key), models={model:dict(name=model,
                limit=dict(context=65536, output=2048))})),
            agent=dict(build=dict(temperature=0, steps=8)),
            tools=dict(task=False, webfetch=False, websearch=False, question=False),
            permission=dict(external_directory='deny'))
        (home/'opencode.json').write_text(json.dumps(config))
        env.update(OPENCODE_CONFIG='/runner-home/opencode.json', OPENCODE_DISABLE_AUTOUPDATE='true',
                   OPENCODE_DISABLE_SHARE='true', OPENCODE_DISABLE_DEFAULT_PLUGINS='true')
        cmd = [str(root/'bin/opencode'), 'run', '--pure', '--format', 'json', '-m', 'wen/'+model, task_prompt(case)]
    elif runner == 'hermes':
        config = home/'hermes'; config.mkdir()
        # Config is JSON, a YAML subset, avoiding a benchmark-only YAML dependency.
        (config/'config.yaml').write_text(json.dumps(dict(model=dict(default=model, provider='custom',
            base_url=endpoint, context_length=65536), agent=dict(max_turns=8, reasoning_effort='none'),
            terminal=dict(backend='local', cwd='/work'), compression=dict(enabled=False),
            memory=dict(memory_enabled=False), platform_toolsets=dict(cli=['terminal', 'file']))))
        env.update(HERMES_HOME='/runner-home/hermes', TERMINAL_CWD='/work', OPENAI_API_KEY=key,
                   HERMES_SKIP_UPDATE_CHECK='1')
        cmd = [str(root/'tools/hermes-venv/bin/hermes'), 'chat', '--oneshot', '--query-file', '/runner-home/task.txt',
               '--provider', 'custom', '--base-url', endpoint, '--api-key', key, '-m', model,
               '--toolsets', 'terminal,file', '--ignore-rules', '--format', 'stream-json',
               '--max-turns', '8', '--run-budget', '180']
    elif runner == 'aider':
        env.update(OPENAI_API_KEY=key, OPENAI_API_BASE=endpoint, AIDER_ANALYTICS='false',
                   LITELLM_LOCAL_MODEL_COST_MAP='True')
        cmd = [str(root/'tools/aider-venv/bin/aider'), '--model', 'openai/'+model, '--no-check-update',
               '--no-show-model-warnings', '--no-auto-commits', '--no-git', '--yes-always',
               '--no-auto-lint', '--no-analytics', '--no-fancy-input', '--no-pretty', '--no-stream',
               '--map-tokens', '0', '--edit-format', 'whole', '--message-file', '/runner-home/task.txt', '/work/'+case['path']]
    else: raise ValueError('Unknown runner')
    return sandbox(root, trial, env, cmd)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--cases', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True); p.add_argument('--model', required=True)
    p.add_argument('--endpoint', required=True); p.add_argument('--key-file', type=Path, required=True)
    p.add_argument('--runners', default='codex,opencode,hermes,aider')
    p.add_argument('--ids', default='preserve-ascii,preserve-identity,train-writing-tool,heldout-writing-tool')
    p.add_argument('--timeout', type=int, default=240)
    a = p.parse_args(); a.run.mkdir(parents=True, exist_ok=False)
    all_cases = json.loads(a.cases.read_text(encoding='utf-8')); wanted = a.ids.split(',')
    cases = {c['id']:c for c in all_cases}; assert set(wanted) <= cases.keys()
    key = a.key_file.read_text().strip(); rows = []
    for runner in a.runners.split(','):
        for name in wanted:
            case = cases[name]; trial = a.run/(runner+'_'+name); trial.mkdir()
            workspace = trial/'workspace'; workspace.mkdir()
            if case['kind'] == 'edit': (workspace/case['path']).write_text(case['original'], encoding='utf-8', newline='')
            elif case['kind'] == 'read': (workspace/case['path']).write_bytes(b'value = 1\n')
            argv = command_for(runner, a.root, trial, a.model, a.endpoint, key, case)
            started = time.monotonic(); timeout = False
            with (trial/'runner.log').open('w') as log:
                process = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT,
                                           text=True, start_new_session=True)
                try: process.communicate(task_prompt(case) if runner=='codex' else None, timeout=a.timeout)
                except subprocess.TimeoutExpired:
                    timeout = True; os.killpg(process.pid, signal.SIGTERM)
                    try: process.wait(timeout=10)
                    except subprocess.TimeoutExpired: os.killpg(process.pid, signal.SIGKILL); process.wait()
            log = (trial/'runner.log').read_text(errors='replace')
            check = verdict(case, workspace, log)
            # A file left before timeout is informative but not a completed run.
            completed = process.returncode == 0 and not timeout
            row = dict(runner=runner, model=a.model, id=name, seconds=time.monotonic()-started,
                       exit_code=process.returncode, timeout=timeout, completed=completed,
                       verdict=check, passed=completed and check['passed'])
            rows.append(row); (a.run/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
            print(json.dumps(row), flush=True)


if __name__ == '__main__': main()
