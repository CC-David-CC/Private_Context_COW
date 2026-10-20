"""Host-side building blocks. This example never executes model-proposed tools."""
import json
from pathlib import Path


STATES = ('READ', 'EDIT', 'TEST', 'FINISH')


def load_grammars(directory=None):
    directory = Path(directory or Path(__file__).with_name('agent_grammars'))
    return {state: {'format': 'gbnf', 'source':
                   (directory / (state.lower() + '.gbnf')).read_text(encoding='utf-8')}
            for state in STATES}


def next_state(state, result):
    """Only a trusted supervisor result can advance the host state machine.

    accepted means the host validated schema/semantics and authorized the
    action. applied and passed report actual host execution, never model text.
    None means an accepted FINISH action completed this host workflow.
    """
    if state not in STATES or not isinstance(result, dict) or type(result.get('accepted')) is not bool:
        raise ValueError('Need a known state and a trusted boolean acceptance result')
    if not result['accepted']:
        return state
    if state == 'READ':
        return 'EDIT'
    if state == 'EDIT':
        if type(result.get('applied')) is not bool:
            raise ValueError('Accepted patch needs an actual boolean applied result')
        return 'TEST' if result['applied'] else 'EDIT'
    if state == 'TEST':
        if type(result.get('passed')) is not bool:
            raise ValueError('Accepted test needs an actual boolean passed result')
        return 'FINISH' if result['passed'] else 'EDIT'
    return None


def generate_action(model, snapshot, *, engine, state, stream, input_ids=(),
                    max_tokens=256, on_token=None):
    """Generate one action. The host must validate and authorize it afterward.

    A token limit may interrupt a JSON string. Return the partial snapshot and
    no action in that case; never send partial text to a tool executor. Resume
    with model.run_state_request(..., resume=True) if the host wants more.
    input_ids are the host-formatted conversation addition (including actual
    supervisor feedback); formatting/template selection stays with the host.
    """
    result = model.run_state_request(snapshot, input_ids, max_tokens,
        engine=engine, state=state, stream=stream, on_token=on_token)
    action = None
    if result['stopped'] and result['grammar_complete']:
        raw = b''.join(model.backend.piece(token) for token in result['tokens']
                       if token not in engine.eos_tokens)
        action = json.loads(raw.decode('utf-8', errors='strict'))
    return result, action
