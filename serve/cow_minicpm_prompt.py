"""MiniCPM template framing over immutable consumed COW input history."""
from .cow_prompt_encoder import CowPromptEncoder
from .server import Service


class MiniCPMPromptEncoder(CowPromptEncoder):
    """Keep the template BOS at the root and explicitly close model-EOS turns.

    MiniCPM can stop on </s> as well as ChatML im_end. Appending a tool result
    never removes either consumed token or re-renders the retained assistant.
    """
    def __init__(self, tokenizer, template, template_kwargs=None):
        super().__init__(tokenizer, template, template_kwargs)
        self.tool_binding = getattr(template, "tool_binding", None)
        self.tool_family = "minicpm5"

    def _control(self,text):
        ids=tuple(self.tok.encode(text,parse_special=True))
        if len(ids)!=1:raise ValueError('MiniCPM requires a single '+text+' control token')
        return ids

    def render_prompt(self,messages,tools,kwargs):
        # The actual GGUF template begins with {{ bos_token }}. General GPT-2
        # add_bos_token is false, so supply the template variable explicitly.
        self._control('<s>')
        return Service.render_prompt(self,messages,tools,dict(kwargs,bos_token='<s>'))

    def _user_frame(self,content):
        if not isinstance(content,str):raise ValueError('New user content must be plain text')
        thinking=self._user_suffix_thinking()
        anchor=[dict(role='system',content='COW MiniCPM suffix boundary reference.')]
        complete=anchor+[dict(role='user',content=content)]
        prefix=self.render(anchor,add_generation_prompt=False)
        rendered=self.render(complete,add_generation_prompt=True)
        prefix_ids=tuple(self.encode_rendered(anchor,prefix,False))
        ids=tuple(self.encode_rendered(complete,rendered,True))
        if not prefix or not rendered.startswith(prefix) or ids[:len(prefix_ids)]!=prefix_ids:
            raise ValueError('MiniCPM root prefix is not stable across its new-user frame')
        frame=ids[len(prefix_ids):];text=rendered[len(prefix):]
        tail='<think>\n' if thinking else '<think>\n\n</think>\n\n'
        expected=('<|im_start|>user\n'+content.strip()+'<|im_end|>\n<|im_start|>assistant\n'+tail)
        if text!=expected:raise ValueError('MiniCPM new-user template frame differs from its qualified layout')
        start,end=self._control('<|im_start|>'),self._control('<|im_end|>')
        if (frame.count(start[0])!=2 or frame.count(end[0])!=1
                or self._control('<s>')[0] in frame or self._control('</s>')[0] in frame):
            raise ValueError('MiniCPM user data introduced an unexpected control token')
        return frame

    def append_user_suffix(self,canonical,controller,content):
        eos=self._control('</s>');end=self._control('<|im_end|>')
        if (canonical and canonical[-1]==eos[0] and isinstance(controller,dict)
                and controller.get('status')=='stopped'
                and eos[0] in controller.get('policy',{}).get('eos_tokens',[])):
            # Preserve the already consumed model EOS, complete its chat frame,
            # then append the new user frame. No existing history is replaced.
            return end+tuple(self.tok.encode('\n',parse_special=False))+self._user_frame(content)
        return super().append_user_suffix(canonical,controller,content)
