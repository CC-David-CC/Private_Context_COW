"""Derive a reusable system-only prefix from Qwen's unmodified template."""
from .cow_prompt_encoder import CowPromptEncoder
from .server import Service

class Qwen35PromptEncoder(CowPromptEncoder):
    def render_prompt(self,messages,tools,kwargs):
        if (len(messages)==1 and messages[0].get('role')=='system'
                and not kwargs.get('add_generation_prompt',False) and not tools):
            prefixes=[]
            for text in ('COW prefix reference one.','A different independent prefix reference.'):
                user={'role':'user','content':text}
                complete=Service.render_prompt(self,list(messages)+[user],tools,kwargs)
                tail=Service.render_prompt(self,[user],tools,kwargs)
                if not tail or not complete.endswith(tail):
                    raise ValueError('Qwen system prefix is not independent of its user suffix')
                prefixes.append(complete[:-len(tail)])
            if not prefixes[0] or prefixes[0]!=prefixes[1]:
                raise ValueError('Qwen template has no stable reusable system prefix')
            return prefixes[0]
        return Service.render_prompt(self,messages,tools,kwargs)
